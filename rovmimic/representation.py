"""SuperMimic 的水下版状态表示和世界模型（见 readme/system_overview.md 4.3、4.4）。

  ROVRepresentation  状态换算、积分、损失分组、传感器读数、目标；奖励和失败判定在 rovmimic/tracking.py
  ROVDynamics        网络只预测航行器 6 个速度变化量；机械臂和夹爪按舵机公式（rovmimic/terms.arm_step）解析推进，
                     所以 apply_prediction 需要这一步的动作（原版 Dynamics.forward 不传动作，这里重写 forward）
  NoContextSpec      不做域随机化时的占位仿真参数（SuperMimic 要求至少 1 维）
  build_rov_world_model  对应原版 runners/supermimic_runner.build_world_model
"""

from __future__ import annotations

import math

import torch
from supermimic.models import Dynamics
from supermimic.representations import StateRepresentation, _root_loss_blocks
from supermimic.runners.supermimic_runner import WorldModel
from supermimic.sim_context import SimContext, SimContextSpec

from rovmimic import servo as S
from rovmimic import state as ST
from rovmimic.terms import arm_step

NQ = len(S.JOINT_NAMES)

# 网络输出的尺度：Δv_b = 输出[0:3] × 尺度，Δw_b = 输出[3:6] × 尺度。
# 取参考轨迹里每步速度变化的标准差量级（训练集 600 条片段，2026-11-16）：
#   Δv_b std (0.0017, 0.0013, 0.0017) m/s，Δw_b std (0.0034, 0.0010, 0.0099) rad/s。
# 这台航行器在仿真里很重（平移等效约 166 kg），每步速度只变 1e-3 量级；网络直接输出时，
# Adam 每步更新带来的输出抖动约 1e-2，比信号还大，离线训练时比“速度保持不变”还差。
# 原版人形每步速度变化 0.1–1 m/s，不需要这一项。
PREDICTION_SCALE = (0.002, 0.002, 0.002, 0.004, 0.002, 0.01)


class ROVRepresentation(StateRepresentation):
  """full_state (27) / rep_state (24) / 预测 (6) 的布局见 rovmimic/state.py。"""

  LOSS_WEIGHT_CONFIG = {
    **StateRepresentation.LOSS_WEIGHT_CONFIG,
    "root_lin_vel": "nominal_error_root_lin_vel",
    "root_ang_vel": "nominal_error_root_ang_vel",
  }

  def __init__(self, proprio_half_widths: list[float], prediction_scale=PREDICTION_SCALE):
    super().__init__()
    self.num_joints = NQ
    self.full_state_dim = 13 + 2 * NQ
    self.rep_state_dim = 10 + 2 * NQ
    self.prediction_dim = 6
    self.actor_obs_mode = "proprio"
    self.register_buffer("_proprio_noise_bounds", torch.tensor(proprio_half_widths), persistent=False)
    self.register_buffer("_prediction_scale", torch.tensor(prediction_scale), persistent=False)
    self.fk = None

  def bind(self, env, sim_context) -> None:
    self.fk = env.command_manager.get_term("motion").motion.fk
    lim = torch.tensor([S.LIMITS[n] for n in S.JOINT_NAMES], device=self._proprio_noise_bounds.device)
    self._soft_joint_pos_limits = lim

  # ---------------------------------------------------------------- 状态

  def represent(self, full_state: torch.Tensor) -> torch.Tensor:
    return ST.represent(full_state, NQ)

  def extract_aux_state(self, full_state: torch.Tensor) -> torch.Tensor:
    return full_state[..., :7]

  def apply_prediction(self, aux_state, curr_rep_state, prediction, dt, action=None):
    """航行器：prediction × PREDICTION_SCALE = [Δv_b, Δw_b]（当前机体系），按半隐式积分推位姿；
    机械臂和夹爪：舵机公式，q_next = arm_step(q, action)，角速度 = 这一步的平均速度。"""
    if action is None:
      raise ValueError("ROVRepresentation.apply_prediction needs the action (use ROVDynamics)")
    _, _, v_b, w_b, qj, _ = ST.split_rep(curr_rep_state, NQ)
    delta = prediction * self._prediction_scale
    next_aux, g_b, nv_b, nw_b = ST.integrate_vehicle(aux_state, v_b, w_b, delta[..., 0:3], delta[..., 3:6], dt,
                                                     self.substep_coeff)
    q_next, vj_next = arm_step(qj, action, dt)
    next_rep = torch.cat([next_aux[..., 2:3], g_b, nv_b, nw_b, q_next, vj_next], -1)
    return next_aux, next_rep

  def reconstruct_full_state(self, aux_state, rep_state):
    return ST.full_from(aux_state, rep_state, NQ)

  def joint_state(self, rep_state):
    _, _, _, _, qj, vj = ST.split_rep(rep_state, NQ)
    return torch.cat([qj, vj], -1)

  def loss_blocks(self, rep_state, aux_state):
    """积分出的航行器位姿 + 机体系速度。关节由公式算，不进损失。"""
    _, _, v_b, w_b, _, _ = ST.split_rep(rep_state, NQ)
    return {**_root_loss_blocks(aux_state), "root_lin_vel": v_b, "root_ang_vel": w_b}

  def world_frame_bodies(self, aux_state, rep_state):
    pos, quat, lin, ang = self.fk(*ST.split_full(self.reconstruct_full_state(aux_state, rep_state), NQ))
    return pos, ST.matrix_from_quat(quat), lin, ang

  def compute_wm_reward_from_reference(self, *args, **kwargs):
    raise NotImplementedError("ROV reward needs the reference joint targets; see ROVWorldModelEnv._wm_step")

  # ---------------------------------------------------------------- 传感器与目标

  def compute_wm_obs(self, rep_state, aux_state, joint_state, sim_ctx):
    """[g_b, w_b, v_b(DVL), qj, vj]，与仿真环境 obs["actor"] 的前 23 维同一个函数。"""
    return ST.sensor(rep_state, NQ)

  def proprio_term_widths(self, num_actions: int) -> list[int]:
    return [*ST.sensor_widths(NQ), num_actions]

  def proprio_noise_bounds(self, num_actions, device):
    return self._proprio_noise_bounds.to(device)

  def compute_wm_goal(self, motion, motion_times, aux_state):
    """策略和世界模型用同一种目标（含相对位置）；策略那份由调用方传入带测量误差的位置。"""
    g = ST.goal(motion, motion_times, aux_state[..., :3], aux_state[..., 3:7])
    return g, g


class ROVDynamics(Dynamics):
  """动作 clip 到 [-1, 1] 后进网络（与仿真里实际生效的一致）；机械臂公式也拿到这一步的动作。"""

  def proc(self, action: torch.Tensor) -> torch.Tensor:
    return action.clamp(-1.0, 1.0)

  def forward(self, rep_state, action, aux_state, sim_ctx):
    state_norm = self.state_normalizer(rep_state)
    action_norm = self.action_normalizer(self.proc(action))
    prediction = self.mlp(torch.cat([state_norm, action_norm, sim_ctx], dim=-1))
    return self._apply_prediction(aux_state, rep_state, prediction, self.dt, action)


class NoContextSpec(SimContextSpec):
  name = "none"
  dim = 1

  def bind(self, env):
    return [(0.0, 0.0)]

  def read(self, env):
    return torch.zeros(env.num_envs, 1, device=env.device)


def build_rov_world_model(cfg, env, device: str) -> WorldModel:
  obs, _ = env.reset()
  num_goals = obs["goal"].shape[-1]
  num_actions = env.num_actions
  spec = NoContextSpec()
  sim_context = SimContext([spec], spec.bind(env)).to(device)
  rep = ROVRepresentation(env.cfg.sensors.proprio_half_widths(NQ, num_actions)).to(device)
  rep.dt = env.step_dt
  dec = env.cfg.decimation
  rep.substep_coeff = (dec + 1) / (2 * dec)  # 多个物理子步在恒定加速度下合成
  rep.bind(env, sim_context)
  dynamics = ROVDynamics(rep, num_actions, sim_context.output_dim, torch.ones(1, num_actions, device=device),
                         torch.zeros(1, num_actions, device=device), hidden_dims=cfg.dynamics_hidden_dims).to(device)
  print(f"[INFO] ROV world model: full={rep.full_state_dim}, rep={rep.rep_state_dim}, pred={rep.prediction_dim}, "
        f"goal={num_goals}, actions={num_actions}, substep_coeff={rep.substep_coeff}")
  assert math.isclose(rep.dt, S.CTRL_DT)
  return WorldModel(representation=rep, dynamics=dynamics, sim_context=sim_context, num_goals=num_goals,
                    num_actions=num_actions)
