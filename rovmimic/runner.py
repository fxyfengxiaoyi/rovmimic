"""水下版 SuperMimic 训练循环：继承 SuperMimicRunner，采集、世界模型训练、策略训练、按回报重置采样、
日志和存档都沿用原版，只替换下面几处：

  _init_common        构建 ROV 世界模型（rovmimic/representation.py）
  _init_policy        策略只有“动作跟踪”一个任务，目标里含相对参考位置（原版策略看不到）；想象环境用 ROVWorldModelEnv
  _build_sim_obs_td   真实仿真里策略的目标用带测量误差的位置算（误差由环境维护，见 rovmimic/env.py）

  _create_log_dir     另存 config.json（推理 / 评估按它重建策略网络），并接管存档回调：
                      原版每次存档都覆盖日志目录里的 actor.pt / world_model.pt / critic.pt，
                      这里每 snapshot_interval 个真实步另存一份到 checkpoints/it_XXXXXXX/，方便回头评估训练过程

原版的 ONNX 导出依赖 mjlab 环境，这里关掉；部署用 rovmimic/policy.py 导出 TorchScript / ONNX。
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import supermimic.runners.supermimic_runner as SMR
from supermimic.learners.policy_learner import (ActorCriticPolicyLearner, ActorCriticPolicyLearnerCfg,
                                                BpttPolicyLearner)
from supermimic.learners.wm_learner import WorldModelLearner, WorldModelLearnerCfg
from supermimic.models import MultiEncoderAE, build_mlp
from supermimic.replay_buffer import ReplayBuffer
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.tasks.task_spec import TaskSpec, build_actor_obs_td

from rovmimic.representation import build_rov_world_model
from rovmimic.wm_env import ROVWorldModelEnv


def _no_onnx_export(**kwargs):
  print("[WARN] ONNX export is not implemented for the ROV yet; skipped")
  return []


_log_model_artifact = SMR.log_model_artifact


def _log_model_artifact_online_only(files):
  """离线 / 关闭 W&B 时跳过：离线模式下每次都会在本地多存一份存档副本，旧版本又删不掉（清理要联网）。
  存档本身照常写在日志目录里。"""
  import wandb

  if wandb.run is None or getattr(wandb.run, "offline", False) or getattr(wandb.run, "disabled", False):
    return
  _log_model_artifact(files)


SMR.export_actor_as_onnx = _no_onnx_export
SMR.log_model_artifact = _log_model_artifact_online_only


@dataclass
class ROVWorldModelLearnerCfg(WorldModelLearnerCfg):
  # 机体系速度的名义误差（Bryson 规则：损失权重 = 1 / 名义误差²），量级与 8 步 0.16 s 内的位姿名义误差相当
  nominal_error_root_lin_vel: float = 0.05
  nominal_error_root_ang_vel: float = 0.1


@dataclass
class ROVRunnerCfg(SuperMimicRunnerCfg):
  wandb_project: str = "rovmimic"
  actor_obs: str = "proprio"
  wm: ROVWorldModelLearnerCfg = field(default_factory=ROVWorldModelLearnerCfg)
  slot_mix: dict[str, float] = field(default_factory=lambda: {"motion": 1.0})
  wm_init: str | None = None
  """本地 world_model.pt（如 tools/fit_wm_refs.py 的输出），用来初始化世界模型（含输入归一化统计）。"""
  max_iterations: int = 50_000
  """真实步数（所有环境各走一步算一次）。4096 环境时约 2 万次更新，A6000 上约 9 小时。"""
  snapshot_interval: int = 5000
  """每隔多少个真实步把存档另存到 checkpoints/it_XXXXXXX/（须是 save_interval 的整数倍；0 表示不另存）。"""


class ROVRunner(SuperMimicRunner):
  cfg: ROVRunnerCfg

  def _init_common(self) -> None:
    if self.cfg.actor_obs != "proprio" or self.cfg.history_length != 1:
      raise ValueError("ROVRunner supports actor_obs='proprio' with history_length=1")
    wm = build_rov_world_model(self.cfg, self.env, self.device)
    self._wm = wm
    self._representation = wm.representation
    self._dynamics = wm.dynamics
    self._sim_context = wm.sim_context
    self._num_goals = wm.num_goals
    self._num_actions = wm.num_actions
    if self.cfg.wandb_run_path is not None:
      SMR.load_wandb_checkpoint(self.cfg.wandb_run_path, wm, self.device)
    if self.cfg.wm_init is not None:
      bundle = torch.load(self.cfg.wm_init, map_location=self.device)
      wm.dynamics.load_state_dict(bundle["dynamics"])
      print(f"[INFO] world model initialized from {self.cfg.wm_init}")
    self.action_manager = self.env.unwrapped.action_manager
    self.motion_cmd = self.env.unwrapped.command_manager.get_term("motion")
    self.motion = self.motion_cmd.motion
    self.replay_buffer = ReplayBuffer(schema=self._buffer_schema(), buffer_size=self.cfg.buffer_size,
                                      num_envs=self.env.num_envs, device=self.device)
    self.world_model_learner = WorldModelLearner(self._dynamics, self._representation, self.cfg.wm,
                                                 device=self.device)
    self._pending_norm_updates = []
    self._sim_history = None

  def _create_log_dir(self, *extra_parts: str) -> Path:
    log_dir = super()._create_log_dir(*extra_parts)
    self._log_dir, self._num_saves = log_dir, 0
    if self.cfg.snapshot_interval % self.cfg.save_interval:
      raise ValueError("snapshot_interval must be a multiple of save_interval")
    cfg = {"task_id": self.task_id, "env_cfg": asdict(self.env.cfg), "runner_cfg": asdict(self.cfg)}
    (log_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    SMR.log_model_artifact = self._on_checkpoint  # run() 里每次存档后调用
    return log_dir

  def _on_checkpoint(self, files) -> None:
    """存档写完后调用（在上传线程里；主循环下一次存档前会等它结束，所以文件不会被同时覆盖）。"""
    cfg = self.cfg
    it = self._num_saves * cfg.save_interval
    self._num_saves += 1
    last = it + cfg.save_interval >= cfg.max_iterations
    if cfg.snapshot_interval > 0 and (it % cfg.snapshot_interval == 0 or last):
      d = self._log_dir / "checkpoints" / f"it_{it:07d}"
      d.mkdir(parents=True, exist_ok=True)
      for f in files:
        shutil.copy2(f, d / Path(f).name)
      shutil.copy2(self._log_dir / "config.json", d / "config.json")
    _log_model_artifact_online_only(files)

  def _motion_task(self) -> TaskSpec:
    rep = self._representation

    def extract(motion, motion_times, aux_state):
      return rep.compute_wm_goal(motion, motion_times, aux_state)[1]

    return TaskSpec(name="motion", goal_dim=self._num_goals, extract_training_goal=extract)

  def _init_policy(self) -> None:
    cfg, rep = self.cfg, self._representation
    task = self._motion_task()
    actor = MultiEncoderAE(obs_dim=rep.proprio_obs_dim(self._num_actions), action_dim=self._num_actions,
                           task_specs=[task], slot_mix=cfg.slot_mix, latent_dim=cfg.latent_dim,
                           hidden_dims=cfg.actor_hidden_dims, latent_bc_weight=cfg.latent_bc_weight,
                           history_length=cfg.history_length).to(self.device)
    self._active_task_specs = [task]
    critic_dim = rep.rep_state_dim + self._num_goals + self._sim_context.output_dim
    critic = build_mlp(critic_dim, cfg.critic_hidden_dims, output_dim=1).to(self.device)
    print(f"[INFO] actor: obs {rep.proprio_obs_dim(self._num_actions)} + goal {task.goal_dim}; critic: {critic_dim}")
    policy_cfg = cfg.policy
    self.wm_env = ROVWorldModelEnv(dynamics=self._dynamics, representation=rep, env=self.env,
                                   batch_size=policy_cfg.batch_size, replay_buffer=self.replay_buffer,
                                   task_specs=[task], history_length=cfg.history_length,
                                   termination_softness=cfg.termination_softness, device=self.device)
    if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
      self.wm_env.step = torch.compile(self.wm_env.step)
      self.wm_env.reset = torch.compile(self.wm_env.reset)
    self._configure_reset_sampling()
    if isinstance(policy_cfg, ActorCriticPolicyLearnerCfg):
      self.policy_learner = ActorCriticPolicyLearner(actor, critic, self.wm_env, policy_cfg, device=self.device)
    else:
      self.policy_learner = BpttPolicyLearner(actor, self.wm_env, policy_cfg, device=self.device)

  def _build_sim_obs_td(self, rep_state, aux_state, sim_actor_obs, motion_times, sim_ctx):
    sensor = self._representation.compute_actor_sensor(rep_state, sim_ctx, aux_state=aux_state,
                                                        sim_actor_obs=sim_actor_obs, noisy=True)
    noise = self.env.position_noise().to(aux_state.device)
    measured = torch.cat([aux_state[:, :3] + noise, aux_state[:, 3:7]], -1)
    goals = {"motion": self._representation.compute_wm_goal(self.motion, motion_times, measured)[1]}
    return build_actor_obs_td(goals, self._collection_history(sensor))
