"""跟踪奖励和失败判定（纯 PyTorch，可微、可批量）。

仿真环境用 Stonefish 的真实状态算，世界模型用预测状态算，调用的是同一个 evaluate，
所以两边的奖励和失败判定在同一状态下完全相同。权重和阈值只在这里定义一次（环境配置里引用它们）。

和 SuperMimic / BeyondMimic 原版的区别（详见 readme/system_overview.md 4.5）：
  - 航行器位置权重 0.5 -> 1.0；新增 TCP 世界系位置项（不对齐，权重 2.0，std 0.1 m）；
  - 姿势项（臂上连杆位置、姿态）先把参考整体对齐到当前航行器的位置和姿态（6 自由度）再比，
    只比“机械臂摆得像不像”；航行器自身的位置姿态误差由航行器项和 TCP 项负责。
    原版只对齐水平位置和偏航；
  - 速度项的 std 按水下的速度量级缩小（线速度 1.0 -> 0.3 m/s，角速度 3.14 -> 1.0 rad/s）；
  - 新增夹爪开合项、动作越界惩罚（|a| > 1 的部分没有梯度，要把策略拉回范围内）、艇体侵入惩罚；
  - 关节超限惩罚 -2 -> -10（仿真里限位不生效）；
  - 失败判定：航行器三维位置误差、姿态转角（含偏航）、机械臂侵入艇体。原版只看高度和倾斜。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from rovmimic import motion as M
from rovmimic.fk import quat_apply, quat_mul
from rovmimic.state import quat_apply_inverse, quat_conj, quat_error_angle, split_full
from rovmimic.terms import hull_intrusion, joint_limit_violation


@dataclass
class RewardTermCfg:
  """跟踪项：weight · exp(-误差² / std²)；惩罚项（weight < 0）：weight · 惩罚量。"""
  weight: float
  std: float = 1.0


@dataclass
class TerminationTermCfg:
  params: dict = field(default_factory=dict)
  time_out: bool = False


KERNEL_TERMS = ("root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "tcp_pos", "gripper")
PENALTY_TERMS = ("action_rate", "action_bound", "joint_limit", "hull")


def default_rewards() -> dict[str, RewardTermCfg]:
  return {
    "root_pos": RewardTermCfg(1.0, 0.3),       # 航行器位置，世界系 (m)
    "root_ori": RewardTermCfg(0.5, 0.4),       # 航行器姿态转角 (rad)
    "body_pos": RewardTermCfg(1.0, 0.1),       # 臂上连杆位置，对齐后 (m)
    "body_ori": RewardTermCfg(1.0, 0.4),       # 臂上连杆姿态，对齐后 (rad)
    "body_lin_vel": RewardTermCfg(1.0, 0.3),   # 航行器 + 臂上连杆线速度，世界系 (m/s)
    "body_ang_vel": RewardTermCfg(1.0, 1.0),   # 角速度 (rad/s)
    "tcp_pos": RewardTermCfg(2.0, 0.1),        # TCP 位置，世界系，不对齐 (m)
    "gripper": RewardTermCfg(0.5, 0.003),      # 夹爪 axis_a (m)
    "action_rate": RewardTermCfg(-0.1),        # Σ(a - a_prev)²
    "action_bound": RewardTermCfg(-1.0),       # Σ relu(|a| - 1)²
    "joint_limit": RewardTermCfg(-10.0),       # 超限量 (rad / m)
    "hull": RewardTermCfg(-20.0),              # 侵入艇体包围盒的深度 (m)
  }


def default_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(time_out=True),
    "motion_end": TerminationTermCfg(time_out=True),
    "anchor_pos": TerminationTermCfg(params={"threshold": 0.5}),   # 航行器三维位置误差 (m)
    "anchor_ori": TerminationTermCfg(params={"threshold": 0.8}),   # 航行器姿态转角 (rad)
    "hull": TerminationTermCfg(params={"threshold": 0.03}),        # 侵入艇体 (m)
  }


FAIL_TERMS = ("anchor_pos", "anchor_ori", "hull")


def evaluate(full: torch.Tensor, action: torch.Tensor, last_action: torch.Tensor, fk,
             ref_bodies: tuple[torch.Tensor, ...], ref_joint_pos: torch.Tensor,
             rewards: dict[str, RewardTermCfg], dt: float):
  """full (B, 27) 是这一步之后的状态，ref_* 是它应该到达的参考帧（ROVMotion.body_state / joint_pos）。
  action / last_action 是策略原始输出 (B, 13)。

  返回 (reward (B,)，terms {名字: (B,) 未乘权重的核值或惩罚量}，errors {anchor_pos, anchor_ori, hull: (B,)})。
  reward = dt · (Σ 跟踪项 weight·核值 + Σ 惩罚项 weight·惩罚量)，与 mjlab 的奖励管理器一样乘 dt。"""
  root_pos, root_quat, qj, lin_w, ang_w, vj = split_full(full)
  pos, quat, lin, ang = fk(root_pos, root_quat, qj, lin_w, ang_w, vj)
  r_pos, r_quat, r_lin, r_ang = ref_bodies

  # 参考整体对齐到当前航行器位姿：T = T_root · T_ref_root⁻¹
  q_delta = quat_mul(root_quat, quat_conj(r_quat[..., 0, :]))
  arm_ref_pos = r_pos[..., M.ARM_TRACK, :]
  n_arm = arm_ref_pos.shape[-2]
  qd = q_delta.unsqueeze(-2).expand(*q_delta.shape[:-1], n_arm, 4)
  tgt_pos = root_pos.unsqueeze(-2) + quat_apply(qd, arm_ref_pos - r_pos[..., 0:1, :])
  tgt_quat = quat_mul(qd, r_quat[..., M.ARM_TRACK, :])

  root_sq = ((root_pos - r_pos[..., 0, :]) ** 2).sum(-1)
  root_ang = quat_error_angle(root_quat, r_quat[..., 0, :])
  sq = {
    "root_pos": root_sq,
    "root_ori": root_ang ** 2,
    "body_pos": ((pos[..., M.ARM_TRACK, :] - tgt_pos) ** 2).sum(-1).mean(-1),
    "body_ori": (quat_error_angle(quat[..., M.ARM_TRACK, :], tgt_quat) ** 2).mean(-1),
    "body_lin_vel": ((lin[..., M.TRACK, :] - r_lin[..., M.TRACK, :]) ** 2).sum(-1).mean(-1),
    "body_ang_vel": ((ang[..., M.TRACK, :] - r_ang[..., M.TRACK, :]) ** 2).sum(-1).mean(-1),
    "tcp_pos": ((pos[..., M.TCP_IDX, :] - r_pos[..., M.TCP_IDX, :]) ** 2).sum(-1),
    "gripper": (qj[..., 4] - ref_joint_pos[..., 4]) ** 2,
  }
  q_root = root_quat.unsqueeze(-2).expand(*root_quat.shape[:-1], pos[..., M.HULL, :].shape[-2], 4)
  hull = hull_intrusion(quat_apply_inverse(q_root, pos[..., M.HULL, :] - root_pos.unsqueeze(-2)))
  terms = {k: torch.exp(-v / rewards[k].std ** 2) for k, v in sq.items()}
  terms["action_rate"] = ((action - last_action) ** 2).sum(-1)
  terms["action_bound"] = (torch.relu(action.abs() - 1.0) ** 2).sum(-1)
  terms["joint_limit"] = joint_limit_violation(qj)
  terms["hull"] = hull
  reward = dt * sum(rewards[k].weight * v for k, v in terms.items())
  # 软失败判定会进生存概率，梯度要经过它：开方加一个小量，避免零点的梯度发散
  errors = {"anchor_pos": torch.sqrt(root_sq + 1e-12), "anchor_ori": root_ang, "hull": hull,
            "tcp_pos": torch.sqrt(sq["tcp_pos"] + 1e-12)}
  return reward, terms, errors


def fail_flags(errors: dict[str, torch.Tensor], terminations: dict[str, TerminationTermCfg]) -> dict[str, torch.Tensor]:
  """仿真里的硬判定：误差超过阈值。"""
  return {k: errors[k] > terminations[k].params["threshold"] for k in FAIL_TERMS}


def fail_prob(errors: dict[str, torch.Tensor], terminations: dict[str, TerminationTermCfg],
              softness: float) -> torch.Tensor:
  """世界模型里的软判定：sigmoid((max_k 误差_k / 阈值_k - 1) / softness)，与 SuperMimic 的形式相同。"""
  ratio = torch.stack([errors[k] / terminations[k].params["threshold"] for k in FAIL_TERMS], -1).amax(-1)
  return torch.sigmoid((ratio - 1.0) / softness)


def max_step_reward(rewards: dict[str, RewardTermCfg], dt: float) -> float:
  return dt * sum(t.weight for t in rewards.values() if t.weight > 0)
