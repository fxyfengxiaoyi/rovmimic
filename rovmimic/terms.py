"""世界模型 / 奖励里用到的解析公式（PyTorch，可微、可批量）。

  arm_step        舵机解析模型：由动作和当前关节角算下一步关节角与“上一步平均速度”
  joint_limit_violation  关节超限量（仿真里限位不生效，靠它惩罚）
  hull_intrusion  机械臂检查点进入艇体包围盒的深度（仿真不做臂与艇体的碰撞检测，靠它惩罚）
约定见 rovmimic/servo.py。
"""

from __future__ import annotations

import torch

from rovmimic import servo as S

# 艇体包围盒（base_link 系，前-右-下，略放大），与 tools/usim_make_refs.py 的 HULL 相同
HULL = ((-0.26, 0.23), (-0.31, 0.31), (-0.16, 0.13))
# 艇体检查用的连杆和附着点（附着点需在 FK 的 sites 里定义）
HULL_CHECK_BODIES = ("m2_joint_link", "m2_1_2_link", "m1_link", "ee_base_link", "rs1_130_link", "rs1_139_link",
                     "tcp", "fingertip")
HULL_SITES = {"fingertip": ("ee_base_link", (0.0, 0.0, 0.085))}


def _index(names: tuple[str, ...], which: tuple[str, ...]) -> list[int]:
  return [names.index(n) for n in which]


# arm_step 依赖的列布局：臂关节 0-3，夹爪 4，两根手指 5-6
assert _index(S.JOINT_NAMES, S.ARM) == [0, 1, 2, 3] and S.JOINT_NAMES.index("axis_a") == 4
assert sorted(_index(S.JOINT_NAMES, S.FINGERS)) == [5, 6]
assert S.VMAX["rs1_130_joint"] == S.VMAX["rs1_139_joint"]

_CONSTS = {
  "vmax_arm": [S.VMAX[n] for n in S.ARM],
  "limit_lo": [S.LIMITS[n][0] for n in S.JOINT_NAMES],
  "limit_hi": [S.LIMITS[n][1] for n in S.JOINT_NAMES],
  "hull_lo": [h[0] for h in HULL],
  "hull_hi": [h[1] for h in HULL],
}
_cache: dict[tuple, torch.Tensor] = {}


def _const(name: str, like: torch.Tensor) -> torch.Tensor:
  """常量张量按 (dtype, device) 缓存：CUDA graph 录制期间不能再从主机拷数据。"""
  key = (name, like.dtype, like.device)
  if key not in _cache:
    with torch.inference_mode(False):  # 同 fk.py：在 inference_mode 里第一次创建时也要是普通张量
      _cache[key] = torch.tensor(_CONSTS[name], dtype=like.dtype, device=like.device)
  return _cache[key]


def arm_step(q: torch.Tensor, action: torch.Tensor, dt: float = S.CTRL_DT) -> tuple[torch.Tensor, torch.Tensor]:
  """q (…, 7，状态顺序)，action (…, 13) -> (q_next, qd_next)，qd_next = (q_next - q) / dt。

  臂关节 q + vmax·dt·clip(a, -1, 1)，对 a 的梯度恒为 vmax·dt（|a| < 1 时）；
  夹爪和手指 q + clip(目标 - q, ±vmax·dt)，目标由夹爪动作换算。
  只用切片和拼接（不用列表索引），可以放进 torch.compile 和 CUDA graph。"""
  a = action[..., S.NUM_THRUSTERS:].clamp(-1.0, 1.0)
  dq_arm = _const("vmax_arm", q) * dt * a[..., 0:4]
  grip = (a[..., 4:5] + 1.0) / 2.0 * S.GRIPPER_RANGE[1]
  step_a = S.VMAX["axis_a"] * dt
  dq_grip = (grip - q[..., 4:5]).clamp(-step_a, step_a)
  step_f = S.VMAX["rs1_130_joint"] * dt
  dq_fing = (S.JAW_RATIO * grip - q[..., 5:7]).clamp(-step_f, step_f)
  dq = torch.cat([dq_arm, dq_grip, dq_fing], -1)
  return q + dq, dq / dt


def joint_limit_violation(q: torch.Tensor) -> torch.Tensor:
  """各关节超出限位的量之和 (…,)，单位 rad / m。"""
  lo, hi = _const("limit_lo", q), _const("limit_hi", q)
  return (torch.relu(lo - q) + torch.relu(q - hi)).sum(-1)


def hull_intrusion(points_b: torch.Tensor) -> torch.Tensor:
  """points_b (…, P, 3) 是机体系下的检查点 -> 进入艇体包围盒最深的距离 (…,)，没进入为 0。

  对每个点取到包围盒六个面中最近一面的距离（在盒内为正），再对所有点取最大值。"""
  lo, hi = _const("hull_lo", points_b), _const("hull_hi", points_b)
  depth = torch.minimum(points_b - lo, hi - points_b).min(-1).values
  return torch.relu(depth).amax(-1)
