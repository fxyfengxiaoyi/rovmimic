"""状态向量的换算（纯 PyTorch，可微、可批量；仿真环境和世界模型共用同一份）。

坐标系：Stonefish 的 NED 世界系（z 向下）和 FRD 机体系，不做任何转换。
  full_state (27) = [root_pos 3, root_quat wxyz 4, qj 7, root_lin_vel_w 3, root_ang_vel_w 3, vj 7]
      与 SuperMimic gencoord 的 full_state 布局相同；vj 是上一步平均速度（rovmimic/servo.py）。
  rep_state (24)  = [depth 1, g_b 3, v_b 3, w_b 3, qj 7, vj 7]
      depth = root_pos z（向下为正）；g_b = 重力方向（世界 +z）在机体系的表示；
      v_b、w_b = 机体系线速度、角速度。
  sensor (23)     = [g_b 3, w_b 3, v_b 3, qj 7, vj 7]   IMU、DVL、编码器的读数（不含噪声）
  goal (230)      = [10 帧 × (qj, vj) 140, 10 帧相对位置（机体系）30, 10 帧相对姿态 rot6d 60]

四元数都是 wxyz。
"""

from __future__ import annotations

import torch

from rovmimic.fk import quat_apply, quat_mul

NUM_GOAL_FRAMES = 10
GOAL_FRAME_STRIDE = 5


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], -1)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conj(q), v)


def quat_angle(q: torch.Tensor) -> torch.Tensor:
  """单位四元数对应的转角 ∈ [0, π]，零附近梯度有限。"""
  s = torch.sqrt((q[..., 1:] ** 2).sum(-1) + 1e-12)
  return 2.0 * torch.atan2(s, q[..., 0].abs())


def quat_error_angle(q: torch.Tensor, q_ref: torch.Tensor) -> torch.Tensor:
  return quat_angle(quat_mul(quat_conj(q_ref), q))


def matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  two_s = 2.0 / (q * q).sum(-1)
  m = torch.stack([
    1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w),
    two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w),
    two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y),
  ], -1)
  return m.reshape(q.shape[:-1] + (3, 3))


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
  """旋转矩阵的前两列，展平顺序与 SuperMimic 的 quat_to_rot6d 相同。"""
  return matrix_from_quat(q)[..., :2].flatten(-2, -1)


def quat_integrate(q: torch.Tensor, ang_vel_w: torch.Tensor, dt: float) -> torch.Tensor:
  """按世界系角速度把姿态转动 dt，与 SuperMimic 的 quat_integrate 相同。"""
  omega = ang_vel_w * dt
  norm = torch.sqrt((omega * omega).sum(-1, keepdim=True) + 1e-8)
  dq = torch.cat([torch.cos(norm / 2), torch.sin(norm / 2) * omega / norm], -1)
  q_next = quat_mul(dq, q)
  return q_next / q_next.norm(dim=-1, keepdim=True)


def _e_z(like: torch.Tensor) -> torch.Tensor:
  e = torch.zeros_like(like)
  e[..., 2] = 1.0
  return e


def split_full(full: torch.Tensor, nq: int = 7):
  """full_state -> (root_pos, root_quat, qj, lin_w, ang_w, vj)。"""
  return (full[..., 0:3], full[..., 3:7], full[..., 7:7 + nq], full[..., 7 + nq:10 + nq],
          full[..., 10 + nq:13 + nq], full[..., 13 + nq:13 + 2 * nq])


def split_rep(rep: torch.Tensor, nq: int = 7):
  """rep_state -> (depth, g_b, v_b, w_b, qj, vj)。"""
  return rep[..., 0:1], rep[..., 1:4], rep[..., 4:7], rep[..., 7:10], rep[..., 10:10 + nq], rep[..., 10 + nq:10 + 2 * nq]


def represent(full: torch.Tensor, nq: int = 7) -> torch.Tensor:
  pos, quat, qj, lin_w, ang_w, vj = split_full(full, nq)
  g_b = quat_apply_inverse(quat, _e_z(lin_w))
  v_b = quat_apply_inverse(quat, lin_w)
  w_b = quat_apply_inverse(quat, ang_w)
  return torch.cat([pos[..., 2:3], g_b, v_b, w_b, qj, vj], -1)


def full_from(aux: torch.Tensor, rep: torch.Tensor, nq: int = 7) -> torch.Tensor:
  """aux [root_pos, root_quat] + rep_state -> full_state。"""
  _, _, v_b, w_b, qj, vj = split_rep(rep, nq)
  quat = aux[..., 3:7]
  return torch.cat([aux[..., :3], quat, qj, quat_apply(quat, v_b), quat_apply(quat, w_b), vj], -1)


def sensor(rep: torch.Tensor, nq: int = 7) -> torch.Tensor:
  _, g_b, v_b, w_b, qj, vj = split_rep(rep, nq)
  return torch.cat([g_b, w_b, v_b, qj, vj], -1)


SENSOR_TERMS = ("projected_gravity", "ang_vel", "dvl", "joint_pos", "joint_vel")


def sensor_widths(nq: int = 7) -> list[int]:
  return [3, 3, 3, nq, nq]


def integrate_vehicle(aux: torch.Tensor, v_b: torch.Tensor, w_b: torch.Tensor, dv_b: torch.Tensor,
                      dw_b: torch.Tensor, dt: float, c: float):
  """航行器的半隐式积分（与 SuperMimic gencoord 相同）：速度变化量在当前机体系里给出，
  位姿按 v + c·Δv 积分（c 是多个物理子步在恒定加速度下合成的系数）。
  返回 (next_aux, next_g_b, next_v_b, next_w_b)。"""
  pos, quat = aux[..., :3], aux[..., 3:7]
  v_w = quat_apply(quat, v_b + dv_b)
  w_w = quat_apply(quat, w_b + dw_b)
  next_pos = pos + quat_apply(quat, v_b + c * dv_b) * dt
  next_quat = quat_integrate(quat, quat_apply(quat, w_b + c * dw_b), dt)
  next_aux = torch.cat([next_pos, next_quat], -1)
  return (next_aux, quat_apply_inverse(next_quat, _e_z(v_w)), quat_apply_inverse(next_quat, v_w),
          quat_apply_inverse(next_quat, w_w))


# ---------------------------------------------------------------- 目标（参考轨迹窗口）

def goal_window_indices(motion_ends: torch.Tensor, motion_times: torch.Tensor) -> torch.Tensor:
  """[t, t+5, …, t+45]，截断在当前片段的最后一帧，(E, 10)。与 SuperMimic 相同。"""
  clip = torch.searchsorted(motion_ends, motion_times, right=True).clamp(max=motion_ends.numel() - 1)
  last = motion_ends[clip] - 1
  offsets = torch.arange(NUM_GOAL_FRAMES, device=motion_times.device) * GOAL_FRAME_STRIDE
  return torch.minimum(motion_times[:, None] + offsets, last[:, None])


def goal(motion, motion_times: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
  """(E, 230)：未来 10 帧的参考关节角/角速度，以及参考航行器位姿相对 (root_pos, root_quat) 的位置和姿态。
  root_pos 传入带测量误差的位置，就得到策略看到的目标。"""
  idx = goal_window_indices(motion.motion_ends, motion_times)
  E, N = idx.shape
  content = torch.cat([motion.joint_pos[idx], motion.joint_vel[idx]], -1).reshape(E, -1)
  q = root_quat[:, None, :].expand(E, N, 4)
  rel_pos = quat_apply_inverse(q, motion.root_pos_w[idx] - root_pos[:, None, :]).reshape(E, -1)
  rel_ori = quat_to_rot6d(quat_mul(quat_conj(q), motion.root_quat_w[idx])).reshape(E, -1)
  return torch.cat([content, rel_pos, rel_ori], -1)


def goal_dim(nq: int = 7) -> int:
  return NUM_GOAL_FRAMES * (2 * nq + 3 + 6)
