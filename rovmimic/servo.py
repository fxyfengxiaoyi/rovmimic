"""Alpha5 + Robotiq 舵机关节的解析模型与动作约定。

已用 tools/check_servo_model.py 验证（2026-11-16）：每个控制步（0.02 s，2 个物理步）
  q_next = q + clip(目标 - q, ±vmax·dt)
与 Stonefish 的误差 ≤ 4e-7 rad，包括机械臂水平伸出、满速换向、推进器满量程的情况（舵机力矩从不饱和）。

约定：
  - 关节角速度统一定义为“上一步的平均速度” (q_k - q_{k-1}) / dt。Stonefish 的步末读数在一步内到位时为 0，
    不反映这一步动了多少，所以不用。参考轨迹的 joint_vel、仿真采集的数据、世界模型都用这个定义。
  - 策略动作 13 维：[8 推进器 ∈ [-1, 1], 4 臂关节相对目标 ∈ [-1, 1], 1 夹爪 ∈ [-1, 1]]
      臂关节目标 = q + vmax·dt·a = q + 0.01·a
      夹爪 axis_a 目标 = (a + 1)/2 · 0.015，两根手指目标 = 0.83 × axis_a 目标
  - 关节限位在 Stonefish 里不生效（场景文件的写法解析器不认），由奖励惩罚。
"""

import numpy as np

JOINT_NAMES = ("axis_e", "axis_d", "axis_c", "axis_b", "axis_a", "rs1_139_joint", "rs1_130_joint")  # 状态顺序
SERVO_NAMES = ("axis_e", "axis_d", "axis_c", "axis_b", "axis_a", "rs1_130_joint", "rs1_139_joint")  # 舵机指令顺序
ARM = ("axis_e", "axis_d", "axis_c", "axis_b")
FINGERS = ("rs1_139_joint", "rs1_130_joint")
VMAX = {"axis_e": 0.5, "axis_d": 0.5, "axis_c": 0.5, "axis_b": 0.5, "axis_a": 0.18,
        "rs1_130_joint": 0.15, "rs1_139_joint": 0.15}
LIMITS = {"axis_e": (0.0, 6.10), "axis_d": (0.0, 3.49), "axis_c": (0.0, 3.22), "axis_b": (0.0, 3.22),
          "axis_a": (0.0, 0.015), "rs1_130_joint": (0.0, 0.0125), "rs1_139_joint": (0.0, 0.0125)}
GRIPPER_RANGE = (0.0, 0.015)
JAW_RATIO = 0.83
CTRL_DT = 0.02
NUM_THRUSTERS = 8
NUM_ACTIONS = NUM_THRUSTERS + len(ARM) + 1


def step_average_velocity(joint_pos: np.ndarray, motion_ends: np.ndarray | None = None,
                          dt: float = CTRL_DT) -> np.ndarray:
  """(N, J) 关节角 -> 上一步平均速度 (q_k - q_{k-1}) / dt；每条片段的第一帧取 0。"""
  v = np.zeros_like(joint_pos)
  v[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
  starts = [0] if motion_ends is None else np.r_[0, motion_ends[:-1]]
  v[np.asarray(starts, dtype=np.int64)] = 0.0
  return v


def servo_targets(action: np.ndarray, q: np.ndarray) -> np.ndarray:
  """策略动作 (…, 13) + 当前关节角 (…, 7，状态顺序) -> Stonefish 舵机目标 (…, 7，舵机指令顺序)。"""
  a = np.clip(action[..., NUM_THRUSTERS:], -1.0, 1.0)
  out = np.empty(q.shape[:-1] + (len(SERVO_NAMES),))
  for k, n in enumerate(ARM):
    out[..., SERVO_NAMES.index(n)] = q[..., JOINT_NAMES.index(n)] + VMAX[n] * CTRL_DT * a[..., k]
  grip = (a[..., len(ARM)] + 1.0) / 2.0 * GRIPPER_RANGE[1]
  out[..., SERVO_NAMES.index("axis_a")] = grip
  for n in FINGERS:
    out[..., SERVO_NAMES.index(n)] = JAW_RATIO * grip
  return out


def sim_action(action: np.ndarray, q: np.ndarray) -> np.ndarray:
  """策略动作 (…, 13) -> sfenv 动作 (…, 15) = [8 推进器设定值, 7 舵机目标]。"""
  return np.concatenate([np.clip(action[..., :NUM_THRUSTERS], -1.0, 1.0), servo_targets(action, q)], -1)


def relative_action_from_targets(targets_servo: np.ndarray, q: np.ndarray) -> np.ndarray:
  """参考轨迹里存的绝对舵机目标 (…, 7，舵机顺序) -> 策略动作的关节部分 (…, 5)，用于预训练世界模型。
  臂关节：clip((目标 - q)/0.01, -1, 1)，超出 0.01 rad 的部分本来就不起作用；夹爪：线性映射回 [-1, 1]。"""
  out = np.empty(q.shape[:-1] + (len(ARM) + 1,))
  for k, n in enumerate(ARM):
    out[..., k] = np.clip((targets_servo[..., SERVO_NAMES.index(n)] - q[..., JOINT_NAMES.index(n)])
                          / (VMAX[n] * CTRL_DT), -1.0, 1.0)
  out[..., len(ARM)] = targets_servo[..., SERVO_NAMES.index("axis_a")] / GRIPPER_RANGE[1] * 2.0 - 1.0
  return out
