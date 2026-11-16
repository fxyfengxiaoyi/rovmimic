"""USIM（LeRobot v2.1，Hugging Face 版）读取、动作映射、传感器仿真等公共函数。

USIM 每帧（10 Hz）：
  observation.state (29) = joint_pos 5 [a, b, c, d, e] | pwm 8 | joint_v 5 | dvl_v 3 | imu_av 3 | imu_la 3 | pressure | dvl_h
  action (13)            = joint_pos 5 | pwm 8   （与同一帧 state[:13] 完全相同，即该帧下发的指令）
没有世界系位姿。传感器（u0env bluerov2*.scn）都挂在 base_link 原点：
  DVL    rpy = (pi, 0, 0)，即 (vx, -vy, -vz)_body，5 Hz
  IMU    与机体系一致（前-右-下），角速度 + 比力（含重力），20 Hz
  压力   表压 / 1e4，表压 = 1031 * 9.81 * 深度（NED z）
推进器顺序与场景一致（FR, FL, BR, BL, DFR, DFL, DBR, DBL），pwm ∈ [-1, 1] 直接作为设定值。
"""

import sys
from pathlib import Path

import numpy as np

NAV_TASKS = {0, 3, 4, 5, 8}  # 导航类任务用 bluerov2.scn（无机械臂），其余用 bluerov2_alpha5_robotiq.scn
RHO_G = 1031.0 * 9.81
PRESSURE_PER_M = RHO_G / 1e4  # USIM pressure 字段每米深度的读数
FPS = 10
JAW_RATIO = 0.83  # Robotiq 两指关节 = axis_a × 0.83（alpha5_controller.py）

BENCH = Path(__file__).resolve().parents[1] / "stonefish_bench" / "scenarios"
SCENARIOS = {"nav": BENCH / "rov_nav.scn", "alpha5": BENCH / "rov_alpha5.scn"}
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovmimic.paths import DATA_DIR  # noqa: E402


def robot_for_task(task_index: int) -> str:
  return "nav" if int(task_index) in NAV_TASKS else "alpha5"


def load_episode(path):
  import pandas as pd
  df = pd.read_parquet(path)
  s = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
  return {
    "path": str(path),
    "episode": int(df["episode_index"].iloc[0]),
    "task": int(df["task_index"].iloc[0]),
    "q": s[:, 0:5],  # [a, b, c, d, e]
    "pwm": s[:, 5:13],
    "qd": s[:, 13:18],
    "dvl": s[:, 18:21],
    "gyro": s[:, 21:24],
    "acc": s[:, 24:27],
    "pressure": s[:, 27],
    "altitude": s[:, 28],
    "target": np.stack(df["target_pos"].to_numpy()).astype(np.float64),
  }


# ---------------------------------------------------------------- 四元数（wxyz）
def quat_to_mat(q):
  q = np.asarray(q)
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  R = np.empty(q.shape[:-1] + (3, 3))
  R[..., 0, 0] = 1 - 2 * (y * y + z * z)
  R[..., 0, 1] = 2 * (x * y - w * z)
  R[..., 0, 2] = 2 * (x * z + w * y)
  R[..., 1, 0] = 2 * (x * y + w * z)
  R[..., 1, 1] = 1 - 2 * (x * x + z * z)
  R[..., 1, 2] = 2 * (y * z - w * x)
  R[..., 2, 0] = 2 * (x * z - w * y)
  R[..., 2, 1] = 2 * (y * z + w * x)
  R[..., 2, 2] = 1 - 2 * (x * x + y * y)
  return R


def euler_to_quat(roll, pitch, yaw):
  """ZYX（偏航-俯仰-横滚）欧拉角 -> wxyz。"""
  cr, sr = np.cos(roll / 2), np.sin(roll / 2)
  cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
  cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
  return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                   cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def quat_to_euler(q):
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.stack([roll, pitch, yaw], -1)


def roll_pitch_from_acc(acc):
  """静止/匀速时比力 f = -R^T g，由此得到横滚、俯仰（有加速度时只是近似）。"""
  ax, ay, az = acc[..., 0], acc[..., 1], acc[..., 2]
  return np.arctan2(-ay, -az), np.arctan2(ax, np.sqrt(ay * ay + az * az))


# ---------------------------------------------------------------- 传感器仿真（由 sfenv body 状态）
def sensors_from_state(X, nq):
  """X: (..., nx) body 状态 -> dict(dvl, gyro, depth, q)。"""
  R = quat_to_mat(X[..., 3:7])
  Rt = np.swapaxes(R, -1, -2)
  vb = np.einsum("...ij,...j->...i", Rt, X[..., 7:10])
  wb = np.einsum("...ij,...j->...i", Rt, X[..., 10:13])
  dvl = vb * np.array([1.0, -1.0, -1.0])
  return {"dvl": dvl, "gyro": wb, "depth": X[..., 2], "q": X[..., 13:13 + nq], "qd": X[..., 13 + nq:13 + 2 * nq],
          "vb": vb}


# ---------------------------------------------------------------- 动作映射
def plateau_targets(q, eps=1e-4):
  """每一帧的舵机目标 = 关节接下来停住的位置（Stonefish 多体舵机按限速匀速走向目标）。"""
  T = len(q)
  moving = np.zeros(T, bool)
  moving[:-1] = np.abs(np.diff(q, axis=0)).max(1) > eps
  tgt = np.empty_like(q)
  nxt = q[-1]
  for t in range(T - 1, -1, -1):
    if not moving[t]:
      nxt = q[t]
    tgt[t] = nxt
  return tgt


def servo_targets_alpha5(q_usim):
  """USIM joint_pos [a, b, c, d, e] -> rov_alpha5 舵机目标 [e, d, c, b, a, jaw130, jaw139]。"""
  a, b, c, d, e = (q_usim[..., i] for i in range(5))
  return np.stack([e, d, c, b, a, JAW_RATIO * a, JAW_RATIO * a], -1)


def ctrl_actions(ep, robot, ctrl_hz=50, lead=0.0):
  """10 Hz 指令零阶保持到 ctrl_hz。lead > 0 表示每条指令提前 lead 秒生效（对齐采样时刻的不确定）。"""
  T = len(ep["pwm"])
  sub = int(round(ctrl_hz / FPS))
  n = T * sub
  t = np.arange(n) / ctrl_hz
  idx = np.clip(np.floor((t + lead) * FPS + 1e-9).astype(int), 0, T - 1)
  A = [ep["pwm"][idx]]
  if robot == "alpha5":
    A.append(servo_targets_alpha5(plateau_targets(ep["q"]))[idx])
  return np.concatenate(A, 1), sub


def initial_state(ep, k, robot, nq, joint_names=None, yaw=0.0):
  """由第 k 帧的传感器构造 sfenv body 状态：深度来自压力，横滚/俯仰来自比力，偏航给定。"""
  roll, pitch = roll_pitch_from_acc(ep["acc"][k])
  quat = euler_to_quat(roll, pitch, yaw)
  R = quat_to_mat(quat)
  vb = ep["dvl"][k] * np.array([1.0, -1.0, -1.0])
  x = np.zeros(13 + 2 * nq)
  x[2] = ep["pressure"][k] / PRESSURE_PER_M
  x[3:7] = quat
  x[7:10] = R @ vb
  x[10:13] = R @ ep["gyro"][k]
  if robot == "alpha5":
    q = usim_to_sim_joints(ep["q"][k], joint_names)
    qd = usim_to_sim_joints(ep["qd"][k], joint_names)
    x[13:13 + nq] = q
    x[13 + nq:] = qd
  return x


def usim_to_sim_joints(qu, joint_names):
  """USIM [a, b, c, d, e] -> sfenv q 顺序（按 joint_names，两指 = 0.83 a）。"""
  a, b, c, d, e = qu
  table = {"axis_a": a, "axis_b": b, "axis_c": c, "axis_d": d, "axis_e": e,
           "rs1_130_joint": JAW_RATIO * a, "rs1_139_joint": JAW_RATIO * a}
  return np.array([table[n.split("/")[-1]] for n in joint_names])


def sim_to_usim_joints(q, joint_names):
  names = [n.split("/")[-1] for n in joint_names]
  return np.stack([q[..., names.index(n)] for n in ("axis_a", "axis_b", "axis_c", "axis_d", "axis_e")], -1)
