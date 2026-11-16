"""核对舵机关节的逐步行为，以及 SuperMimic 积分公式对舵机关节是否成立。

python -I check_servo_model.py [refs_dir]

  [1] 逐步行为：机械臂停在 armed 姿态，每步把 axis_d 的目标设在当前角度前方 δ 处，看每步实际走多少、步末角速度读数
  [2] 解析模型 Δq = clip(目标 - q, ±vmax·0.02) 是否精确：随机构型、随机目标、随机推进器指令和航行器速度
      （包括机械臂水平伸出、满速换向这种力矩最大的情况）；不精确就说明舵机力矩饱和或有其他效应
  [3] SuperMimic 的积分公式 q' = q + (qd + 0.75·Δqd)·dt 在参考轨迹上的误差：臂关节和航行器位姿
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sfenv" / "python"))
import sfenv  # noqa: E402
import usim_lib as U  # noqa: E402

sim = sfenv.Sim(str(U.SCENARIOS["alpha5"]), U.DATA_DIR, 100.0, 50.0)
J = [n.split("/")[-1] for n in sim.joint_names]      # 状态里的关节顺序
S = [n.split("/")[-1] for n in sim.servo_joints]     # 舵机指令顺序
NT, NQ = sim.nt, sim.nq
DT = sim.ctrl_dt
VMAX = {"axis_e": 0.5, "axis_d": 0.5, "axis_c": 0.5, "axis_b": 0.5, "axis_a": 0.18,
        "rs1_130_joint": 0.15, "rs1_139_joint": 0.15}
ARMED = {"axis_e": 0.0, "axis_d": 0.51, "axis_c": 0.5, "axis_b": 0.5}
ARM = ("axis_e", "axis_d", "axis_c", "axis_b")


def state(q, v=np.zeros(3), w=np.zeros(3)):
  x = np.zeros(13 + 2 * NQ)
  x[2], x[3] = 5.0, 1.0
  x[7:10], x[10:13] = v, w
  for i, n in enumerate(J):
    x[13 + i] = q.get(n, 0.0)
  return x


def q_of(x):
  return {n: x[13 + i] for i, n in enumerate(J)}


def qd_of(x):
  return {n: x[13 + NQ + i] for i, n in enumerate(J)}


def action(thrust, tgt):
  return np.r_[thrust, [tgt[n] for n in S]]


def test_steps():
  print("\n[1] axis_d 每步目标 = 当前角度 + δ（其余关节保持）：")
  for delta in (0.001, 0.003, 0.005, 0.008, 0.01, 0.02, 0.1):
    sim.set_state(state(ARMED))
    moves, reads = [], []
    for k in range(12):
      x = np.array(sim.get_state())
      q = q_of(x)
      tgt = dict(q)
      tgt["axis_d"] = q["axis_d"] + delta
      sim.step(action(np.zeros(NT), tgt))
      y = np.array(sim.get_state())
      if k >= 2:
        moves.append(q_of(y)["axis_d"] - q["axis_d"])
        reads.append(qd_of(y)["axis_d"])
    print(f"    δ={delta:<6g} 每步位移 {np.mean(moves) * 1e3:6.3f} mrad（平均速度 {np.mean(moves) / DT:.3f} rad/s），"
          f"步末角速度读数 {np.mean(reads):.3f} rad/s")


def test_analytic(n_eps=300, n_steps=50, seed=0):
  rng = np.random.default_rng(seed)
  lim = {"axis_e": (0, 6.10), "axis_d": (0, 3.49), "axis_c": (0, 3.22), "axis_b": (0, 3.22),
         "axis_a": (0, 0.015), "rs1_130_joint": (0, 0.0125), "rs1_139_joint": (0, 0.0125)}
  err = {n: [] for n in J}
  for ep in range(n_eps):
    q0 = {n: rng.uniform(*lim[n]) for n in J}
    if ep % 3 == 0:  # 水平伸出：肩、肘使大臂小臂与重力垂直附近
      q0.update({"axis_d": rng.uniform(1.2, 1.9), "axis_c": rng.uniform(0, 0.4)})
    sim.set_state(state(q0, rng.normal(0, 0.3, 3), rng.normal(0, 0.5, 3)))
    sign = {n: 1.0 for n in J}
    for k in range(n_steps):
      x = np.array(sim.get_state())
      q = q_of(x)
      tgt = {}
      for n in J:
        if ep % 3 == 0 and n in ARM:  # 满速来回换向
          if k % 10 == 0:
            sign[n] = -sign[n]
          tgt[n] = q[n] + sign[n] * 1.0
        elif rng.random() < 0.3:
          tgt[n] = rng.uniform(*lim[n])
        else:
          tgt[n] = q[n] + rng.uniform(-2, 2) * VMAX[n] * DT
      thrust = rng.uniform(-1, 1, NT)
      sim.step(action(thrust, tgt))
      y = q_of(np.array(sim.get_state()))
      for n in J:
        pred = q[n] + np.clip(tgt[n] - q[n], -VMAX[n] * DT, VMAX[n] * DT)
        err[n].append(y[n] - pred)
  print(f"\n[2] 解析模型 Δq = clip(目标 - q, ±vmax·{DT:g} s)，{n_eps} 段 × {n_steps} 步"
        f"（1/3 是水平伸出 + 满速换向，推进器随机满量程）：")
  for n in J:
    e = np.abs(np.array(err[n]))
    print(f"    {n:14s} |实际 - 预测| max {e.max():.1e}，> 1e-6 的步 {(e > 1e-6).mean() * 100:.2f}%")


def test_integration(refs_dir):
  d = Path(refs_dir)
  names = json.loads((d / "joint_names.json").read_text())
  q, qd = np.load(d / "joint_pos.npy").astype(np.float64), np.load(d / "joint_vel.npy").astype(np.float64)
  p, v = np.load(d / "root_pos_w.npy").astype(np.float64), np.load(d / "root_lin_vel_w.npy").astype(np.float64)
  quat, w = np.load(d / "root_quat_w.npy").astype(np.float64), np.load(d / "root_ang_vel_w.npy").astype(np.float64)
  ends = np.load(d / "motion_ends.npy")
  starts = np.r_[0, ends[:-1]]
  ok = np.ones(len(q) - 1, bool)
  ok[ends[:-1] - 1] = False          # 跨片段的相邻帧不算
  c = 0.75
  i = np.where(ok)[0]
  print(f"\n[3] SuperMimic 积分公式 x' = x + (速度 + {c}·速度变化)·dt 在 {d.name} 上的误差：")
  for n in ARM:
    j = names.index(n)
    pred = q[i, j] + (qd[i, j] + c * (qd[i + 1, j] - qd[i, j])) * DT
    e = np.abs(q[i + 1, j] - pred)
    moving = np.abs(q[i + 1, j] - q[i, j]) > 1e-6
    print(f"    {n}: 运动帧 {moving.sum()}，误差中位数 {np.median(e[moving]):.1e} rad，"
          f"> 1 mrad 的 {(e[moving] > 1e-3).mean() * 100:.1f}%，max {e.max() * 1e3:.2f} mrad")
  pred = p[i] + (v[i] + c * (v[i + 1] - v[i])) * DT
  e = np.linalg.norm(p[i + 1] - pred, axis=1)
  print(f"    航行器位置: 误差中位数 {np.median(e):.1e} m，p99 {np.percentile(e, 99):.1e} m，max {e.max():.1e} m "
        f"（每步位移中位数 {np.median(np.linalg.norm(p[i + 1] - p[i], axis=1)):.1e} m）")
  wm = w[i] + c * (w[i + 1] - w[i])
  th = np.linalg.norm(wm, axis=1) * DT
  ax = wm / np.maximum(np.linalg.norm(wm, axis=1, keepdims=True), 1e-300)
  dq = np.c_[np.cos(th / 2), np.sin(th / 2)[:, None] * ax]
  pw, px, py, pz = dq.T
  qw, qx, qy, qz = quat[i].T
  qp = np.stack([pw * qw - px * qx - py * qy - pz * qz, pw * qx + px * qw + py * qz - pz * qy,
                 pw * qy - px * qz + py * qw + pz * qx, pw * qz + px * qy - py * qx + pz * qw], 1)
  # 转角用 atan2(|vec(qp* ⊗ q')|, |w|)：arccos(|qp·q'|) 会把 float32 舍入放大到 1e-3 rad 量级
  aw, ax_, ay, az = qp[:, 0], -qp[:, 1], -qp[:, 2], -qp[:, 3]
  bw, bx, by, bz = quat[i + 1].T
  rel = np.stack([aw * bw - ax_ * bx - ay * by - az * bz, aw * bx + ax_ * bw + ay * bz - az * by,
                  aw * by - ax_ * bz + ay * bw + az * bx, aw * bz + ax_ * by - ay * bx + az * bw], 1)
  ang = 2 * np.arctan2(np.linalg.norm(rel[:, 1:], axis=1), np.abs(rel[:, 0]))
  print(f"    航行器姿态: 误差中位数 {np.median(ang):.1e} rad，p99 {np.percentile(ang, 99):.1e} rad，max {ang.max():.1e} rad")


if __name__ == "__main__":
  test_steps()
  test_analytic()
  if len(sys.argv) > 1:
    test_integration(sys.argv[1])
