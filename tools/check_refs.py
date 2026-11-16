"""检查 usim_make_refs.py 输出的参考轨迹：格式、自洽性、覆盖范围、机械臂对航行器的扰动。

python -I check_refs.py <refs_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usim_lib as U  # noqa: E402
from usim_make_refs import HULL  # noqa: E402

d = Path(sys.argv[1])
meta = json.loads((d / "meta.json").read_text())
clips = meta["clips"]
names = json.loads((d / "joint_names.json").read_text())
F = {k: np.load(d / f"{k}.npy") for k in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos",
                                          "joint_vel", "actions", "ee_pos_w", "ee_quat_w")}
ends = np.load(d / "motion_ends.npy")
starts = np.r_[0, ends[:-1]]
N = ends[-1]
dt = 1.0 / meta["ctrl_hz"]
print(f"{d.name}: {len(clips)} clips, {N} frames = {N * dt / 3600:.2f} h, joint_names {names}")
print("  shapes:", {k: v.shape for k, v in F.items()})
assert all(len(v) == N for v in F.values())

# 1) 自洽：位置差分 vs 速度（同一片段内）
fd, vel = [], []
for s, e in zip(starts, ends):
  fd.append((F["root_pos_w"][s + 1:e] - F["root_pos_w"][s:e - 1]) / dt)
  vel.append(0.5 * (F["root_lin_vel_w"][s + 1:e] + F["root_lin_vel_w"][s:e - 1]))
fd, vel = np.concatenate(fd), np.concatenate(vel)
print(f"[1] |d(pos)/dt - vel| RMSE {np.sqrt(((fd - vel) ** 2).mean()):.2e} m/s (速度 RMS {np.sqrt((vel ** 2).mean()):.3f})")

# 2) 航行器运动范围
speed = np.linalg.norm(F["root_lin_vel_w"], axis=1)
rate = np.linalg.norm(F["root_ang_vel_w"], axis=1)
rpy = U.quat_to_euler(F["root_quat_w"].astype(np.float64))
print(f"[2] 速度 p50/p95/max {np.percentile(speed, 50):.3f}/{np.percentile(speed, 95):.3f}/{speed.max():.3f} m/s, "
      f"角速度 p95/max {np.percentile(rate, 95):.3f}/{rate.max():.3f} rad/s, "
      f"|roll| p95 {np.degrees(np.percentile(np.abs(rpy[:, 0]), 95)):.1f}°, |pitch| p95 "
      f"{np.degrees(np.percentile(np.abs(rpy[:, 1]), 95)):.1f}°, 最浅深度 {F['root_pos_w'][:, 2].min():.3f} m "
      f"(深度 < 0.15 m 的帧 {(F['root_pos_w'][:, 2] < 0.15).mean() * 100:.1f}%)")

# 3) 机械臂覆盖范围（按模式）
for mode in ("usim", "random"):
  sel = np.concatenate([np.arange(s, e) for s, e, c in zip(starts, ends, clips) if c["arm_mode"] == mode] or [[]]).astype(int)
  if not len(sel):
    continue
  q = F["joint_pos"][sel]
  qd = F["joint_vel"][sel]
  rng = " ".join(f"{n}[{q[:, i].min():.2f},{q[:, i].max():.2f}]" for i, n in enumerate(names))
  moving = (np.abs(qd[:, :4]) > 0.05).any(1).mean()
  print(f"[3] {mode:6s}: {len(sel)} 帧，关节范围 {rng}；臂在运动的帧 {moving * 100:.0f}%，|qd| max {np.abs(qd).max():.2f}")

# 4) TCP 是否进入艇体包围盒（random 模式下做过路径检查，这里复核每一帧）
R = U.quat_to_mat(F["root_quat_w"].astype(np.float64))
tcp_b = np.einsum("nji,nj->ni", R, F["ee_pos_w"] - F["root_pos_w"])
inside = np.all((tcp_b >= HULL[:, 0]) & (tcp_b <= HULL[:, 1]), axis=1)
print(f"[4] TCP 在机体系的范围 x[{tcp_b[:, 0].min():.2f},{tcp_b[:, 0].max():.2f}] y[{tcp_b[:, 1].min():.2f},"
      f"{tcp_b[:, 1].max():.2f}] z[{tcp_b[:, 2].min():.2f},{tcp_b[:, 2].max():.2f}]，进入艇体包围盒的帧 {inside.sum()}")

# 5) 机械臂运动对航行器的扰动：同一 episode 的 usim 与 random 两条片段，推进器指令相同
by_ep = {}
for i, c in enumerate(clips):
  by_ep.setdefault(c["episode"], {})[c["arm_mode"]] = i
dp, dang = [], []
for ep, m in by_ep.items():
  if "usim" not in m or "random" not in m:
    continue
  a, b = m["usim"], m["random"]
  n = min(ends[a] - starts[a], ends[b] - starts[b], int(10 / dt))
  pa, pb = F["root_pos_w"][starts[a]:starts[a] + n], F["root_pos_w"][starts[b]:starts[b] + n]
  qa, qb = F["root_quat_w"][starts[a]:starts[a] + n], F["root_quat_w"][starts[b]:starts[b] + n]
  dp.append(np.linalg.norm(pa[-1] - pb[-1]))
  dang.append(np.degrees(2 * np.arccos(np.clip(np.abs((qa[-1] * qb[-1]).sum()), 0, 1))))
if dp:
  print(f"[5] 同样的推进器指令，random 机械臂 vs usim 机械臂，10 s 后航行器位置差 p50/max "
        f"{np.median(dp):.3f}/{np.max(dp):.3f} m，姿态差 p50/max {np.median(dang):.1f}/{np.max(dang):.1f}°")
