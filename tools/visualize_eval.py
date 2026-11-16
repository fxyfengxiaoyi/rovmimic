"""把 rovmimic/eval.py 的输出画出来（Stonefish 是无界面编译的，这里用 matplotlib 从状态重建画面）。

  python -I tools/visualize_eval.py data/eval/<名字>                  # 汇总图 + 每条保存的片段的曲线图
  python -I tools/visualize_eval.py data/eval/<名字> --video --clips 0 1 --seconds 30   # 再加 3D 动画（mp4）

输出在 <eval 目录>/vis/：
  summary.png         各片段的归一化回报、误差分布、失败原因
  clip_XX.png         航行器位置 / 姿态、臂关节、夹爪、误差（带失败阈值）、推进器指令，实线 = 实际，虚线 = 参考
  clip_XX.mp4         左：跟随航行器的 3D 近景（艇体框 + 机械臂连杆，蓝 = 实际，橙 = 参考）；右：俯视轨迹
坐标是 Stonefish 的 NED（z 向下），画图时把 z 翻成向上。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rovmimic import servo as S  # noqa: E402
from rovmimic import state as ST  # noqa: E402

SIM_C, REF_C = "#1f6fd1", "#e8890c"
# 画机械臂用的点（rovmimic/motion.py BODY_NAMES 的下标）：m1_link -> m2_1_1 -> m2_joint -> m2_1_2 -> ee_base -> tcp，另画两指
CHAIN = ("m1_link", "m2_1_1_link", "m2_joint_link", "m2_1_2_link", "ee_base_link", "tcp")
FINGERS = (("ee_base_link", "rs1_130_link"), ("ee_base_link", "rs1_139_link"))
HULL_HALF = np.array([0.23, 0.29, 0.13])  # BlueROV2 Heavy 外形的一半（长、宽、高，约值，只用于画框）


def rpy_from_quat(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return np.stack([roll, pitch, yaw], -1)


def rotmat(q: np.ndarray) -> np.ndarray:
  return ST.matrix_from_quat(torch.from_numpy(q).double()).numpy()


def summary_plot(per_clip: list[dict], metrics: dict, path: Path) -> None:
  s = metrics["summary"]
  fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
  ax[0].hist([r["norm_return"] for r in per_clip], bins=30, color=SIM_C)
  ax[0].set(title=f"normalized return (mean {s['norm_return_mean']:.3f})", xlabel="return / max", ylabel="clips")
  ax[1].hist([100 * r["mean_tcp_pos"] for r in per_clip], bins=30, color=SIM_C)
  ax[1].set(title=f"TCP error per clip (p50 {100 * s['tcp_pos_p50']:.1f} cm)", xlabel="mean error (cm)")
  ax[2].hist([100 * r["mean_anchor_pos"] for r in per_clip], bins=30, color=SIM_C)
  ax[2].set(title=f"vehicle position error (p50 {100 * s['anchor_pos_p50']:.1f} cm)", xlabel="mean error (cm)")
  reasons = {"completed": sum(r["completed"] for r in per_clip), **s["fail_reasons"]}
  ax[3].bar(list(reasons), list(reasons.values()), color=[SIM_C] + ["#c23b3b"] * (len(reasons) - 1))
  ax[3].set(title=f"outcome (completion {s['completion_rate']:.1%})", ylabel="clips")
  ax[3].tick_params(axis="x", rotation=20)
  src = metrics["meta"]["policy"] or metrics["meta"]["baseline"]
  fig.suptitle(f"{src}  |  {s['clips']} clips, noisy={metrics['meta']['noisy']}", fontsize=9)
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def clip_plot(R: dict, i: int, thresholds: dict, path: Path) -> None:
  n = int(R["length"][i])
  if n == 0:
    return
  dt = float(R["dt"])
  t = np.arange(n) * dt
  st, ref, act = R["state"][i, :n], R["ref"][i, :n], R["action"][i, :n]
  fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
  for k, (lab, c) in enumerate(zip("xyz", ("tab:red", "tab:green", "tab:blue"), strict=True)):
    ax[0, 0].plot(t, st[:, k], color=c, label=lab)
    ax[0, 0].plot(t, ref[:, k], color=c, ls="--", lw=1)
  ax[0, 0].set(title="vehicle position (NED, m)", ylabel="m")
  ax[0, 0].legend(loc="upper right", fontsize=8)
  rs, rr = np.degrees(rpy_from_quat(st[:, 3:7])), np.degrees(rpy_from_quat(ref[:, 3:7]))
  for k, (lab, c) in enumerate(zip(("roll", "pitch", "yaw"), ("tab:red", "tab:green", "tab:blue"), strict=True)):
    ax[0, 1].plot(t, rs[:, k], color=c, label=lab)
    ax[0, 1].plot(t, rr[:, k], color=c, ls="--", lw=1)
  ax[0, 1].set(title="vehicle attitude (deg)")
  ax[0, 1].legend(loc="upper right", fontsize=8)
  for k, name in enumerate(S.ARM):
    c = f"C{k}"
    ax[1, 0].plot(t, st[:, 7 + k], color=c, label=name)
    ax[1, 0].plot(t, ref[:, 7 + k], color=c, ls="--", lw=1)
  ax[1, 0].set(title="arm joints (rad)")
  ax[1, 0].legend(loc="upper right", fontsize=8, ncol=4)
  ax[1, 1].plot(t, 1e3 * st[:, 11], color=SIM_C, label="actual")
  ax[1, 1].plot(t, 1e3 * ref[:, 11], color=REF_C, ls="--", label="reference")
  ax[1, 1].set(title="gripper opening axis_a (mm)", ylim=(-0.5, 1e3 * S.GRIPPER_RANGE[1] + 0.5))
  ax[1, 1].legend(loc="upper right", fontsize=8)
  e = ax[2, 0]
  e.plot(t, R["err_anchor_pos"][i, :n], label="vehicle pos (m)")
  e.plot(t, R["err_tcp_pos"][i, :n], label="TCP pos (m)")
  e.plot(t, R["err_anchor_ori"][i, :n], label="vehicle ori (rad)")
  for key, c in (("anchor_pos", "C0"), ("anchor_ori", "C2")):
    if key in thresholds:
      e.axhline(thresholds[key], color=c, ls=":", lw=1)
  e.set(title="tracking errors (dotted = failure thresholds)", xlabel="time (s)")
  e.legend(loc="upper right", fontsize=8)
  for k in range(S.NUM_THRUSTERS):
    ax[2, 1].plot(t, np.clip(act[:, k], -1, 1), lw=0.8, label=f"T{k}")
  ax[2, 1].set(title="thruster setpoints (clipped)", xlabel="time (s)", ylim=(-1.05, 1.05))
  ax[2, 1].legend(loc="upper right", fontsize=7, ncol=4)
  fig.suptitle(f"clip {i}: {R['name'][i]}  ({n * dt:.1f} s tracked)", fontsize=10)
  fig.tight_layout()
  fig.savefig(path, dpi=100)
  plt.close(fig)


def body_points(full: np.ndarray) -> np.ndarray:
  """(T, 27) -> (T, R, 3) 各连杆世界系位置（rovmimic/motion.py 的 BODY_NAMES 顺序）。"""
  from rovmimic.motion import make_fk

  fk = make_fk("cpu", torch.float64)
  pos, quat, q, lin, ang, vj = ST.split_full(torch.from_numpy(full).double())
  return fk(pos, quat, q, lin, ang, vj)[0].numpy()


def hull_edges(p: np.ndarray, R: np.ndarray) -> list[np.ndarray]:
  c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * HULL_HALF
  w = p + c @ R.T
  edges = []
  for a in range(8):
    for b in range(a + 1, 8):
      if np.sum(np.abs(c[a] - c[b]) > 1e-9) == 1:
        edges.append(w[[a, b]])
  return edges


def clip_video(R: dict, i: int, seconds: float, fps: int, path: Path) -> None:
  from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

  from rovmimic.motion import BODY_NAMES

  n = int(R["length"][i])
  dt = float(R["dt"])
  if seconds > 0:
    n = min(n, int(seconds / dt))
  if n < 2:
    return
  stride = max(1, int(round(1 / (fps * dt))))
  frames = np.arange(0, n, stride)
  st, ref = R["state"][i, :n], R["ref"][i, :n]
  bs, br = body_points(st), body_points(ref)
  flip = np.array([1.0, 1.0, -1.0])  # NED -> 画图时 z 向上
  idx = [BODY_NAMES.index(b) for b in CHAIN]
  fidx = [(BODY_NAMES.index(a), BODY_NAMES.index(b)) for a, b in FINGERS]
  Rs, Rr = rotmat(st[:, 3:7]), rotmat(ref[:, 3:7])

  fig = plt.figure(figsize=(13, 6))
  ax3 = fig.add_subplot(1, 2, 1, projection="3d")
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.plot(ref[:, 1], ref[:, 0], color=REF_C, ls="--", lw=1, label="reference")
  ax2.plot(st[:, 1], st[:, 0], color=SIM_C, lw=1, label="actual")
  ax2.set(xlabel="east y (m)", ylabel="north x (m)", title="top view")
  ax2.axis("equal")
  ax2.legend(loc="upper right", fontsize=8)
  mk_s, = ax2.plot([], [], "o", color=SIM_C)
  mk_r, = ax2.plot([], [], "o", color=REF_C, mfc="none")
  title = fig.suptitle("")

  def draw(k):
    ax3.cla()
    c = st[k, :3] * flip
    for pts, Rm, col, alpha in ((br[k], Rr[k], REF_C, 0.6), (bs[k], Rs[k], SIM_C, 1.0)):
      for e in hull_edges(pts[0], Rm):
        e = e * flip
        ax3.plot(e[:, 0], e[:, 1], e[:, 2], color=col, lw=1, alpha=alpha)
      ch = pts[idx] * flip
      ax3.plot(ch[:, 0], ch[:, 1], ch[:, 2], "-o", color=col, lw=2.5, ms=3, alpha=alpha)
      for a, b in fidx:
        f = pts[[a, b]] * flip
        ax3.plot(f[:, 0], f[:, 1], f[:, 2], color=col, lw=2, alpha=alpha)
    h = 0.8
    ax3.set(xlim=(c[0] - h, c[0] + h), ylim=(c[1] - h, c[1] + h), zlim=(c[2] - h, c[2] + h),
            xlabel="x north", ylabel="y east", zlabel="up (-z)")
    ax3.set_box_aspect((1, 1, 1))
    ax3.view_init(elev=20, azim=-60)
    mk_s.set_data([st[k, 1]], [st[k, 0]])
    mk_r.set_data([ref[k, 1]], [ref[k, 0]])
    title.set_text(f"clip {i}: {R['name'][i]}   t = {k * dt:5.1f} s   "
                   f"vehicle err {R['err_anchor_pos'][i, k]:.3f} m   TCP err {R['err_tcp_pos'][i, k]:.3f} m")
    return []

  anim = FuncAnimation(fig, draw, frames=frames, blit=False)
  if FFMpegWriter.isAvailable():
    anim.save(str(path), writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=90)
  else:
    path = path.with_suffix(".gif")
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=70)
  plt.close(fig)
  print(f"[vis] {path}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("eval_dir")
  ap.add_argument("--clips", type=int, nargs="*", default=None, help="画哪几条保存的片段（默认全部）")
  ap.add_argument("--video", action="store_true", help="另外渲染 3D 动画")
  ap.add_argument("--seconds", type=float, default=30.0, help="动画最多多少秒（0 = 全程）")
  ap.add_argument("--fps", type=int, default=10)
  args = ap.parse_args()

  d = Path(args.eval_dir)
  vis = d / "vis"
  vis.mkdir(exist_ok=True)
  metrics = json.loads((d / "metrics.json").read_text())
  with open(d / "per_clip.csv") as f:
    per_clip = [{k: (float(v) if k.startswith(("mean_", "max_", "norm_")) else v) for k, v in row.items()}
                for row in csv.DictReader(f)]
  for r in per_clip:
    r["completed"] = r["completed"] == "True"
  summary_plot(per_clip, metrics, vis / "summary.png")
  R = dict(np.load(d / "rollouts.npz"))
  clips = range(len(R["length"])) if args.clips is None else args.clips
  for i in clips:
    clip_plot(R, i, metrics["meta"]["fail_thresholds"], vis / f"clip_{i:02d}.png")
    if args.video:
      clip_video(R, i, args.seconds, args.fps, vis / f"clip_{i:02d}.mp4")
  print(f"[vis] 图在 {vis}")


if __name__ == "__main__":
  main()
