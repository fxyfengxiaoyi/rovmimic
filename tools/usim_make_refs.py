"""由 USIM 生成全身跟踪参考轨迹（50 Hz），输出 SuperMimic / mjlab 的打包格式。

做法：在无界面 Stonefish 里，从每条 episode 第 0 帧的状态出发，用 USIM 记录的推进器指令
（10 Hz 零阶保持到 50 Hz）开环回放，记录完整状态。回放检查（usim_replay_check.py）表明开环回放
与记录的 DVL/IMU/深度基本一致，所以回放得到的状态就是 USIM 轨迹在我们模型上的复现，并且天然
满足动力学可行性。所有 episode（包括导航类）都在 Alpha5 模型上回放，机械臂动作两种：
  usim   ：复现 USIM 记录的关节运动（导航类任务里机械臂保持 0）
  random ：在关节限位内随机生成的点到点运动（舵机按 0.5 rad/s 匀速走向目标），夹爪随机开合；
           目标与途经构型都检查不与艇体包围盒干涉
机械臂运动的反作用会扰动航行器，回放是物理仿真，所以得到的全身状态仍然一致、可行。

数据清洗：
  - 跳过 USIM 里明显异常的 episode（见 episode_problem()：开头姿态翻倒、关节速度超过舵机限速很多、
    关节单帧跳变、角速度异常），原因记录在 meta.json 的 skipped 里；
  - 前往充电站等任务贴着水面走，USIM 里本身就有冒出水面的时刻。第一版跟踪器不考虑水面工况，
    记录里最浅深度不足 MIN_DEPTH 的 episode 整体下移（meta 的 depth_offset），回放后仍浅于
    DROP_DEPTH 的片段丢弃。

输出目录（mjlab MotionLoader 可直接读取的打包格式 + 额外字段）：
  root_pos_w / root_quat_w(wxyz) / root_lin_vel_w / root_ang_vel_w / joint_pos / joint_vel   (N, ·) float32
    joint_vel 是“上一步的平均速度” (q_k - q_{k-1}) / dt，不是 Stonefish 的步末读数（舵机在一步内到位时
    读数为 0），与世界模型、仿真采集用同一定义，见 rovmimic/servo.py。每条片段第一帧取 0。
  motion_ends.npy, joint_names.json, motion_names.json
  actions.npy   (N, 15)  该帧施加的指令 [8 推进器设定值, 7 舵机目标]
  ee_pos_w.npy / ee_quat_w.npy  TCP 位姿（TCP_LINK 沿局部 +z 偏移 TCP_OFFSET，与 U0 抓取规划器一致），供以后校验 FK
  meta.json     每条片段的来源、任务、机械臂模式、随机种子等

python -I usim_make_refs.py --data <usim_lowdim> --split train --arm usim,random --workers 32 --out <dir>
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sfenv" / "python"))
sys.path.insert(0, str(HERE.parent))
import usim_lib as U  # noqa: E402
from rovmimic.servo import step_average_velocity  # noqa: E402

CTRL_HZ = 50
# TCP = U0 抓取时物体中心所在的点。MoveIt 规划组 alpha5 的末端是 ee_base_link（moveit_alpha_robotiq/config/alpha.srdf）；
# U0 抓取规划器 TRY_GRASP 阶段把 ROV 开到使 ee_base_link 位于物体中心上方 0.145+0.15-0.075-0.145 = 0.075 m 处
# （track_target_pose()）。由 USIM 的 target_pos 反推，测试集 86% 的抓取里物体中心沿接近方向距 ee_base_link
# p50 7.4 cm、横向 1.5 cm，落在指垫（z ∈ [0.047, 0.082]）之间。另外 calc_ee_pose_world() 用的 +0.05 m
# 只是成功判定（距离 < 3.2 cm）的参考点，比物体实际位置高 2.5 cm，不取它。
# U0 的偏移加在 ROV 机体 z 上；抓取时夹爪竖直向下（与机体 +z 夹角 p95 0.7°），这里沿 ee_base_link 局部 +z
# （接近方向）偏移，抓取时一致，且始终固连在夹爪上。
TCP_LINK = "ee_base_link"
TCP_OFFSET = 0.075  # m
FINGERTIP = 0.085   # m，指尖沿局部 +z 的位置（指垫 z ∈ [0.047, 0.082]），只用于艇体干涉检查
FIELDS = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")

# 随机机械臂运动（舵机顺序 [e, d, c, b, a, jaw130, jaw139]，前 4 个为臂关节）
ARM_LIMITS = np.array([[0.0, 6.10], [0.0, 3.49], [0.0, 3.22], [0.0, 3.22]])
ARM_VMAX = 0.5
GRIPPER_CLOSED = 0.015
MAX_STEP = 1.5      # 单段每个关节最大变化（rad）
HOLD = (0.5, 3.0)   # 到位后保持时间（s）
# 艇体包围盒（base_link 系，前-右-下，略放大）：臂上任何一点不得进入
HULL = np.array([[-0.26, 0.23], [-0.31, 0.31], [-0.16, 0.13]])
ARM_POINT_LINKS = ("m2_joint_link", "m2_1_2_link", "m1_link", "ee_base_link", "EEBaseLink", "rs1_130_link",
                   "rs1_139_link")

MIN_DEPTH = 0.6    # m，回放起点按 USIM 最浅深度下移到至少这么深
DROP_DEPTH = 0.3   # m，回放后最浅深度小于它的片段丢弃

_sim = None
_names = None


def _init_worker():
  global _sim, _names
  os.environ["OMP_NUM_THREADS"] = "1"
  import sfenv
  _sim = sfenv.Sim(str(U.SCENARIOS["alpha5"]), U.DATA_DIR, 100.0, CTRL_HZ)
  _names = {"links": [n.split("/")[-1] for n in _sim.link_names],
            "joints": [n.split("/")[-1] for n in _sim.joint_names]}


def _tcp(L, i_ee, offset=TCP_OFFSET):
  R = U.quat_to_mat(L[..., i_ee, 3:7])
  return L[..., i_ee, :3] + offset * R[..., :, 2], L[..., i_ee, 3:7]


def _arm_points_body(x_template, q_servo):
  """给定舵机顺序的关节角，返回臂上检查点在 base_link 系的位置。会改动仿真状态。"""
  sim = _sim
  x = x_template.copy()
  x[13 + sim.nq:] = 0.0
  qd = dict(zip(("axis_e", "axis_d", "axis_c", "axis_b", "axis_a"), q_servo[:5]))
  qd["rs1_130_joint"] = qd["rs1_139_joint"] = U.JAW_RATIO * q_servo[4]
  for i, n in enumerate(_names["joints"]):
    x[13 + i] = qd[n]
  sim.set_state(x)
  L = sim.link_poses()
  R0 = U.quat_to_mat(L[0, 3:7])
  idx = [_names["links"].index(n) for n in ARM_POINT_LINKS]
  pts = L[idx, :3]
  i_ee = _names["links"].index(TCP_LINK)
  pts = np.vstack([pts, _tcp(L, i_ee)[0], _tcp(L, i_ee, FINGERTIP)[0]])
  return (R0.T @ (pts - L[0, :3]).T).T


def _inside_hull(p):
  return bool(np.any(np.all((p >= HULL[:, 0]) & (p <= HULL[:, 1]), axis=1)))


def _path_ok(x_template, q0, q1):
  d = q1[:4] - q0[:4]
  travel = np.abs(d).max() / ARM_VMAX
  for t in np.linspace(0.0, travel, 16):
    q = q0.copy()
    q[:4] = q0[:4] + np.clip(d, -ARM_VMAX * t, ARM_VMAX * t)
    if _inside_hull(_arm_points_body(x_template, q)):
      return False
  return True


def random_arm_targets(n_ctrl, rng, x_template, q_start):
  """(n_ctrl, 7) 舵机目标序列。"""
  out = np.empty((n_ctrl, 7))
  q = np.zeros(5)
  q[:4] = q_start[:4]
  q[4] = q_start[4]
  t = 0
  while t < n_ctrl:
    new = q.copy()
    for _ in range(50):
      cand = q.copy()
      for j in range(4):
        if rng.random() < 0.6:
          lo, hi = ARM_LIMITS[j]
          m = 0.05 * (hi - lo)
          cand[j] = np.clip(q[j] + rng.uniform(-MAX_STEP, MAX_STEP), lo + m, hi - m)
      if rng.random() < 0.3:
        cand[4] = GRIPPER_CLOSED - q[4]
      if _path_ok(x_template, q, cand):
        new = cand
        break
    travel = max(np.abs(new[:4] - q[:4]).max() / ARM_VMAX, abs(new[4] - q[4]) / 0.18)
    n = max(1, int(round((travel + rng.uniform(*HOLD)) * CTRL_HZ)))
    out[t:t + n] = np.r_[new, U.JAW_RATIO * new[4], U.JAW_RATIO * new[4]]
    q = new
    t += n
  return out


def episode_problem(ep):
  """USIM episode 的异常检查，返回原因字符串或 None。"""
  acc0 = ep["acc"][0]
  roll, pitch = U.roll_pitch_from_acc(acc0)
  if abs(np.linalg.norm(acc0) - 9.81) > 1.0 or max(abs(roll), abs(pitch)) > np.radians(20):
    return f"initial attitude/acc abnormal (acc0={np.round(acc0, 2).tolist()})"
  if np.abs(ep["qd"]).max() > 1.0:
    return f"recorded joint velocity {np.abs(ep['qd']).max():.2f} rad/s exceeds servo limits"
  if len(ep["q"]) > 1 and np.abs(np.diff(ep["q"], axis=0)).max() > 0.1:
    return f"joint position jump {np.abs(np.diff(ep['q'], axis=0)).max():.3f} rad in one frame"
  if np.linalg.norm(ep["gyro"], axis=1).max() > 3.0:
    return f"angular rate {np.linalg.norm(ep['gyro'], axis=1).max():.2f} rad/s"
  return None


def _servo_order_q(x, joints):
  """sfenv 状态里的关节角 -> 舵机顺序 [e, d, c, b, a]。"""
  return np.array([x[13 + joints.index(n)] for n in ("axis_e", "axis_d", "axis_c", "axis_b", "axis_a")])


def make_clip(args):
  path, arm_mode, seed = args
  sim = _sim
  ep = U.load_episode(path)
  base = {"name": f"ep{ep['episode']:05d}_{arm_mode}", "episode": ep["episode"], "task": ep["task"], "arm_mode": arm_mode}
  problem = episode_problem(ep)
  if problem:
    return {**base, "skip": problem}, None
  A_pwm, _ = U.ctrl_actions({**ep}, "nav", CTRL_HZ, 0.0)  # 只取推进器部分
  n = len(A_pwm)
  x0 = U.initial_state(ep, 0, "alpha5", sim.nq, sim.joint_names)
  depth_offset = max(0.0, MIN_DEPTH - float(ep["pressure"].min()) / U.PRESSURE_PER_M)
  x0[2] += depth_offset
  if arm_mode == "usim":
    tgt = U.servo_targets_alpha5(U.plateau_targets(ep["q"]))
    idx = np.clip(np.floor(np.arange(n) / CTRL_HZ * U.FPS + 1e-9).astype(int), 0, len(ep["q"]) - 1)
    arm = tgt[idx]
  else:
    rng = np.random.default_rng(seed)
    arm = random_arm_targets(n, rng, x0, _servo_order_q(x0, _names["joints"]))
  A = np.concatenate([A_pwm, arm], 1)
  sim.set_state(x0)
  L0 = sim.link_poses()
  X, L = sim.rollout(x0, A, return_links=True)
  S = np.vstack([x0[None], X[:-1]])           # 第 k 帧 = 施加 A[k] 之前的状态
  L = np.concatenate([L0[None], L[:-1]], 0)
  # 四元数取连续的符号（q 与 -q 等价）
  quat = S[:, 3:7].copy()
  flip = np.cumprod(np.where(np.sum(quat[1:] * quat[:-1], 1) < 0, -1.0, 1.0))
  quat[1:] *= flip[:, None]
  ee_pos, ee_quat = _tcp(L, _names["links"].index(TCP_LINK))
  nq = sim.nq
  clip = {
    "root_pos_w": S[:, 0:3], "root_quat_w": quat, "root_lin_vel_w": S[:, 7:10], "root_ang_vel_w": S[:, 10:13],
    "joint_pos": S[:, 13:13 + nq], "joint_vel": step_average_velocity(S[:, 13:13 + nq], dt=1.0 / CTRL_HZ),
    "actions": A,
    "ee_pos_w": ee_pos, "ee_quat_w": ee_quat,
  }
  clip = {k: v.astype(np.float32) for k, v in clip.items()}
  depth = S[:, 2]
  meta = {**base, "source_robot": U.robot_for_task(ep["task"]), "seed": seed, "frames": n,
          "seconds": n / CTRL_HZ, "min_depth": float(depth.min()), "depth_offset": depth_offset,
          "max_speed": float(np.linalg.norm(S[:, 7:10], axis=1).max()),
          "nonfinite": bool(not all(np.isfinite(v).all() for v in clip.values()))}
  if meta["nonfinite"]:
    return {**meta, "skip": "non-finite values in rollout"}, None
  if meta["min_depth"] < DROP_DEPTH:
    return {**meta, "skip": f"replay reached depth {meta['min_depth']:.3f} m < {DROP_DEPTH} m"}, None
  return meta, clip


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--data", required=True, help="usim_lowdim 根目录")
  p.add_argument("--split", default="train")
  p.add_argument("--arm", default="usim,random")
  p.add_argument("--workers", type=int, default=32)
  p.add_argument("--limit", type=int, default=0, help="只处理前 N 条 episode（调试用）")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--out", required=True)
  args = p.parse_args()

  files = sorted(Path(args.data, args.split, "data").glob("chunk-*/episode_*.parquet"))
  if args.limit:
    files = files[:args.limit]
  tasks_text = {}
  for line in Path(args.data, args.split, "meta", "tasks.jsonl").read_text().splitlines():
    t = json.loads(line)
    tasks_text[t["task_index"]] = t["task"]
  jobs = [(str(f), mode, args.seed * 1_000_003 + i) for i, f in enumerate(files) for mode in args.arm.split(",")]
  print(f"[refs] {len(files)} episodes x {args.arm} = {len(jobs)} clips, {args.workers} workers", flush=True)

  t0 = time.time()
  ctx = mp.get_context("spawn")
  metas, clips, skipped = [], [], []
  with ctx.Pool(args.workers, initializer=_init_worker) as pool:
    for i, (meta, clip) in enumerate(pool.imap(make_clip, jobs, chunksize=2)):
      meta["task_text"] = tasks_text.get(meta["task"], "")
      if clip is None:
        skipped.append(meta)
      else:
        metas.append(meta)
        clips.append(clip)
      if (i + 1) % 200 == 0 or i + 1 == len(jobs):
        print(f"[refs] {i + 1}/{len(jobs)} clips, {time.time() - t0:.0f}s", flush=True)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  # joint_names 取 sfenv 的关节顺序（与 FK / 状态向量一致）
  ctx_names = mp.get_context("spawn").Pool(1, initializer=_init_worker)
  joint_names = ctx_names.apply(_get_joint_names)
  ctx_names.close()
  (out / "joint_names.json").write_text(json.dumps(joint_names))
  (out / "motion_names.json").write_text(json.dumps([m["name"] for m in metas]))
  (out / "meta.json").write_text(json.dumps({"tcp_link": TCP_LINK, "tcp_offset": TCP_OFFSET, "ctrl_hz": CTRL_HZ,
                                             "joint_vel": "step_average",
                                             "split": args.split,
                                             "scenario": str(U.SCENARIOS["alpha5"]), "min_depth": MIN_DEPTH,
                                             "drop_depth": DROP_DEPTH, "clips": metas, "skipped": skipped},
                                            indent=1))
  lengths = np.array([c["joint_pos"].shape[0] for c in clips], dtype=np.int64)
  np.save(out / "motion_ends.npy", np.cumsum(lengths))
  for field in (*FIELDS, "actions", "ee_pos_w", "ee_quat_w"):
    merged = np.concatenate([c[field] for c in clips], 0)
    np.save(out / f"{field}.npy", merged)
  shifted = sum(m["depth_offset"] > 0 for m in metas)
  print(f"[refs] wrote {out}: {len(clips)} clips, {int(lengths.sum())} frames "
        f"({lengths.sum() / CTRL_HZ / 3600:.2f} h), shifted down {shifted}, skipped {len(skipped)}, "
        f"{time.time() - t0:.0f}s", flush=True)
  for m in skipped:
    print(f"[refs]   skipped {m['name']}: {m['skip']}", flush=True)


def _get_joint_names():
  return list(_names["joints"])


if __name__ == "__main__":
  main()
