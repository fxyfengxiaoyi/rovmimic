"""闭环评估：在 Stonefish 里让策略从每条参考片段的开头跟到结尾（或到 --max-seconds），统计跟踪效果。

  # 评估训练出的策略（目录里要有 actor.pt；训练日志目录或 checkpoints/it_XXXXXXX 都行）
  python -I rovmimic/eval.py --policy data/runs/logs/ROVMimic-BlueROV2-Alpha5/<时间> --out data/eval/<名字>
  # 基线：开环重放参考轨迹里记录的指令（无扰动、无噪声时与参考完全重合；加了初始扰动后会漂走）
  python -I rovmimic/eval.py --baseline open_loop --out data/eval/open_loop

每条片段一个环境，从片段第一帧开始；失败（位置误差 > 0.5 m、姿态 > 0.8 rad、侵入艇体 > 3 cm）就停止计分。
默认与训练时一致：传感器噪声、位置测量误差、初始扰动都开（--clean 全关，用来看策略的上限）。

输出（--out 目录）：
  metrics.json    汇总：完成率、失败原因、归一化回报、各项误差的中位数 / 90% 分位
  per_clip.csv    每条片段一行
  rollouts.npz    前 --save-clips 条片段的逐步轨迹（真实状态、参考状态、动作、误差），给 tools/visualize_eval.py 画图
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))

import numpy as np  # noqa: E402
import torch  # noqa: E402  先于 sfenv 导入

from rovmimic import servo as S  # noqa: E402
from rovmimic import state as ST  # noqa: E402
from rovmimic import tracking as TR  # noqa: E402
from rovmimic.env import ResetNoiseCfg, StonefishTrackingEnv, StonefishTrackingEnvCfg  # noqa: E402

ERR_KEYS = ("anchor_pos", "anchor_ori", "tcp_pos", "arm_joint", "gripper", "hull")


def parse_args():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  src = ap.add_mutually_exclusive_group(required=True)
  src.add_argument("--policy", help="含 actor.pt 的目录（或 actor.pt 文件）")
  src.add_argument("--baseline", choices=["open_loop"], help="不用策略的基线")
  ap.add_argument("--refs", default=str(WS / "data/refs/usim_test_v1"))
  ap.add_argument("--num-clips", type=int, default=256, help="评估前 n 条片段（0 = 全部）")
  ap.add_argument("--max-seconds", type=float, default=60.0, help="每条片段最多跟多久（0 = 到片段结尾）")
  ap.add_argument("--clean", action="store_true", help="关掉传感器噪声、位置测量误差和初始扰动")
  ap.add_argument("--save-clips", type=int, default=8, help="保存逐步轨迹的片段数（给可视化用）")
  ap.add_argument("--num-workers", type=int, default=None, help="仿真进程数（默认同训练）")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--out", required=True)
  return ap.parse_args()


def main():
  args = parse_args()
  torch.manual_seed(args.seed)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  dev = torch.device(args.device)

  num_clips = args.num_clips or None
  cfg = StonefishTrackingEnvCfg(refs_dir=args.refs, num_clips=num_clips, num_envs=1, sampling_mode="start",
                                episode_length_s=1e6)  # 只按失败和片段结束停止
  ends_all = np.load(Path(args.refs) / "motion_ends.npy")
  n_clips = len(ends_all) if num_clips is None else min(num_clips, len(ends_all))
  cfg.num_envs = n_clips
  if args.num_workers is not None:
    cfg.num_workers = args.num_workers
  if args.clean:
    cfg.reset = ResetNoiseCfg(pos=0, rp=0, yaw=0, lin_vel=0, ang_vel=0, arm_joint=0)
  t0 = time.time()
  env = StonefishTrackingEnv(cfg, str(dev))
  try:
    result = run(env, args, dev)
  finally:
    env.close()
  result["meta"]["wall_s"] = round(time.time() - t0, 1)
  write_outputs(out, result, args)


def run(env: StonefishTrackingEnv, args, dev) -> dict:
  motion, N = env.motion, env.num_envs
  ends = motion.motion_ends
  starts = torch.cat([ends.new_zeros(1), ends[:-1]])
  clip_steps = (ends - starts - 1).to(dev)  # 从第一帧走到最后一帧的步数
  cap = clip_steps if args.max_seconds <= 0 else clip_steps.clamp(max=int(round(args.max_seconds / env.step_dt)))
  T = int(cap.max())

  noisy = not args.clean
  bounds = torch.tensor(env.cfg.sensors.proprio_half_widths(), device=dev)
  if args.policy:
    from rovmimic.policy import ROVPolicy
    pol = ROVPolicy.load(args.policy, dev)
  else:
    ref_actions = torch.from_numpy(np.load(Path(args.refs) / "actions.npy", mmap_mode="r")[:motion.time_step_total]
                                   .astype(np.float32)).to(dev)

  obs, _ = env.reset_to(starts)
  K = min(args.save_clips, N)
  roll = {"state": np.zeros((K, T, 27), np.float32), "ref": np.zeros((K, T, 27), np.float32),
          "action": np.zeros((K, T, S.NUM_ACTIONS), np.float32),
          **{f"err_{k}": np.zeros((K, T), np.float32) for k in ERR_KEYS}}
  alive = torch.ones(N, dtype=torch.bool, device=dev)
  steps = torch.zeros(N, dtype=torch.long, device=dev)
  ret = torch.zeros(N, device=dev)
  err_sum = {k: torch.zeros(N, device=dev) for k in ERR_KEYS}
  err_max = {k: torch.zeros(N, device=dev) for k in ERR_KEYS}
  failed = torch.zeros(N, dtype=torch.bool, device=dev)
  reason = ["" for _ in range(N)]
  tick = time.time()
  for k in range(T):
    full = obs["full_state"]
    t = env.motion_command.time_steps.clone()
    actor_obs = obs["actor"]
    pos = full[:, :3]
    if noisy:
      actor_obs = actor_obs + (torch.rand_like(actor_obs) * 2 - 1) * bounds
      pos = pos + env.position_noise()
    if args.policy:
      action = pol(actor_obs, ST.goal(motion, t, pos, full[:, 3:7]))
    else:  # 参考第 t-1 帧施加的指令（产生第 t 帧），臂关节换成相对量
      A = ref_actions[t - 1].double().cpu().numpy()
      rel = S.relative_action_from_targets(A[:, 8:], full[:, 7:14].double().cpu().numpy())
      action = torch.from_numpy(np.concatenate([A[:, :8], rel], -1)).float().to(dev)
    obs, reward, dones, extras = env.step(action)

    live = alive & (k < cap)
    e, st, rt = extras["errors"], extras["state"], extras["ref_t"]
    qref = motion.joint_pos[rt]
    errs = {"anchor_pos": e["anchor_pos"], "anchor_ori": e["anchor_ori"], "tcp_pos": e["tcp_pos"], "hull": e["hull"],
            "arm_joint": (st[:, 7:11] - qref[:, 0:4]).abs().mean(-1), "gripper": (st[:, 11] - qref[:, 4]).abs()}
    steps += live.long()
    ret += torch.where(live, reward, torch.zeros_like(reward))
    for name, v in errs.items():
      err_sum[name] += torch.where(live, v, torch.zeros_like(v))
      err_max[name] = torch.where(live, torch.maximum(err_max[name], v), err_max[name])
    if k < T and K > 0:
      roll["state"][:, k] = st[:K].cpu().numpy()
      roll["ref"][:, k] = motion.full_state(rt[:K]).cpu().numpy()
      roll["action"][:, k] = action[:K].cpu().numpy()
      for name, v in errs.items():
        roll[f"err_{name}"][:, k] = torch.where(live[:K], v[:K], torch.full_like(v[:K], math.nan)).cpu().numpy()
    fail_now = live & extras["terminated"]
    for i in fail_now.nonzero().flatten().tolist():
      reason[i] = next(n for n, f in extras["fails"].items() if bool(f[i]))
    failed |= fail_now
    alive = live & ~dones.bool() & (k + 1 < cap)
    if k % 500 == 0:
      print(f"[eval] step {k}/{T}  alive {int(alive.sum())}/{N}  failed {int(failed.sum())}  "
            f"{(time.time() - tick) / (k + 1) * 1e3:.0f} ms/step", flush=True)
    if not alive.any():
      break

  denom = steps.clamp(min=1).float()
  per_clip = []
  max_r = TR.max_step_reward(env.cfg.rewards, env.step_dt)
  for i in range(N):
    row = {"clip": i, "name": motion.motion_names[i], "steps_target": int(cap[i]), "steps_survived": int(steps[i]),
           "completed": bool(not failed[i] and steps[i] >= cap[i]), "fail_reason": reason[i],
           "norm_return": float(ret[i] / (max_r * cap[i].clamp(min=1)))}
    row.update({f"mean_{n}": float(err_sum[n][i] / denom[i]) for n in ERR_KEYS})
    row.update({f"max_{n}": float(err_max[n][i]) for n in ERR_KEYS})
    per_clip.append(row)
  lengths = np.minimum(steps[:K].cpu().numpy(), T)
  return {"per_clip": per_clip, "rollouts": {**roll, "length": lengths, "clip": np.arange(K)},
          "meta": {"policy": args.policy, "baseline": args.baseline, "refs": args.refs, "clips": N,
                   "max_seconds": args.max_seconds, "noisy": noisy, "dt": env.step_dt,
                   "fail_thresholds": {k: v.params["threshold"] for k, v in env.cfg.terminations.items()
                                       if "threshold" in v.params}}}


def summarize(per_clip: list[dict]) -> dict:
  def q(key, p):
    return float(np.quantile([r[key] for r in per_clip], p))

  n = len(per_clip)
  reasons = {}
  for r in per_clip:
    if r["fail_reason"]:
      reasons[r["fail_reason"]] = reasons.get(r["fail_reason"], 0) + 1
  s = {"clips": n,
       "completion_rate": sum(r["completed"] for r in per_clip) / n,
       "fail_reasons": reasons,
       "survival_fraction": float(np.mean([r["steps_survived"] / max(r["steps_target"], 1) for r in per_clip])),
       "norm_return_mean": float(np.mean([r["norm_return"] for r in per_clip])),
       "norm_return_p10": q("norm_return", 0.1)}
  for name in ERR_KEYS:
    s[f"{name}_p50"] = q(f"mean_{name}", 0.5)
    s[f"{name}_p90"] = q(f"mean_{name}", 0.9)
  return s


def write_outputs(out: Path, result: dict, args) -> None:
  per_clip = result["per_clip"]
  summary = summarize(per_clip)
  (out / "metrics.json").write_text(json.dumps({"summary": summary, "meta": result["meta"]}, indent=1,
                                               ensure_ascii=False))
  with open(out / "per_clip.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(per_clip[0]))
    w.writeheader()
    w.writerows(per_clip)
  roll = result["rollouts"]
  names = np.array([per_clip[i]["name"] for i in roll["clip"]])
  np.savez_compressed(out / "rollouts.npz", **roll, name=names, dt=result["meta"]["dt"])
  s = summary
  print(f"\n[eval] {s['clips']} clips  完成率 {s['completion_rate']:.1%}  存活比例 {s['survival_fraction']:.1%}  "
        f"归一化回报 {s['norm_return_mean']:.3f}（p10 {s['norm_return_p10']:.3f}）  失败原因 {s['fail_reasons']}")
  print(f"[eval] 每条片段的平均误差，中位数 / 90% 分位：航行器位置 {s['anchor_pos_p50']:.3f}/{s['anchor_pos_p90']:.3f} m，"
        f"姿态 {s['anchor_ori_p50']:.3f}/{s['anchor_ori_p90']:.3f} rad，TCP {s['tcp_pos_p50']:.3f}/{s['tcp_pos_p90']:.3f} m，"
        f"臂关节 {s['arm_joint_p50']:.3f}/{s['arm_joint_p90']:.3f} rad，夹爪 {s['gripper_p50'] * 1e3:.1f}/"
        f"{s['gripper_p90'] * 1e3:.1f} mm")
  print(f"[eval] 写入 {out}/metrics.json、per_clip.csv、rollouts.npz（用时 {result['meta']['wall_s']} s）")


if __name__ == "__main__":
  main()
