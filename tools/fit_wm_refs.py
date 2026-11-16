"""用参考轨迹离线训练世界模型，检查它能不能把航行器动力学学准（conda 环境 rovmimic）。

参考轨迹本身就是 Stonefish 的真实转移：第 k 帧状态 + 动作 A[k] -> 第 k+1 帧。
动作按 rovmimic/servo.py 换算成策略动作（臂关节相对量）。训练用 SuperMimic 原版的 WorldModelLearner
（8 步窗口、连续滚动、Bryson 加权），测试集上比较连续滚 8 / 25 / 50 步的误差，基线是“速度保持不变”（网络输出 0）。

  CUDA_VISIBLE_DEVICES=3 python -I tools/fit_wm_refs.py --iters 300 --out data/wm_refs/run0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))

from rovmimic import servo as S  # noqa: E402
from rovmimic import state as ST  # noqa: E402
from rovmimic.env import SensorNoiseCfg  # noqa: E402
from rovmimic.motion import ROVMotion  # noqa: E402
from rovmimic.representation import ROVDynamics, ROVRepresentation  # noqa: E402
from rovmimic.runner import ROVWorldModelLearnerCfg  # noqa: E402


def load(refs: str, num_clips: int | None):
  m = ROVMotion(refs, "cpu", num_clips)
  T = m.time_step_total
  A = np.load(Path(refs) / "actions.npy", mmap_mode="r")[:T].astype(np.float64)
  q = m.joint_pos.double().numpy()
  act = np.concatenate([A[:, :8], S.relative_action_from_targets(A[:, 8:], q)], -1).astype(np.float32)
  full = m.full_state(torch.arange(T))
  ends = m.motion_ends
  clip_end = ends[torch.searchsorted(ends, torch.arange(T), right=True).clamp(max=len(ends) - 1)]
  return m, full, torch.from_numpy(act), clip_end


def window_starts(clip_end: torch.Tensor, H: int) -> torch.Tensor:
  idx = torch.arange(len(clip_end))
  return idx[clip_end - idx >= H]


def gather(full, act, starts, H, device):
  idx = starts[:, None] + torch.arange(H)
  return full[idx].to(device, non_blocking=True), act[idx].to(device, non_blocking=True)


@torch.no_grad()
def rollout_errors(dyn, rep, full, act, starts, L, device, zero=False):
  """从 starts 出发用真实动作连续滚 L 步，返回每步的位置 / 姿态 / 机体速度误差 (L, ·)。"""
  out = {"pos": [], "ori": [], "vel": [], "angvel": []}
  for chunk in starts.split(2048):
    s, a = gather(full, act, chunk, L + 1, device)
    aux, r = s[:, 0, :7], rep.represent(s[:, 0])
    ctx = torch.zeros(len(chunk), 1, device=device)
    errs = {k: [] for k in out}
    for k in range(L):
      if zero:
        aux, r = rep.apply_prediction(aux, r, torch.zeros(len(chunk), 6, device=device), rep.dt, a[:, k])
      else:
        aux, r = dyn(r, a[:, k], aux, ctx)
      gt = s[:, k + 1]
      gr = rep.represent(gt)
      errs["pos"].append((aux[:, :3] - gt[:, :3]).norm(dim=-1))
      errs["ori"].append(ST.quat_error_angle(aux[:, 3:7], gt[:, 3:7]))
      errs["vel"].append((r[:, 4:7] - gr[:, 4:7]).norm(dim=-1))
      errs["angvel"].append((r[:, 7:10] - gr[:, 7:10]).norm(dim=-1))
    for k in out:
      out[k].append(torch.stack(errs[k]))
  return {k: torch.cat(v, 1).cpu() for k, v in out.items()}


def summarize(e, steps=(8, 25, 50)):
  """各量在第 n 步的中位数和 90% 分位数（悬停帧多，中位数偏乐观，p90 看机动时的表现）。"""
  out = {}
  for k in e:
    for n in steps:
      if n <= e[k].shape[0]:
        out[f"{k}@{n}"] = float(e[k][n - 1].median())
        out[f"{k}@{n}_p90"] = float(e[k][n - 1].quantile(0.9))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train", default=str(WS / "data/refs/usim_train_v1"))
  ap.add_argument("--test", default=str(WS / "data/refs/usim_test_v1"))
  ap.add_argument("--train-clips", type=int, default=None)
  ap.add_argument("--test-clips", type=int, default=200)
  ap.add_argument("--iters", type=int, default=300, help="WorldModelLearner.update 次数（每次 num_batches 个小批）")
  ap.add_argument("--batch-size", type=int, default=1024)
  ap.add_argument("--hidden", type=int, nargs="+", default=[2048, 2048, 2048])
  ap.add_argument("--eval-every", type=int, default=50)
  ap.add_argument("--eval-windows", type=int, default=4096)
  ap.add_argument("--mem-gb", type=float, default=1.8, help="本进程 GPU 显存上限（共享 GPU）")
  ap.add_argument("--out", default=str(WS / "data/wm_refs/run0"))
  args = ap.parse_args()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  if device == "cuda":
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.mem_gb * 2**30 / total))
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  torch.manual_seed(0)

  from supermimic.learners.wm_learner import WorldModelLearner
  from tensordict import TensorDict

  t0 = time.time()
  _, full_tr, act_tr, end_tr = load(args.train, args.train_clips)
  m_te, full_te, act_te, end_te = load(args.test, args.test_clips)
  cfg = ROVWorldModelLearnerCfg(batch_size=args.batch_size)
  H = cfg.horizon
  starts_tr = window_starts(end_tr, H)
  g = torch.Generator().manual_seed(1)
  starts_te = window_starts(end_te, 51)
  starts_te = starts_te[torch.randperm(len(starts_te), generator=g)[:args.eval_windows]]
  print(f"[fit] 训练 {len(full_tr)} 帧（{len(starts_tr)} 个 8 步窗口），测试 {len(full_te)} 帧；读取 {time.time() - t0:.0f}s",
        flush=True)

  rep = ROVRepresentation(SensorNoiseCfg().proprio_half_widths()).to(device)
  rep.dt, rep.substep_coeff = S.CTRL_DT, 0.75
  dyn = ROVDynamics(rep, S.NUM_ACTIONS, 1, torch.ones(1, S.NUM_ACTIONS, device=device),
                    torch.zeros(1, S.NUM_ACTIONS, device=device), hidden_dims=tuple(args.hidden)).to(device)
  learner = WorldModelLearner(dyn, rep, cfg, device=device)
  perm = torch.randperm(len(full_tr), generator=g)[:200_000]
  learner.update_normalization(full_tr[perm].to(device), act_tr[perm].to(device))

  base = summarize(rollout_errors(dyn, rep, full_te, act_te, starts_te, 50, device, zero=True))
  print("[fit] 基线（速度保持不变）: " + "  ".join(f"{k}={v:.4f}" for k, v in base.items()), flush=True)
  history = []
  for it in range(1, args.iters + 1):
    def batches():
      for _ in range(cfg.num_batches):
        st = starts_tr[torch.randint(len(starts_tr), (cfg.batch_size,), generator=g)]
        s, a = gather(full_tr, act_tr, st, H, device)
        yield TensorDict({"state": s, "action": a, "sim_ctx": torch.zeros(*s.shape[:2], 1, device=device)},
                         batch_size=[cfg.batch_size, H])
    logs = learner.update(batches())
    if it % args.eval_every == 0 or it == args.iters:
      dyn.eval()
      ev = summarize(rollout_errors(dyn, rep, full_te, act_te, starts_te, 50, device))
      row = {"iter": it, "time_s": round(time.time() - t0), **{k: float(v) for k, v in logs.items()}, **ev}
      history.append(row)
      print(f"[fit] it {it}  " + "  ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "iter"), flush=True)
      torch.save({"dynamics": dyn.state_dict()}, out / "world_model.pt")
  if device == "cuda":
    print(f"[fit] GPU 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
  (out / "history.json").write_text(json.dumps({"baseline": base, "history": history, "args": vars(args)}, indent=1))


if __name__ == "__main__":
  main()
