"""把训练的控制台输出（train.py 的 stdout，用 tee 存下来的文件）解析成表格和曲线，不依赖 W&B。

  python -I tools/plot_train_log.py data/runs/train_first.log            # -> 同目录 *.csv、*.png
  python -I tools/plot_train_log.py data/runs/train_first.log --smooth 20

SuperMimic 每 log_interval（默认 10）个真实步打印一块 "Learning iteration k/K"，块里是 "名字: 数值"。
表格的 iteration 列换算成真实步（= k × log_interval）。
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEAD = re.compile(r"Learning iteration (\d+)/(\d+)")
ROW = re.compile(r"^\s*(.+?):\s+(-?[0-9.]+(?:e[-+]?\d+)?|nan|inf|-inf)s?\s*$", re.IGNORECASE)
CLOCK = re.compile(r"^\s*(Time elapsed|ETA):\s+(\d+):(\d\d):(\d\d)\s*$")

PANELS = [  # (标题, 选列的前缀或全名)
  ("episode reward / length", ["Mean reward", "Mean episode length"]),
  ("imagined return (policy update)", ["return", "Policy/reward"]),
  ("terminations (fraction of envs per step)", ["Termination/"]),
  ("tracking errors in sim", ["Metrics/err_"]),
  ("world-model loss (normalized)", ["Loss/"]),
  ("reward terms", ["Reward/"]),
  ("policy / critic", ["grad_norm", "value_function", "Policy/grad_norm", "Policy/value_function"]),
  ("time per log block (s)", ["Collection time", "WM Learning time", "Policy Learning time", "Iteration time"]),
]


def parse(path: Path, log_interval: int) -> list[dict]:
  rows, cur = [], None
  for raw in path.read_text(errors="replace").splitlines():
    line = ANSI.sub("", raw)
    m = HEAD.search(line)
    if m:
      if cur:
        rows.append(cur)
      cur = {"iteration": int(m.group(1)) * log_interval}
      continue
    if cur is None:
      continue
    m = CLOCK.match(line)
    if m:
      cur[m.group(1)] = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4))
      continue
    m = ROW.match(line)
    if m:
      key = m.group(1).strip()
      if key.startswith("Episode "):
        key = key[len("Episode "):]
      cur[key] = float(m.group(2))
  if cur:
    rows.append(cur)
  return rows


def smooth(y: np.ndarray, k: int) -> np.ndarray:
  if k <= 1 or len(y) < k:
    return y
  ok = np.isfinite(y)
  out = np.full_like(y, np.nan)
  kernel = np.ones(k) / k
  out[ok] = np.convolve(np.pad(y[ok], (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("log")
  ap.add_argument("--log-interval", type=int, default=10)
  ap.add_argument("--smooth", type=int, default=10)
  args = ap.parse_args()
  path = Path(args.log)
  rows = parse(path, args.log_interval)
  if not rows:
    raise SystemExit(f"{path}: no 'Learning iteration' blocks found")
  keys = ["iteration"] + sorted({k for r in rows for k in r} - {"iteration"})
  csv_path = path.with_suffix(".csv")
  with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)

  x = np.array([r["iteration"] for r in rows])
  fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
  for ax, (title, sel) in zip(axes.flat, PANELS, strict=True):
    cols = [k for k in keys if any(k == s or (s.endswith(("/", "_")) and k.startswith(s)) for s in sel)]
    for k in cols:
      y = np.array([r.get(k, np.nan) for r in rows], float)
      ax.plot(x, smooth(y, args.smooth), lw=1, label=k)
    ax.set_title(title)
    if cols:
      ax.legend(fontsize=7, ncol=2)
  for ax in axes[-1]:
    ax.set_xlabel("real steps (iterations)")
  last = rows[-1]
  fig.suptitle(f"{path.name}: {len(rows)} log blocks, last iteration {last['iteration']}, "
               f"elapsed {last.get('Time elapsed', 0) / 3600:.2f} h", fontsize=10)
  fig.tight_layout()
  png = path.with_suffix(".png")
  fig.savefig(png, dpi=100)
  print(f"[log] {len(rows)} blocks -> {csv_path}, {png}")


if __name__ == "__main__":
  main()
