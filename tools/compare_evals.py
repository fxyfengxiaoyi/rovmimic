"""把多个 rovmimic/eval.py 的结果汇成一张 Markdown 表（汇报用）。

  python -I tools/compare_evals.py data/eval/open_loop data/eval/first/it_*        # 按给出的顺序
  python -I tools/compare_evals.py data/eval/first/* --out data/eval/first/table.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COLS = [  # (表头, metrics.json summary 的键, 格式)
  ("完成率", "completion_rate", "{:.1%}"),
  ("存活比例", "survival_fraction", "{:.1%}"),
  ("归一化回报", "norm_return_mean", "{:.3f}"),
  ("回报 p10", "norm_return_p10", "{:.3f}"),
  ("航行器位置 p50 / p90 (cm)", ("anchor_pos_p50", "anchor_pos_p90"), "{:.1f} / {:.1f}", 100),
  ("姿态 p50 (rad)", "anchor_ori_p50", "{:.3f}"),
  ("TCP p50 / p90 (cm)", ("tcp_pos_p50", "tcp_pos_p90"), "{:.1f} / {:.1f}", 100),
  ("臂关节 p50 (rad)", "arm_joint_p50", "{:.3f}"),
  ("夹爪 p50 (mm)", "gripper_p50", "{:.1f}", 1000),
]


def row(d: Path) -> list[str]:
  m = json.loads((d / "metrics.json").read_text())
  s, meta = m["summary"], m["meta"]
  cells = [d.name, "噪声+扰动" if meta["noisy"] else "无噪声", str(s["clips"])]
  for col in COLS:
    _, key, fmt = col[:3]
    scale = col[3] if len(col) > 3 else 1
    vals = [s[k] * scale for k in (key if isinstance(key, tuple) else (key,))]
    cells.append(fmt.format(*vals))
  fails = ", ".join(f"{k} {v}" for k, v in sorted(s["fail_reasons"].items(), key=lambda kv: -kv[1])) or "—"
  return cells + [fails]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("dirs", nargs="+")
  ap.add_argument("--out")
  args = ap.parse_args()
  head = ["评估", "设置", "片段数"] + [c[0] for c in COLS] + ["失败原因"]
  lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
  for d in args.dirs:
    p = Path(d)
    if (p / "metrics.json").is_file():
      lines.append("| " + " | ".join(row(p)) + " |")
  text = "\n".join(lines)
  print(text)
  if args.out:
    Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
  main()
