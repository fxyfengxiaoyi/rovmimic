"""逐个文件下载 USIM 低维部分（meta + parquet，不含视频）。已存在的跳过，限流时退避，最后按条数核对。

文件名由 episode 编号确定（chunk = idx // 1000），不需要调用列目录的 API。
"""

import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Vincent2025hello/usim"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rovmimic.paths import USIM_LOWDIM as OUT  # noqa: E402
SPLITS = {"train": 1750, "test": 525}
META = ["info.json", "modality.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl"]


def fetch(rel):
  if (OUT / rel).exists():
    return False
  for attempt in range(60):
    try:
      hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(OUT))
      return True
    except Exception as e:  # noqa: BLE001
      msg = str(e)
      wait = 300 if "429" in msg else min(120, 10 * (attempt + 1))
      print(f"{rel}: {type(e).__name__} {msg[:100]!r} -> sleep {wait}s", flush=True)
      time.sleep(wait)
  raise RuntimeError(f"giving up on {rel}")


files = [f"{s}/meta/{m}" for s in SPLITS for m in META]
files += [f"{s}/data/chunk-{i // 1000:03d}/episode_{i:06d}.parquet" for s, n in SPLITS.items() for i in range(n)]
new = 0
for k, rel in enumerate(files):
  if fetch(rel):
    new += 1
    time.sleep(0.3)
  if (k + 1) % 200 == 0:
    print(f"{k + 1}/{len(files)} checked, {new} downloaded", flush=True)
missing = [f for f in files if not (OUT / f).exists()]
print(f"done: {len(files) - len(missing)}/{len(files)} present, missing {len(missing)}", flush=True)
sys.exit(1 if missing else 0)
