#!/usr/bin/env bash
# 下载 USIM 低维数据（HuggingFace，约 220 MB，只有 meta + parquet，不含视频），生成训练 / 测试参考轨迹（约 1.7 GB）。
# 在 rovmimic 环境里运行。下载有限流退避，中断后重跑会跳过已下载的文件。
# 参考轨迹生成是确定性的：开发机上 3469 条训练片段（38.5 h）32 进程约 5 分钟，测试集 1037 条。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORKERS=${WORKERS:-$(( $(nproc) < 32 ? $(nproc) : 32 ))}
export OMP_NUM_THREADS=1

python -I tools/fetch_usim_lowdim.py
for split in train test; do
  out=data/refs/usim_${split}_v1
  if [ -f "$out/motion_ends.npy" ]; then echo "[skip] $out 已存在"; continue; fi
  python -I tools/usim_make_refs.py --data data/usim_lowdim --split "$split" --arm usim,random \
    --workers "$WORKERS" --out "$out"
  python -I tools/check_refs.py "$out"
done
