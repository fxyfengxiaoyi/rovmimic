#!/usr/bin/env bash
# 建训练用的 conda 环境 rovmimic（Python 3.13，SuperMimic 要求 3.13）。先运行 scripts/fetch_deps.sh。
#   TORCH_INDEX  PyTorch wheel 源，默认 cu130（需要 NVIDIA 驱动 >= 580）。驱动较旧时换成
#                https://download.pytorch.org/whl/cu126 之类，torch 版本保持 2.14.x。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
ENV_NAME=${ROVMIMIC_ENV:-rovmimic}

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" -c conda-forge python=3.13
fi
conda activate "$ENV_NAME"
pip install "torch==2.14.1" --index-url "$TORCH_INDEX"
pip install -e "$ROOT/mjlab_research" -e "$ROOT/supermimic"
pip install -r "$ROOT/scripts/requirements.txt"
python - <<'EOF'
import torch
print(f"[ok] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
EOF
