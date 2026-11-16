#!/usr/bin/env bash
# 按固定版本拉取上游依赖（都是公开仓库）。可重复运行：已存在的仓库只核对版本，不改动。
#   supermimic、mjlab_research  -> 仓库根目录（supermimic 的 pyproject 要求两者同级）
#   stonefish（仿真器源码）、u0env（BlueROV2 / Robotiq 网格、Alpha5 网格转换脚本）-> third_party/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/third_party"

clone() {  # url 目录 commit
  local url=$1 dir=$2 rev=$3
  if [ ! -e "$dir/.git" ]; then
    git clone --filter=blob:none "$url" "$dir"
    git -C "$dir" checkout --quiet "$rev"
  fi
  local head
  head=$(git -C "$dir" rev-parse HEAD)
  if [ "$head" != "$rev" ]; then
    echo "[WARN] $dir 在 ${head:0:7}，不是固定版本 ${rev:0:7}（已有仓库不自动切换，需要时手动 git checkout $rev）"
  else
    echo "[ok] $dir @ ${rev:0:7}"
  fi
}

clone https://github.com/HybridRobotics/supermimic     "$ROOT/supermimic"             abc124c1bd5976341034668af3d12c087f2f1f55
clone https://github.com/HybridRobotics/mjlab_research "$ROOT/mjlab_research"         dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd
clone https://github.com/patrykcieslak/stonefish       "$ROOT/third_party/stonefish"  7d52673791834caa743907dde83a1949134ef2f0
clone https://github.com/VincentGu2000/u0env           "$ROOT/third_party/u0env"      3f4b8409a9942935df1521fac2d905b65a2d36e2
