#!/usr/bin/env bash
# Alpha5 臂段网格：来自 Reach Robotics 的 alpha_description（专有文件，不在 u0env 的 Apache-2.0 许可范围内）。
# 使用前必须已取得 Reach Robotics 授权；文件只限组内使用，任何时候都不要提交到 git 或对外分发。
#
#   bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license                    # 从 alpha 仓库下载并转换
#   bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license --from data/alpha5  # 用数据包里现成的网格（不联网）
#
# 下载转换的做法与 u0env README 3.1 相同：取 alpha 仓库里的 alpha_description，放到 u0env 指定位置，
# 再用 u0env 的 tools/dataprocess/pkg_install.py 转成 Stonefish 用的 OBJ（写到 u0env 的 data/alpha5/，已被 u0env 忽略）。
# 转换在 stonefish 环境里跑（scripts/build_stonefish.sh 装了固定版本的 trimesh 5.2.1 + fast_simplification 0.2.0）。
# 转换时会做网格简化，而 Stonefish 按网格几何算臂段的质心和惯量主轴，所以网格不同 = 动力学不同；两种方式最后都按 md5 核对。
set -euo pipefail
if [ "${1:-}" != "--i-have-reach-license" ]; then
  echo "Alpha5 网格受 Reach Robotics 许可约束。确认已取得授权后，加参数 --i-have-reach-license 重新运行。"
  exit 1
fi
FROM=""
if [ "${2:-}" = "--from" ]; then
  FROM=$(realpath "${3:?--from 需要一个目录，例如 data/alpha5}")
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
U0=${ROVMIMIC_U0ENV:-$ROOT/third_party/u0env}
MESH_DIR=$U0/ros_ws/src/stonefish_bluerov2/data/alpha5

if [ -n "$FROM" ]; then
  echo "=== 从 $FROM 复制到 $MESH_DIR"
  mkdir -p "$MESH_DIR/end_effectors"
  cp "$FROM"/*.obj "$MESH_DIR/"
  cp "$FROM"/end_effectors/*.obj "$MESH_DIR/end_effectors/"
else
  SRC=$ROOT/third_party/alpha
  REV=5e1a6648b78584d207e79890bc3905b770892dcd
  if [ ! -e "$SRC/.git" ]; then
    git clone --filter=blob:none --sparse https://github.com/Robotic-Decision-Making-Lab/alpha.git "$SRC"
    git -C "$SRC" sparse-checkout set alpha_description
    git -C "$SRC" checkout --quiet "$REV"
  fi
  DST=$U0/ros_ws/src/description_alpha/origin/alpha_description
  mkdir -p "$DST"
  cp -r "$SRC/alpha_description/." "$DST/"
  conda run --no-capture-output -n stonefish python "$U0/tools/dataprocess/pkg_install.py" --project-root "$U0"
fi

# 与开发机（2026-11-14 生成，训练和参考轨迹都用的这一份）逐字节核对
if ! (cd "$MESH_DIR" && md5sum -c --quiet "$ROOT/scripts/alpha5_meshes.md5"); then
  echo "[ERROR] Alpha5 网格与开发机不同：动力学会和参考轨迹不一致。检查 stonefish 环境的 trimesh / fast_simplification / numpy 版本。"
  exit 1
fi
echo "[ok] Alpha5 网格与开发机逐字节一致"
conda run --no-capture-output -n "${ROVMIMIC_ENV:-rovmimic}" python -I -c "
import sys
sys.path.insert(0, '$ROOT')
from rovmimic.paths import check_assets, STONEFISH_DATA
check_assets()
print(f'[ok] Stonefish 网格齐全：{STONEFISH_DATA}')"
