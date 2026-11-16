#!/usr/bin/env bash
# 把训练要用的数据打成一个包。拷到训练机、解压到仓库根目录后，就不用再跑 prepare_data.sh（也不用访问 HuggingFace）。
#
#   bash scripts/pack_data.sh [输出目录，默认 <仓库>/../rovmimic_transfer]
#
# 包含：
#   data/refs/usim_train_v1、data/refs/usim_test_v1   参考轨迹（训练、评估都要）
#   data/usim_lowdim                                 USIM 低维原始数据（重新生成参考轨迹、回放检查时用）
#   data/wm_refs/run1                                世界模型离线拟合结果（默认训练不用，留作对照）
#   data/alpha5                                      Alpha5 网格（Reach Robotics 授权文件，只限组内使用，不要外传）
#   data/MANIFEST.sha256                             逐文件校验值
#
# 训练机上装网格：bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license --from data/alpha5
#
# 训练机上解压（需要 zstd：apt install zstd 或 conda install -c conda-forge zstd）：
#   sha256sum -c rovmimic_data_v1.tar.zst.sha256              # 在压缩包所在目录，核对传输是否完整
#   tar -I zstd -xf <路径>/rovmimic_data_v1.tar.zst -C <仓库根目录>
#   cd <仓库根目录> && sha256sum -c --quiet data/MANIFEST.sha256 && echo OK
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=$(realpath -m "${1:-$ROOT/../rovmimic_transfer}")
NAME=rovmimic_data_v1.tar.zst
mkdir -p "$OUT"
cd "$ROOT"
# Alpha5 网格：从当前使用的位置复制，先按 md5 确认是训练和参考轨迹用的那一份
MESH_DIR=$(python3 -I -c "import sys; sys.path.insert(0, '$ROOT'); from rovmimic.paths import STONEFISH_DATA; print(STONEFISH_DATA / 'alpha5')")
(cd "$MESH_DIR" && md5sum -c --quiet "$ROOT/scripts/alpha5_meshes.md5")
rm -rf data/alpha5 && mkdir -p data/alpha5/end_effectors
cp "$MESH_DIR"/*.obj data/alpha5/ && cp "$MESH_DIR"/end_effectors/*.obj data/alpha5/end_effectors/
ITEMS=(data/refs/usim_train_v1 data/refs/usim_test_v1 data/usim_lowdim data/wm_refs/run1 data/alpha5)
find "${ITEMS[@]}" -type f -not -path "*/.cache/*" -print0 | sort -z | xargs -0 sha256sum > data/MANIFEST.sha256
tar --exclude=.cache -cf - data/MANIFEST.sha256 "${ITEMS[@]}" | zstd -T0 -3 -q -o "$OUT/$NAME" --force
(cd "$OUT" && sha256sum "$NAME" > "$NAME.sha256")
echo "[ok] $OUT/$NAME（$(du -h "$OUT/$NAME" | cut -f1)，$(wc -l < data/MANIFEST.sha256) 个文件）"
