#!/usr/bin/env bash
# 编译 Stonefish（C++ 库，无界面使用）和 sfenv（Stonefish 的 Python 模块，给 rovmimic 环境用）。
# 先运行 scripts/fetch_deps.sh 和 scripts/create_env.sh。
#
# 用到两个 conda 环境：
#   stonefish  编译器 + Stonefish 依赖的 C++ 库（SDL2、GL、freetype、libstdc++ …）。sfenv 运行时也从这里加载这些库
#              （写在 RPATH 里），所以编完不能删这个环境。Alpha5 网格转换也在这个环境里跑。
#   rovmimic   训练环境，sfenv 按它的 Python 3.13 编译。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS=${JOBS:-$(nproc)}
TRAIN_ENV=${ROVMIMIC_ENV:-rovmimic}

eval "$(conda shell.bash hook)"
PY=$(conda run -n "$TRAIN_ENV" python -c "import sys; print(sys.executable)")
if ! conda env list | awk '{print $1}' | grep -qx stonefish; then
  conda create -y -n stonefish -c conda-forge python=3.11 c-compiler cxx-compiler cmake make glm sdl2 freetype \
    libegl-devel libgl-devel libglvnd-devel llvm-openmp "numpy=2.4" pybind11
fi
conda activate stonefish
# Alpha5 网格转换（scripts/fetch_alpha5_meshes.sh）在这个环境里跑；版本固定，保证生成的网格与开发机逐字节相同
python -c "import trimesh, fast_simplification" 2>/dev/null || pip install "trimesh==5.2.1" "fast_simplification==0.2.0"
SF_ENV=$CONDA_PREFIX
SF_PREFIX=$ROOT/third_party/stonefish_install

# make >= 4.4 默认用 fifo jobserver，conda 编译器的 LTO 链接（pybind11 默认开 LTO）不认，改用 pipe
MAKE_ARGS=()
if make --version | head -1 | awk '{exit !($3 >= 4.4)}'; then MAKE_ARGS=(--jobserver-style=pipe); fi

echo "=== Stonefish -> $SF_PREFIX"
cmake -S "$ROOT/third_party/stonefish" -B "$ROOT/third_party/stonefish_build" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DCMAKE_INSTALL_PREFIX="$SF_PREFIX" -DCMAKE_PREFIX_PATH="$SF_ENV"
cmake --build "$ROOT/third_party/stonefish_build" --target install -- -j"$JOBS" "${MAKE_ARGS[@]}"

echo "=== sfenv (Python: $PY)"
cmake -S "$ROOT/sfenv" -B "$ROOT/sfenv/build313" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$SF_ENV/bin/c++" -DCMAKE_PREFIX_PATH="$SF_PREFIX;$SF_ENV" \
  -DStonefish_DIR="$SF_PREFIX/lib/cmake/Stonefish" \
  -DPython3_EXECUTABLE="$PY" -DPython3_FIND_STRATEGY=LOCATION
cmake --build "$ROOT/sfenv/build313" -- -j"$JOBS" "${MAKE_ARGS[@]}"
ls -l "$ROOT"/sfenv/python/sfenv.cpython-313-*.so

# 按训练脚本的顺序（先 torch 后 sfenv）在训练环境里导入一次。
# 系统 libstdc++ 较旧时（如 Ubuntu 22.04，只到 GLIBCXX_3.4.30），torch 会先加载系统的那份，sfenv 随后报
# "GLIBCXX_3.4.xx not found"：sfenv 的 RPATH 指向 stonefish 环境，但同名库已加载就不会再找。
# 这时给训练环境设 LD_PRELOAD，改用 stonefish 环境里较新的 libstdc++（向后兼容；仿真子进程会继承）。
check_import() {
  conda run -n "$TRAIN_ENV" python -I -c "import sys; sys.path.insert(0, '$ROOT/sfenv/python'); import torch, sfenv" 2>&1
}
if ! out=$(check_import); then
  if ! grep -q GLIBCXX <<< "$out"; then echo "$out"; echo "[ERROR] $TRAIN_ENV 环境里 import sfenv 失败"; exit 1; fi
  echo "=== 系统 libstdc++ 太旧，给 $TRAIN_ENV 环境设置 LD_PRELOAD=$SF_ENV/lib/libstdc++.so.6"
  conda env config vars set -n "$TRAIN_ENV" LD_PRELOAD="$SF_ENV/lib/libstdc++.so.6" > /dev/null
  if ! out=$(check_import); then echo "$out"; echo "[ERROR] 设置 LD_PRELOAD 后仍然无法 import sfenv"; exit 1; fi
  echo "    已写入环境配置，conda activate $TRAIN_ENV 后生效（已激活的终端要重新激活）"
fi
echo "[ok] $TRAIN_ENV 环境里 import torch, sfenv 正常"
echo "[ok] 下一步：bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license，然后跑 README 里的测试"
