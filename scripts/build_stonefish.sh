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
echo "[ok] 下一步：bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license，然后跑 README 里的测试"
