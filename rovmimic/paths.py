"""工作区里的路径，全部相对仓库根目录。换机器时用环境变量覆盖，代码里不写绝对路径。

  ROVMIMIC_U0ENV            u0env 仓库（BlueROV2 / Robotiq 网格、Alpha5 网格转换脚本），默认 third_party/u0env
  ROVMIMIC_STONEFISH_DATA   Stonefish 场景引用的网格目录，默认 <u0env>/ros_ws/src/stonefish_bluerov2/data
  ROVMIMIC_USIM             USIM 低维数据（meta + parquet），默认 data/usim_lowdim
"""

from __future__ import annotations

import os
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
U0ENV_DIR = Path(os.environ.get("ROVMIMIC_U0ENV", WS / "third_party" / "u0env"))
STONEFISH_DATA = Path(os.environ.get("ROVMIMIC_STONEFISH_DATA",
                                     U0ENV_DIR / "ros_ws" / "src" / "stonefish_bluerov2" / "data"))
DATA_DIR = str(STONEFISH_DATA) + "/"  # sfenv 直接拼接文件名，末尾要有 /
SCENARIO_DIR = WS / "stonefish_bench" / "scenarios"
ALPHA5_SCN = SCENARIO_DIR / "rov_alpha5.scn"
USIM_LOWDIM = Path(os.environ.get("ROVMIMIC_USIM", WS / "data" / "usim_lowdim"))

# rov_alpha5.scn 引用的网格。Alpha5 臂段来自 Reach Robotics 授权的 alpha_description（只限内部使用，不进仓库），
# 由 scripts/fetch_alpha5_meshes.sh 生成；缺了它们 Stonefish 会解析失败。
REQUIRED_MESHES = (
  "bluerov2/bluerov2_phy.obj",
  "alpha5/M3-INLINE.obj", "alpha5/M2-1-1.obj", "alpha5/M2.obj", "alpha5/M2-1-3.obj", "alpha5/RS1-100-101-123.obj",
  "alpha5/robotiq/hand-e.obj",
)


def check_assets() -> None:
  """场景网格缺失时给出明确的报错（Stonefish 自己只会报“scenario parsing failed”）。"""
  missing = [m for m in REQUIRED_MESHES if not (STONEFISH_DATA / m).is_file()]
  if missing:
    raise FileNotFoundError(
      f"Stonefish meshes missing under {STONEFISH_DATA}: {missing}. "
      "Run scripts/fetch_deps.sh (u0env) and scripts/fetch_alpha5_meshes.sh (Alpha5, Reach license), "
      "or set ROVMIMIC_STONEFISH_DATA.")
