"""水下版 SuperMimic 训练入口（conda 环境 rovmimic）。

  CUDA_VISIBLE_DEVICES=0 python -I rovmimic/train.py --env.num-envs 1024 --env.num-workers 32
  python -I rovmimic/train.py --help

W&B 默认离线（WANDB_MODE=offline），日志和存档在 data/runs/logs/<task_id>/<时间>/ 下。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))

import torch  # noqa: E402  先于 sfenv 导入
import tyro  # noqa: E402

from rovmimic.env import StonefishTrackingEnv, StonefishTrackingEnvCfg  # noqa: E402
from rovmimic.runner import ROVRunner, ROVRunnerCfg  # noqa: E402


@dataclass
class TrainConfig:
  env: StonefishTrackingEnvCfg = field(default_factory=StonefishTrackingEnvCfg)
  runner: ROVRunnerCfg = field(default_factory=ROVRunnerCfg)
  device: str = "cuda:0"
  task_id: str = "ROVMimic-BlueROV2-Alpha5"
  run_root: str = str(WS / "data" / "runs")
  seed: int = 0


def main(cfg: TrainConfig) -> None:
  os.environ.setdefault("WANDB_MODE", "offline")
  torch.manual_seed(cfg.seed)
  if torch.device(cfg.device).type == "cuda":
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
  cfg.env.refs_dir = str((WS / cfg.env.refs_dir).resolve()) if not Path(cfg.env.refs_dir).is_absolute() \
    else cfg.env.refs_dir
  Path(cfg.run_root).mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("WANDB_DIR", cfg.run_root)
  env = StonefishTrackingEnv(cfg.env, cfg.device)
  try:
    os.chdir(cfg.run_root)  # 原版把 logs/ 建在当前目录
    runner = ROVRunner(env, cfg.runner, cfg.task_id, None, cfg.device)
    runner.run()
  finally:
    env.close()


if __name__ == "__main__":
  main(tyro.cli(TrainConfig))
