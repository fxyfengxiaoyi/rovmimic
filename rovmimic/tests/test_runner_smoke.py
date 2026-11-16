"""训练循环端到端冒烟测试（CPU，小网络，几十步）：采集 -> 回放缓存 -> 世界模型更新 -> 策略更新 -> 重置采样 -> 存档。

TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python -I rovmimic/tests/test_runner_smoke.py
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=disabled python -I rovmimic/tests/test_runner_smoke.py cuda   # 含 torch.compile 和 CUDA graph
"""

import math
import os
import sys
import tempfile
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))

import torch  # noqa: E402
from supermimic.learners.policy_learner import ActorCriticPolicyLearnerCfg  # noqa: E402

from rovmimic.env import StonefishTrackingEnv, StonefishTrackingEnvCfg  # noqa: E402
from rovmimic.runner import ROVRunner, ROVRunnerCfg, ROVWorldModelLearnerCfg  # noqa: E402


def main():
  os.environ.setdefault("WANDB_MODE", "disabled")
  device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
  env_cfg = StonefishTrackingEnvCfg(refs_dir=str(WS / "data/refs/usim_test_v1"), num_clips=40, num_envs=32,
                                    num_workers=4)
  cfg = ROVRunnerCfg(buffer_size=64, dynamics_hidden_dims=(64, 64), actor_hidden_dims=(64, 64),
                     critic_hidden_dims=(32,), latent_dim=8, replay_ratio=64, max_iterations=40, log_interval=10,
                     save_interval=1000, wm=ROVWorldModelLearnerCfg(batch_size=64, num_batches=2, horizon=8),
                     policy=ActorCriticPolicyLearnerCfg(batch_size=32, horizon=8))
  env = StonefishTrackingEnv(env_cfg, device)
  try:
    runner = ROVRunner(env, cfg, "ROVMimic-smoke", None, device)
    # 手动跑几轮，检查各部分的损失
    for i in range(12):
      runner.step()
    batches = (runner.replay_buffer.sample(cfg.wm.horizon, cfg.wm.batch_size) for _ in range(cfg.wm.num_batches))
    wm_logs = runner.world_model_learner.update(batches)
    pol_logs = runner.policy_learner.update()
    logs = {k: float(v) for k, v in {**wm_logs, **pol_logs}.items()}
    print("\n[smoke] " + "  ".join(f"{k}={v:.3g}" for k, v in logs.items()))
    assert all(math.isfinite(v) for v in logs.values())
    # 完整训练循环（含日志、按回报重置采样、存档）
    with tempfile.TemporaryDirectory() as d:
      os.chdir(d)
      runner.run()
      ckpts = sorted(str(p.relative_to(d)) for p in Path(d).rglob("*.pt"))
      # 推理模块按 config.json 重建（这里是小网络，结构不对会加载失败）
      from rovmimic.policy import ROVPolicy
      from rovmimic import state as ST
      snap = next(Path(d).rglob("checkpoints/it_0000000"))
      a = ROVPolicy.load(snap, device)(torch.randn(4, 36), torch.randn(4, ST.goal_dim()))
      assert a.shape == (4, 13) and torch.isfinite(a).all()
      print(f"[smoke] 推理模块从 {snap.relative_to(d)} 加载成功")
    if device != "cpu":
      print(f"[smoke] GPU 显存峰值 {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"[smoke] run() 完成，存档 {ckpts}；回放缓存 {runner.replay_buffer.num_transitions} 条转移")
    assert any(c.endswith("world_model.pt") for c in ckpts)
  finally:
    env.close()


if __name__ == "__main__":
  main()
