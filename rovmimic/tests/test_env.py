"""StonefishTrackingEnv 校验：不加重置扰动、从片段开头开始、每步执行参考轨迹存的动作，
仿真应逐帧复现参考轨迹，奖励接近满分，不触发失败判定。

python -I rovmimic/tests/test_env.py [refs_dir]
"""

import sys
from pathlib import Path

import numpy as np
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))

from rovmimic import servo as S  # noqa: E402
from rovmimic.env import ResetNoiseCfg, StonefishTrackingEnv, StonefishTrackingEnvCfg  # noqa: E402
from rovmimic.tracking import max_step_reward  # noqa: E402


def main():
  refs = sys.argv[1] if len(sys.argv) > 1 else str(WS / "data/refs/usim_test_v1")
  zero = ResetNoiseCfg(pos=0, rp=0, yaw=0, lin_vel=0, ang_vel=0, arm_joint=0)
  cfg = StonefishTrackingEnvCfg(refs_dir=refs, num_clips=8, num_envs=8, num_workers=4, sampling_mode="start",
                                reset=zero, episode_length_s=1000.0)
  env = StonefishTrackingEnv(cfg)
  A = torch.from_numpy(np.load(Path(refs) / "actions.npy", mmap_mode="r")[:env.motion.time_step_total].copy())
  rmax = max_step_reward(cfg.rewards, env.step_dt)
  pos_err, ori_err, rew, fails, steps = [], [], [], 0, 600
  for _ in range(steps):
    t = env.motion_command.time_steps.clone()
    a_ref = A[t - 1].double().numpy()
    q = env._x[:, 13:20]
    act = np.concatenate([a_ref[:, :8], S.relative_action_from_targets(a_ref[:, 8:], q)], -1)
    _, r, dones, extras = env.step(torch.from_numpy(act).float())
    tm = env.termination_manager
    fails += int(tm.terminated.sum())
    done = dones.bool()
    full_ref = env.motion.full_state(t)
    st = env.full_state()  # 已重置的环境不比较
    pos_err.append((st[~done, :3] - full_ref[~done, :3]).norm(dim=-1))
    ori_err.append(extras["log"]["Metrics/err_anchor_ori"].reshape(1))
    rew.append(r / rmax)
  pos_err = torch.cat(pos_err)
  rew = torch.cat(rew)
  print(f"\n[env] {cfg.num_envs} 个环境 × {steps} 步执行参考动作：航行器位置误差 p50/max {pos_err.median():.1e}/{pos_err.max():.1e} m，"
        f"奖励 / 满分 p50/min {rew.median():.4f}/{rew.min():.4f}，失败判定 {fails} 次")
  obs = env.get_observations()
  print(f"[env] 观测维数 full_state {obs['full_state'].shape[-1]}，actor {obs['actor'].shape[-1]}，goal {obs['goal'].shape[-1]}；"
        f"重置采样箱数 {env.motion_command.bin_count}")
  env.close()
  assert fails == 0 and pos_err.max() < 1e-2


if __name__ == "__main__":
  main()
