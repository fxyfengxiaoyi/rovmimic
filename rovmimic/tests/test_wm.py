"""世界模型一侧的数学校验（conda 环境 rovmimic，不需要 GPU 和仿真）：

  [1] 把网络输出换成“正确答案”（由参考轨迹相邻两帧反推的 Δv_b、Δw_b），ROVRepresentation.apply_prediction
      应复现下一帧：航行器位姿靠积分，关节靠舵机公式（动作由参考轨迹存的舵机目标换算）
  [2] 参考状态对参考帧的奖励：跟踪项全为 1，惩罚项只剩动作变化
  [3] 梯度：奖励对动作的梯度经网络（推进器）和舵机公式（臂关节）都能传到
  [4] 世界模型的目标与仿真环境观测里的目标一致；策略目标带位置误差后相对位置随之平移

python -I rovmimic/tests/test_wm.py [refs_dir]
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))

from rovmimic import servo as S  # noqa: E402
from rovmimic import state as ST  # noqa: E402
from rovmimic import tracking as TR  # noqa: E402
from rovmimic.env import SensorNoiseCfg  # noqa: E402
from rovmimic.motion import ROVMotion  # noqa: E402
from rovmimic.representation import NQ, ROVDynamics, ROVRepresentation  # noqa: E402


def make_rep(motion):
  rep = ROVRepresentation(SensorNoiseCfg().proprio_half_widths())
  rep.dt, rep.substep_coeff = S.CTRL_DT, 0.75
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda n: SimpleNamespace(motion=motion)))
  rep.bind(env, None)
  return rep


def policy_action(A: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
  a = A.double().numpy()
  rel = S.relative_action_from_targets(a[:, 8:], q.double().numpy())
  return torch.from_numpy(np.concatenate([a[:, :8], rel], -1))


def main():
  refs = sys.argv[1] if len(sys.argv) > 1 else str(WS / "data/refs/usim_test_v1")
  motion = ROVMotion(refs, num_clips=40)
  rep = make_rep(motion)
  T = motion.time_step_total
  A = torch.from_numpy(np.load(Path(refs) / "actions.npy", mmap_mode="r")[:T].copy())
  ends = set(motion.motion_ends.tolist())
  k = torch.tensor([i for i in range(T - 1) if i + 1 not in ends])  # 同一片段内的相邻帧
  full = motion.full_state(torch.arange(T)).double()
  f0, f1 = full[k], full[k + 1]
  r0, r1 = rep.represent(f0), rep.represent(f1)
  aux0, aux1 = f0[:, :7], f1[:, :7]
  q0 = f0[:, 3:7]
  dv = ST.quat_apply_inverse(q0, f1[:, 7 + NQ:10 + NQ]) - r0[:, 4:7]
  dw = ST.quat_apply_inverse(q0, f1[:, 10 + NQ:13 + NQ]) - r0[:, 7:10]
  act = policy_action(A[k], f0[:, 7:7 + NQ])
  rep64 = rep.double()
  oracle = torch.cat([dv, dw], -1) / rep64._prediction_scale  # 网络输出 × 尺度 = Δv
  na, nr = rep64.apply_prediction(aux0, r0, oracle, S.CTRL_DT, act)
  pos_err = (na[:, :3] - aux1[:, :3]).norm(dim=-1)
  ang_err = ST.quat_error_angle(na[:, 3:7], aux1[:, 3:7])
  q_err = (nr[:, 10:10 + NQ] - r1[:, 10:10 + NQ]).abs().amax(-1)
  vj_err = (nr[:, 10 + NQ:] - r1[:, 10 + NQ:]).abs().amax(-1)
  vel_err = (nr[:, 4:10] - r1[:, 4:10]).abs().amax(-1)
  print(f"\n[1] {len(k)} 个相邻帧，网络输出用正确答案：位置误差 p50/max {pos_err.median():.1e}/{pos_err.max():.1e} m，"
        f"姿态 p50/max {ang_err.median():.1e}/{ang_err.max():.1e} rad，速度 max {vel_err.max():.1e}；"
        f"关节角 max {q_err.max():.1e}，关节角速度 max {vj_err.max():.1e}（参考轨迹按 float32 存）")
  assert pos_err.max() < 1e-4 and q_err.max() < 1e-5 and vj_err.max() < 1e-3

  # [2] 参考状态的奖励
  frames = k[:2000]
  ref_b = motion.body_state(frames)
  a_ref = act[:2000].float()
  reward, terms, errors = TR.evaluate(motion.full_state(frames), a_ref, a_ref, motion.fk, ref_b,
                                      motion.joint_pos[frames], TR.default_rewards(), S.CTRL_DT)
  kernel = torch.stack([terms[n] for n in TR.KERNEL_TERMS])
  print(f"[2] 参考状态 vs 参考帧：跟踪项最小 {kernel.min():.6f}，奖励 / 满分 {(reward / TR.max_step_reward(TR.default_rewards(), S.CTRL_DT)).min():.6f}，"
        f"失败概率 max {TR.fail_prob(errors, TR.default_terminations(), 0.05).max():.1e}")
  assert kernel.min() > 0.9999

  # [3] 梯度：随机初始化的世界模型，奖励对动作求导
  dyn = ROVDynamics(rep.float(), S.NUM_ACTIONS, 1, torch.ones(1, S.NUM_ACTIONS), torch.zeros(1, S.NUM_ACTIONS),
                    hidden_dims=(64, 64))
  B = 256
  fr = k[:B]
  s0 = motion.full_state(fr)
  a = (torch.rand(B, S.NUM_ACTIONS) * 1.6 - 0.8).requires_grad_(True)
  next_aux, next_rep = dyn(rep.represent(s0), a, s0[:, :7], torch.zeros(B, 1))
  rwd, _, _ = TR.evaluate(ST.full_from(next_aux, next_rep), a, torch.zeros_like(a), motion.fk,
                          motion.body_state(fr + 1), motion.joint_pos[fr + 1], TR.default_rewards(), S.CTRL_DT)
  g = torch.autograd.grad(rwd.sum(), a)[0].abs().mean(0)
  print(f"[3] 奖励对动作的梯度（平均绝对值）：推进器 {g[:8].mean():.1e}，臂关节 {g[8:12].mean():.1e}，夹爪 {g[12]:.1e}")
  assert (g[:12] > 0).all()

  # [4] 目标
  g_wm, _ = rep.float().compute_wm_goal(motion, fr, s0[:, :7])
  g_env = ST.goal(motion, fr, s0[:, :3], s0[:, 3:7])
  shift = torch.tensor([0.1, -0.05, 0.02])
  meas = torch.cat([s0[:, :3] + shift, s0[:, 3:7]], -1)
  _, g_actor = rep.compute_wm_goal(motion, fr, meas)
  rel = slice(140, 170)
  d_rel = (g_actor[:, rel] - g_wm[:, rel]).reshape(B, 10, 3)
  expect = -ST.quat_apply_inverse(s0[:, None, 3:7].expand(B, 10, 4), shift.expand(B, 10, 3))
  print(f"[4] 目标维数 {g_wm.shape[-1]}，世界模型 vs 仿真环境 max 差 {(g_wm - g_env).abs().max():.1e}；"
        f"位置误差 (0.1, -0.05, 0.02) m 引起的相对位置变化与预期差 {(d_rel - expect).abs().max():.1e}")
  assert (g_wm - g_env).abs().max() == 0 and (d_rel - expect).abs().max() < 1e-5


if __name__ == "__main__":
  main()
