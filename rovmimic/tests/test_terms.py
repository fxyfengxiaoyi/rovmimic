"""terms.py 校验：arm_step 与 Stonefish 逐步一致；hull_intrusion 与参考轨迹生成时的包围盒检查一致。

python -I rovmimic/tests/test_terms.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))
import sfenv  # noqa: E402

from rovmimic import servo as S  # noqa: E402
from rovmimic.fk import ALPHA5_SCN, ALPHA5_SITES, ForwardKinematics  # noqa: E402
from rovmimic.terms import HULL_CHECK_BODIES, HULL_SITES, arm_step, hull_intrusion, joint_limit_violation  # noqa: E402
from rovmimic.vec_sim import DATA_DIR  # noqa: E402

SIM = sfenv.Sim(str(ALPHA5_SCN), DATA_DIR, 100.0, 50.0)
assert tuple(n.split("/")[-1] for n in SIM.joint_names) == S.JOINT_NAMES
assert tuple(n.split("/")[-1] for n in SIM.servo_joints) == S.SERVO_NAMES


def test_arm_step(n_eps=100, n_steps=40):
  rng = np.random.default_rng(0)
  err = 0.0
  for _ in range(n_eps):
    x = np.zeros(SIM.nx)
    x[2], x[3] = 5.0, 1.0
    x[13:17] = [rng.uniform(*S.LIMITS[n]) for n in S.ARM]
    x[17] = rng.choice([0.0, 0.015])
    x[18:20] = 0.83 * x[17]
    SIM.set_state(x)
    for _ in range(n_steps):
      x = np.array(SIM.get_state())
      a = rng.uniform(-1.2, 1.2, S.NUM_ACTIONS)
      if rng.random() < 0.5:
        a[S.NUM_THRUSTERS + 4] = rng.choice([-1.0, 1.0])
      SIM.step(S.sim_action(a, x[13:20]))
      y = np.array(SIM.get_state())
      q_next, _ = arm_step(torch.tensor(x[13:20]), torch.tensor(a))
      err = max(err, np.abs(q_next.numpy() - y[13:20]).max())
  print(f"\n[1] arm_step vs Stonefish，{n_eps} 段 × {n_steps} 步随机动作：关节角最大差 {err:.1e}")
  assert err < 1e-6


def test_gradient():
  q = torch.zeros(7, dtype=torch.float64)
  a = torch.zeros(13, dtype=torch.float64, requires_grad=True)
  q_next, _ = arm_step(q, a)
  g = torch.autograd.grad(q_next[:4].sum(), a)[0]
  print(f"[2] 臂关节对相对动作的梯度：{g[8:12].tolist()}（应为 0.01）")
  assert torch.allclose(g[8:12], torch.full((4,), 0.01, dtype=torch.float64))


def test_limits():
  q = torch.tensor([[-0.1, 3.6, 1.0, 1.0, 0.016, 0.0, 0.0]], dtype=torch.float64)
  v = joint_limit_violation(q)
  print(f"[3] 超限量：{v.item():.4f}（应为 0.1 + 0.11 + 0.001 = 0.211）")
  assert abs(v.item() - 0.211) < 1e-9


def test_hull():
  joints = [n.split("/")[-1] for n in SIM.joint_names]
  fk = ForwardKinematics(ALPHA5_SCN, list(HULL_CHECK_BODIES), joints, sites={**ALPHA5_SITES, **HULL_SITES},
                         dtype=torch.float64)
  rng = np.random.default_rng(1)
  n = 4000
  q = np.zeros((n, 7))
  for k, name in enumerate(S.ARM):
    q[:, joints.index(name)] = rng.uniform(*S.LIMITS[name], n)
  zero = torch.zeros(n, 3, dtype=torch.float64)
  ident = torch.zeros(n, 4, dtype=torch.float64)
  ident[:, 0] = 1
  pos, _, _, _ = fk(zero, ident, torch.tensor(q), zero, zero, torch.zeros(n, 7, dtype=torch.float64))
  d = hull_intrusion(pos)
  inside = (d > 0).float().mean().item()
  print(f"[4] 关节在限位内均匀随机：{inside * 100:.0f}% 的构型有检查点进入艇体，最深 {d.max().item() * 100:.1f} cm")
  pts = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.5, 0.0, 0.0], [0.22, 0.0, 0.0]]], dtype=torch.float64)
  dd = hull_intrusion(pts)
  assert abs(dd[0].item() - 0.13) < 1e-12 and abs(dd[1].item() - 0.01) < 1e-12, dd


if __name__ == "__main__":
  test_arm_step()
  test_gradient()
  test_limits()
  test_hull()
