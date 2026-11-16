"""sfenv 自检：每个场景单独起一个进程（Sim 是进程内单例）。

python -I tests/test_sfenv.py <scenario.scn> <data_dir>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import sfenv  # noqa: E402


def quat_to_mat(q):
  w, x, y, z = q
  return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                   [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                   [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main(scn, data):
  sim = sfenv.Sim(scn, data, 100.0, 50.0)
  print(f"scenario {Path(scn).name}: nt={sim.nt} ns={sim.ns} nq={sim.nq} nx={sim.nx} ctrl_dt={sim.ctrl_dt}")
  print("  thrusters:", [n.split('/')[-1] for n in sim.thruster_names])
  print("  servos -> joints:", list(zip([n.split('/')[-1] for n in sim.servo_names],
                                       [n.split('/')[-1] for n in sim.servo_joints])))
  print("  joint_names (q order):", [n.split('/')[-1] for n in sim.joint_names])
  print("  links:", [n.split('/')[-1] for n in sim.link_names])
  print("  base_cg2o:", np.round(sim.base_cg2o, 4))

  x0 = sim.get_state()
  rng = np.random.default_rng(0)
  T = 200
  A = np.zeros((T, sim.na))
  A[:, :sim.nt] = np.clip(np.cumsum(rng.normal(0, 0.05, (T, sim.nt)), 0), -0.5, 0.5)
  if sim.ns:
    # 舵机目标取各自关节当前位置附近的小幅运动（关节限位都含 0 附近的区间）
    A[:, sim.nt:] = 0.3 * (1 - np.cos(np.linspace(0, 2 * np.pi, T)))[:, None] * 0.5
    A[:, sim.nt + 4:] = 0.0  # 夹爪与两指保持 0

  # 1) set/get 往返
  sim.set_state(x0)
  e = np.abs(sim.get_state() - x0).max()
  print(f"[1] set/get round trip max err {e:.2e}")

  # 2) rollout vs 逐步 step
  X1, _ = sim.rollout(x0, A)
  sim.set_state(x0)
  X2 = []
  for a in A:
    sim.step(a)
    X2.append(sim.get_state())
  print(f"[2] rollout vs step loop max err {np.abs(X1 - np.array(X2)).max():.2e}")

  # 3) step_batch（分时复用）vs 逐条 rollout：4 条不同轨迹交错推进
  starts = [X1[i] for i in (0, 50, 100, 150)]
  acts = [A[(i + 7) % T: (i + 7) % T + 40] for i in (0, 50, 100, 150)]
  ref = [sim.rollout(s, a)[0] for s, a in zip(starts, acts)]
  xs = np.array(starts)
  err = 0.0
  for t in range(40):
    xs = sim.step_batch(xs, np.array([a[t] for a in acts]))
    err = max(err, max(np.abs(xs[i] - ref[i][t]).max() for i in range(4)))
  print(f"[3] step_batch (4 interleaved envs) vs separate rollouts max err {err:.2e}")

  # 4) link 0 位姿 == 状态里的基座位姿
  sim.set_state(X1[-1])
  L = sim.link_poses()
  print(f"[4] link_poses[0] vs state base pose max err {np.abs(L[0] - X1[-1][:7]).max():.2e}")

  # 5) 机体系方向：纵荡指令（FR/FL 负、BR/BL 正）应该让机体系速度沿 +x
  sim.set_state(x0)
  a = np.zeros(sim.na)
  a[:4] = [-0.5, -0.5, 0.5, 0.5]
  for _ in range(100):
    sim.step(a)
  x = sim.get_state()
  vb = quat_to_mat(x[3:7]).T @ x[7:10]
  print(f"[5] surge 2 s -> body-frame velocity {np.round(vb, 4)}  (expect +x)")
  a[:4] = 0
  a[4:8] = 0.5
  sim.set_state(x0)
  for _ in range(100):
    sim.step(a)
  x = sim.get_state()
  vb = quat_to_mat(x[3:7]).T @ x[7:10]
  print(f"    heave sp +0.5 2 s -> body-frame velocity {np.round(vb, 4)}")


if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2])
