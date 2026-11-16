"""VecSim 校验：多进程推进与单进程 step_batch 逐位一致，以及吞吐。

python -I rovmimic/tests/test_vec_sim.py [num_envs] [num_workers]
"""

import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))

from rovmimic.servo import NUM_ACTIONS, sim_action  # noqa: E402
from rovmimic.vec_sim import ALPHA5_SCN, DATA_DIR, VecSim  # noqa: E402


def initial_states(rng, n):
  x = np.zeros((n, 27))
  x[:, :2] = rng.uniform(-5, 5, (n, 2))
  x[:, 2] = rng.uniform(2, 10, n)
  x[:, 3] = 1.0
  x[:, 13:17] = rng.uniform(0, 1.0, (n, 4))
  return x


def main():
  n = int(sys.argv[1]) if len(sys.argv) > 1 else 256
  w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
  rng = np.random.default_rng(0)
  X0 = initial_states(rng, n)
  pol = [rng.uniform(-1, 1, (n, NUM_ACTIONS)) for _ in range(20)]

  vs = VecSim(n, w)
  X = X0.copy()
  t0 = time.time()
  for a in pol:
    X = vs.step(X, sim_action(a, X[:, 13:20]))
  dt = time.time() - t0
  vs.close()

  import sfenv
  ref = sfenv.Sim(str(ALPHA5_SCN), DATA_DIR, 100.0, 50.0)
  Y = X0.copy()
  for a in pol:
    Y = np.asarray(ref.step_batch(Y, sim_action(a, Y[:, 13:20])))
  err = np.abs(X - Y).max()
  print(f"\n[vec_sim] {n} 个环境 × 20 步，{w} 个工作进程：与单进程 step_batch 最大差 {err:.1e}；"
        f"{n * 20 / dt:.0f} 环境步/s（{dt / 20 * 1e3:.1f} ms/步）")
  assert err == 0.0


if __name__ == "__main__":
  main()
