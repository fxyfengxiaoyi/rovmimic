"""多进程 Stonefish：N 个环境分给 P 个工作进程，每个进程一个 sfenv.Sim，用 step_batch 分时复用。

状态和动作放在共享内存里，主进程写入、工作进程原地推进，避免每步序列化大数组。
VecSim 本身不保存环境状态：每步传入 (N, nx) 状态和 (N, na) 动作，返回推进一个控制步后的状态。
状态是 sfenv 的 body 向量 [pos 3, quat wxyz 4, lin_vel_w 3, ang_vel_w 3, q nq, qd nq]，
其中 qd 是 Stonefish 的步末读数（真实物理状态）；“上一步平均速度”的换算在环境层做，见 rovmimic/servo.py。

  sim = VecSim(num_envs=4096, num_workers=32)
  Y = sim.step(X, A)
  sim.close()
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from rovmimic.paths import ALPHA5_SCN, DATA_DIR, WS, check_assets  # noqa: F401  （测试从这里导入）


def _worker(conn, scn, data_dir, phys_hz, ctrl_hz, shm_x, shm_a, shape_x, shape_a, lo, hi):
  os.environ["OMP_NUM_THREADS"] = "1"
  sys.path.insert(0, str(WS / "sfenv" / "python"))
  import sfenv
  sim = sfenv.Sim(scn, data_dir, phys_hz, ctrl_hz)
  bx = shared_memory.SharedMemory(name=shm_x)
  ba = shared_memory.SharedMemory(name=shm_a)
  X = np.ndarray(shape_x, dtype=np.float64, buffer=bx.buf)
  A = np.ndarray(shape_a, dtype=np.float64, buffer=ba.buf)
  conn.send({"nx": sim.nx, "na": sim.na, "nt": sim.nt, "nq": sim.nq, "ctrl_dt": sim.ctrl_dt,
             "joint_names": [n.split("/")[-1] for n in sim.joint_names],
             "servo_names": [n.split("/")[-1] for n in sim.servo_joints],
             "thruster_names": [n.split("/")[-1] for n in sim.thruster_names]})
  try:
    while True:
      cmd = conn.recv()
      if cmd == "step":
        if hi > lo:
          X[lo:hi] = sim.step_batch(X[lo:hi], A[lo:hi])
        conn.send(True)
      elif cmd == "close":
        break
  finally:
    del X, A
    bx.close()
    ba.close()


class VecSim:
  def __init__(self, num_envs: int, num_workers: int = 32, scenario: str | Path = ALPHA5_SCN,
               data_dir: str = DATA_DIR, phys_hz: float = 100.0, ctrl_hz: float = 50.0, nx: int = 27, na: int = 15):
    if data_dir == DATA_DIR:
      check_assets()
    self.num_envs = num_envs
    num_workers = max(1, min(num_workers, num_envs))
    self._shm_x = shared_memory.SharedMemory(create=True, size=num_envs * nx * 8)
    self._shm_a = shared_memory.SharedMemory(create=True, size=num_envs * na * 8)
    self._X = np.ndarray((num_envs, nx), dtype=np.float64, buffer=self._shm_x.buf)
    self._A = np.ndarray((num_envs, na), dtype=np.float64, buffer=self._shm_a.buf)
    bounds = np.linspace(0, num_envs, num_workers + 1).round().astype(int)
    ctx = mp.get_context("spawn")
    self._conns, self._procs = [], []
    for w in range(num_workers):
      parent, child = ctx.Pipe()
      p = ctx.Process(target=_worker, daemon=True,
                      args=(child, str(scenario), data_dir, phys_hz, ctrl_hz, self._shm_x.name, self._shm_a.name,
                            (num_envs, nx), (num_envs, na), int(bounds[w]), int(bounds[w + 1])))
      p.start()
      self._conns.append(parent)
      self._procs.append(p)
    infos = [c.recv() for c in self._conns]
    info = infos[0]
    if info["nx"] != nx or info["na"] != na:
      self.close()
      raise ValueError(f"scenario has nx={info['nx']}, na={info['na']}; expected {nx}, {na}")
    self.nx, self.na, self.nt, self.nq = info["nx"], info["na"], info["nt"], info["nq"]
    self.ctrl_dt = info["ctrl_dt"]
    self.joint_names, self.servo_names = info["joint_names"], info["servo_names"]
    self.thruster_names = info["thruster_names"]

  def step(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """(N, nx) 状态 + (N, na) 动作 -> 推进一个控制步后的 (N, nx) 状态（新数组）。"""
    self._X[:] = states
    self._A[:] = actions
    for c in self._conns:
      c.send("step")
    for c in self._conns:
      c.recv()
    return self._X.copy()

  def close(self):
    for c in getattr(self, "_conns", []):
      try:
        c.send("close")
      except (BrokenPipeError, OSError):
        pass
    for p in getattr(self, "_procs", []):
      p.join(timeout=5)
      if p.is_alive():
        p.terminate()
    self._conns, self._procs = [], []
    for name in ("_X", "_A"):
      if hasattr(self, name):
        delattr(self, name)
    for shm in (getattr(self, "_shm_x", None), getattr(self, "_shm_a", None)):
      if shm is not None:
        shm.close()
        shm.unlink()
    self._shm_x = self._shm_a = None

  def __del__(self):
    if getattr(self, "_shm_x", None) is not None:
      self.close()
