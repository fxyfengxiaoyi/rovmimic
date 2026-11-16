"""正运动学校验：与 Stonefish 对比位姿、速度，与参考轨迹里存的 TCP 对比，并检查梯度。

python -I rovmimic/tests/test_fk.py [refs_dir]

  [1] 位姿：随机状态写入 sfenv，Stonefish 给出的 12 个连杆原点系位姿 vs 正运动学（float64）
  [2] 速度：沿状态速度方向前后各推进 ±eps，用 Stonefish 的位姿做中心差分，vs 正运动学速度
      （Stonefish 自带的连杆速度接口在分叉的运动学树上会把兄弟连杆的关节速度也累加进去，不能当真值）
  [3] 参考轨迹：由打包的基座 + 关节状态算出 TCP，vs usim_make_refs.py 用 Stonefish 存下的 ee_pos_w / ee_quat_w
  [4] 梯度：torch.autograd.gradcheck
  [5] 速度：CPU 上 float32 批量的耗时
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS))
sys.path.insert(0, str(WS / "sfenv" / "python"))
sys.path.insert(0, str(WS / "tools"))
import sfenv  # noqa: E402
import usim_lib as U  # noqa: E402
from rovmimic.fk import ALPHA5_SCN, ALPHA5_SITES, ForwardKinematics, quat_mul  # noqa: E402

SIM = sfenv.Sim(str(ALPHA5_SCN), U.DATA_DIR, 100.0, 50.0)
LINKS = [n.split("/")[-1] for n in SIM.link_names]
JOINTS = [n.split("/")[-1] for n in SIM.joint_names]
NQ = SIM.nq
FK = ForwardKinematics(ALPHA5_SCN, LINKS + ["tcp"], JOINTS, sites=ALPHA5_SITES, dtype=torch.float64)
PRISMATIC = {"rs1_130_joint", "rs1_139_joint"}


def random_states(rng, n):
  x = np.zeros((n, 13 + 2 * NQ))
  x[:, :3] = rng.uniform(-20, 20, (n, 3))
  q = rng.normal(size=(n, 4))
  x[:, 3:7] = q / np.linalg.norm(q, axis=1, keepdims=True)
  x[:, 7:13] = rng.normal(0, 0.5, (n, 6))
  for i, name in enumerate(JOINTS):
    if name in PRISMATIC:
      x[:, 13 + i] = rng.uniform(0, 0.0125, n)
      x[:, 13 + NQ + i] = rng.normal(0, 0.02, n)
    else:
      x[:, 13 + i] = rng.uniform(-np.pi, np.pi, n)
      x[:, 13 + NQ + i] = rng.normal(0, 0.5, n)
  return x


def fk_of(x):
  t = torch.as_tensor(x, dtype=torch.float64)
  return FK(t[..., 0:3], t[..., 3:7], t[..., 13:13 + NQ], t[..., 7:10], t[..., 10:13], t[..., 13 + NQ:])


def stonefish_links(x):
  SIM.set_state(x)
  return np.array(SIM.link_poses())


def quat_angle(a, b):
  """两个 wxyz 四元数之间的转角。用 atan2 而不是 arccos(|a·b|)：后者在角度接近 0 时把 float32 舍入误差
  （约 1e-7）放大到 1e-3 rad 量级。"""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  d = quat_mul(torch.from_numpy(a * np.array([1, -1, -1, -1])), torch.from_numpy(b)).numpy()
  return 2 * np.arctan2(np.linalg.norm(d[..., 1:], axis=-1), np.abs(d[..., 0]))


def advance(x, eps):
  """沿状态速度推进 eps 秒（纯运动学）：位置 + eps·v，姿态绕世界系角速度转，关节 + eps·qd。"""
  y = x.copy()
  y[:3] += eps * x[7:10]
  w = x[10:13]
  th = np.linalg.norm(w) * eps
  dq = np.r_[np.cos(th / 2), np.sin(th / 2) * w / max(np.linalg.norm(w), 1e-300)]
  y[3:7] = quat_mul(torch.tensor(dq), torch.tensor(x[3:7])).numpy()
  y[13:13 + NQ] += eps * x[13 + NQ:]
  return y


def test_poses(n=300):
  x = random_states(np.random.default_rng(0), n)
  pos, quat, _, _ = fk_of(x)
  ep, er = 0.0, 0.0
  for k in range(n):
    L = stonefish_links(x[k])
    ep = max(ep, np.abs(pos[k, :len(LINKS)].numpy() - L[:, :3]).max())
    er = max(er, quat_angle(quat[k, :len(LINKS)].numpy(), L[:, 3:7]).max())
  print(f"\n[1] 位姿 vs Stonefish，{n} 个随机状态 × {len(LINKS)} 个连杆：位置误差 max {ep:.1e} m，姿态误差 max {er:.1e} rad")
  assert ep < 1e-9 and er < 1e-7


def test_velocities(n=100, eps=1e-6):
  x = random_states(np.random.default_rng(1), n)
  _, _, lin, ang = fk_of(x)
  el, ea, scale = 0.0, 0.0, 0.0
  for k in range(n):
    Lp, Lm = stonefish_links(advance(x[k], eps)), stonefish_links(advance(x[k], -eps))
    v_fd = (Lp[:, :3] - Lm[:, :3]) / (2 * eps)
    qm_conj = Lm[:, 3:7] * np.array([1, -1, -1, -1])
    d = quat_mul(torch.tensor(Lp[:, 3:7]), torch.tensor(qm_conj)).numpy()
    d *= np.sign(d[:, :1])
    w_fd = 2 * d[:, 1:] / (2 * eps)
    el = max(el, np.abs(lin[k, :len(LINKS)].numpy() - v_fd).max())
    ea = max(ea, np.abs(ang[k, :len(LINKS)].numpy() - w_fd).max())
    scale = max(scale, np.abs(v_fd).max())
  print(f"[2] 速度 vs Stonefish 位姿中心差分（eps {eps:g} s），{n} 个随机状态：线速度误差 max {el:.1e} m/s"
        f"（速度最大 {scale:.2f}），角速度误差 max {ea:.1e} rad/s")
  assert el < 1e-6 and ea < 1e-6


def test_refs(refs_dir):
  d = Path(refs_dir)
  meta = json.loads((d / "meta.json").read_text())
  names = json.loads((d / "joint_names.json").read_text())
  assert names == JOINTS, (names, JOINTS)
  link, off = meta.get("tcp_link", "EEBaseLink"), meta["tcp_offset"]
  assert link in ("ee_base_link", "EEBaseLink") and abs(off - ALPHA5_SITES["tcp"][1][2]) < 1e-12, (link, off)
  F = {k: torch.from_numpy(np.load(d / f"{k}.npy")).double() for k in
       ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")}
  fk = ForwardKinematics(ALPHA5_SCN, ["tcp"], names, sites=ALPHA5_SITES, dtype=torch.float64)
  pos, quat, lin, _ = fk(F["root_pos_w"], F["root_quat_w"], F["joint_pos"], F["root_lin_vel_w"],
                         F["root_ang_vel_w"], F["joint_vel"])
  pos, quat, lin = pos[:, 0].numpy(), quat[:, 0].numpy(), lin[:, 0].numpy()
  ee_pos, ee_quat = np.load(d / "ee_pos_w.npy"), np.load(d / "ee_quat_w.npy")
  ep = np.abs(pos - ee_pos).max()
  er = quat_angle(quat, ee_quat.astype(np.float64)).max()
  # TCP 速度 vs 存下的 TCP 位置的中心差分（50 Hz，含 O(dt²) 截断误差，只作量级参考）
  ends = np.load(d / "motion_ends.npy")
  starts = np.r_[0, ends[:-1]]
  dt = 1.0 / meta["ctrl_hz"]
  fd, vv = [], []
  for s, e in zip(starts, ends):
    fd.append((ee_pos[s + 2:e] - ee_pos[s:e - 2]) / (2 * dt))
    vv.append(lin[s + 1:e - 1])
  fd, vv = np.concatenate(fd), np.concatenate(vv)
  e_fd = np.linalg.norm(vv - fd, axis=1)
  print(f"[3] 参考轨迹 {d.name}（{len(pos)} 帧）：TCP 位置 vs Stonefish 存值 max {ep:.1e} m，姿态 max {er:.1e} rad；"
        f"TCP 速度 vs 位置中心差分 p50/RMSE {np.median(e_fd):.1e}/{np.sqrt((e_fd ** 2).mean()):.1e} m/s"
        f"（速度 RMS {np.sqrt((vv ** 2).mean()):.3f}）")
  assert ep < 1e-4 and er < 1e-5


def test_gradcheck():
  x = torch.as_tensor(random_states(np.random.default_rng(2), 3), dtype=torch.float64)
  parts = [x[:, 0:3], x[:, 3:7], x[:, 13:13 + NQ], x[:, 7:10], x[:, 10:13], x[:, 13 + NQ:]]
  parts = [p.clone().requires_grad_(True) for p in parts]
  ok = torch.autograd.gradcheck(lambda *a: torch.cat([o.flatten(-2) for o in FK(*a)], -1), parts)
  print(f"[4] gradcheck（对基座位姿/速度、关节角/角速度求导）: {'通过' if ok else '失败'}")
  assert ok


def bench(n=131072, reps=10):
  fk = ForwardKinematics(ALPHA5_SCN, ["base_link", "m2_joint_link", "m1_link", "tcp"], JOINTS,
                         sites=ALPHA5_SITES, dtype=torch.float32)
  x = torch.as_tensor(random_states(np.random.default_rng(3), n), dtype=torch.float32)
  args = (x[:, 0:3], x[:, 3:7], x[:, 13:13 + NQ], x[:, 7:10], x[:, 10:13], x[:, 13 + NQ:])
  fk(*args)
  t0 = time.time()
  for _ in range(reps):
    fk(*args)
  dt = (time.time() - t0) / reps
  print(f"[5] CPU float32，{n} 个状态 × 4 个输出，{torch.get_num_threads()} 线程：{dt * 1e3:.0f} ms/次")


if __name__ == "__main__":
  test_poses()
  test_velocities()
  if len(sys.argv) > 1:
    test_refs(sys.argv[1])
  test_gradcheck()
  bench()
