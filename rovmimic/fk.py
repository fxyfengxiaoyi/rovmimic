"""BlueROV2 + Alpha5 的正运动学：PyTorch 实现，可微、可批量，由 Stonefish 场景文件直接构建。

接口与 mjlab 的 ForwardKinematics 相同，mjlab MotionLoader / SuperMimic 可以直接使用：
  fk(root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel)
    -> pos_w (..., R, 3), quat_w (..., R, 4) wxyz, lin_vel_w (..., R, 3), ang_vel_w (..., R, 3)
  fk.joint_names  关节列顺序，必须与参考轨迹的 joint_names.json、sfenv 状态向量一致
  fk.body_names   输出的 R 个连杆 / 附着点

约定与 Stonefish 一致（FeatherstoneRobot::BuildKinematicStructure、ScenarioParser::ParseTransform）：
  - 根是 base_link 的原点系（不是质心系）。root_lin_vel 是这个原点的速度，各角速度都在世界系表示；
  - 子连杆原点系 = 父连杆原点系 · 关节 origin · 关节运动。
    转动关节绕子连杆原点、沿子系里的 axis 转 q；移动关节沿子系里的 axis 平移 q；
  - origin 的 rpy 按 ZYX 组合：R = Rz(yaw)·Ry(pitch)·Rx(roll)（Stonefish 编译时定义了 BT_EULER_DEFAULT_ZYX）；
  - 输出的线速度是各连杆原点（附着点则是该点）的速度。
附着点（sites）是固连在某个连杆上的点，例如 TCP = ee_base_link 沿局部 +z 偏移 0.075 m。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FIXED, REVOLUTE, PRISMATIC = 0, 1, 2
_TYPES = {"fixed": FIXED, "revolute": REVOLUTE, "prismatic": PRISMATIC}

ALPHA5_SCN = Path(__file__).resolve().parents[1] / "stonefish_bench" / "scenarios" / "rov_alpha5.scn"
# TCP：U0 抓取时物体中心所在的点（依据见 tools/usim_make_refs.py）
ALPHA5_SITES = {"tcp": ("ee_base_link", (0.0, 0.0, 0.075))}


@dataclass(frozen=True)
class Joint:
  name: str
  type: int
  parent: str
  child: str
  xyz: tuple[float, float, float]
  quat: tuple[float, float, float, float]  # wxyz
  axis: tuple[float, float, float]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
  """ZYX 欧拉角 -> wxyz，与 Stonefish 的 Quaternion(yaw, pitch, roll) 相同。"""
  cr, sr = np.cos(roll / 2), np.sin(roll / 2)
  cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
  cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
  return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


def _vec(text: str | None, n: int = 3) -> tuple[float, ...]:
  v = tuple(float(s) for s in (text or "").split())
  if len(v) != n:
    raise ValueError(f"expected {n} numbers, got {text!r}")
  return v


def parse_robot(scn_path: str | Path) -> tuple[str, str, dict[str, Joint]]:
  """读取场景里唯一的 <robot>：返回 (机器人名, 基座连杆名, {子连杆名: 连接它的关节})。"""
  robots = ET.parse(scn_path).getroot().findall(".//robot")
  if len(robots) != 1:
    raise ValueError(f"{scn_path}: expected one <robot>, found {len(robots)}")
  robot = robots[0]
  base = robot.find("base_link")
  if base is None:
    raise ValueError(f"{scn_path}: <robot> has no <base_link>")
  joints: dict[str, Joint] = {}
  for el in robot.findall("joint"):  # 只取 <robot> 的直接子元素；执行器/传感器里的 <joint name=.../> 不算
    typ = _TYPES[el.get("type")]
    parent, child, origin = el.find("parent"), el.find("child"), el.find("origin")
    if parent is None or child is None or origin is None:
      raise ValueError(f"joint {el.get('name')}: parent/child/origin missing")
    axis = (0.0, 0.0, 0.0)
    if typ != FIXED:
      a = np.array(_vec(el.find("axis").get("xyz")))
      axis = tuple((a / np.linalg.norm(a)).tolist())
    cname = child.get("name")
    if cname in joints:
      raise ValueError(f"link {cname} has two parent joints")
    joints[cname] = Joint(el.get("name"), typ, parent.get("name"), cname, _vec(origin.get("xyz")),
                          rpy_to_quat(*_vec(origin.get("rpy"))), axis)
  return robot.get("name"), base.get("name"), joints


# ---------------------------------------------------------------- 四元数（wxyz），支持广播
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw], -1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  w, u = q[..., :1], q[..., 1:]
  t = 2.0 * torch.linalg.cross(u, v, dim=-1)
  return v + w * t + torch.linalg.cross(u, t, dim=-1)


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  return torch.linalg.cross(a, b, dim=-1)


class ForwardKinematics:
  """从根（base_link 原点系）到 body_names 中各连杆 / 附着点的运动学链，以及沿链的批量计算。"""

  def __init__(self, scn_path: str | Path, body_names: list[str], joint_names: list[str],
               sites: dict[str, tuple[str, tuple[float, float, float]]] | None = None,
               dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
    _, base, joints = parse_robot(scn_path)
    sites = dict(sites or {})
    moving = [j.name for j in joints.values() if j.type != FIXED]
    if sorted(joint_names) != sorted(moving):
      raise ValueError(f"joint_names {joint_names} must be a permutation of the scenario's {moving}")
    col = {n: i for i, n in enumerate(joint_names)}

    # 第 0 步是根；其余每步 = 一个连杆或附着点，父步一定在它之前
    meta: list[tuple[int, int, int]] = []  # (父步, 类型, 关节列)
    consts: list[tuple[float, ...]] = []   # xyz(3) + quat(4) + axis(3)
    step_of = {base: 0}

    def add(name: str) -> int:
      if name in step_of:
        return step_of[name]
      if name in sites:
        parent, off = sites[name]
        entry, c = (add(parent), FIXED, -1), (*off, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
      elif name in joints:
        j = joints[name]
        entry, c = (add(j.parent), j.type, col.get(j.name, -1)), (*j.xyz, *j.quat, *j.axis)
      else:
        raise ValueError(f"unknown link or site {name!r}")
      meta.append(entry)
      consts.append(c)
      step_of[name] = len(meta)
      return step_of[name]

    self.base_name = base
    self.body_names = list(body_names)
    self.joint_names = list(joint_names)
    self._rows = [add(b) for b in self.body_names]
    self._meta = meta
    self._consts64 = torch.tensor(consts, dtype=torch.float64).reshape(-1, 10)
    self._cache: dict[tuple, torch.Tensor] = {}
    self._get_consts(dtype, torch.device(device))

  def _get_consts(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (dtype, device)
    if key not in self._cache:
      # 缓存可能在 inference_mode 里第一次创建（仿真环境），必须建成普通张量，世界模型反传时才能用
      with torch.inference_mode(False):
        self._cache[key] = self._consts64.to(dtype=dtype, device=device).clone()
    return self._cache[key]

  def __call__(self, root_pos: torch.Tensor, root_quat: torch.Tensor, joint_pos: torch.Tensor,
               root_lin_vel: torch.Tensor, root_ang_vel: torch.Tensor, joint_vel: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """世界系下 R 个连杆 / 附着点的位置、wxyz 四元数、线速度、角速度，形状 (..., R, ·)。"""
    C = self._get_consts(root_pos.dtype, root_pos.device)
    zero = root_pos.new_zeros(root_pos.shape)
    ident = torch.zeros_like(root_quat)
    ident[..., 0] = 1.0
    # 相对根的量（在根系表示）：位置、姿态、线速度、角速度
    pos, quat, lin, ang = [zero], [ident], [zero], [zero]
    for s, (p, typ, col) in enumerate(self._meta):
      xyz, q_off, axis = C[s, 0:3], C[s, 3:7], C[s, 7:10]
      P, Q, V, W = pos[p], quat[p], lin[p], ang[p]
      origin = P + quat_apply(Q, xyz.expand_as(P))
      Qo = quat_mul(Q, q_off.expand_as(Q))
      if typ == REVOLUTE:
        half = joint_pos[..., col:col + 1] / 2
        Qs = quat_mul(Qo, torch.cat([half.cos(), half.sin() * axis], -1))
        a_root = quat_apply(Qo, axis.expand_as(P))
        Ps = origin
        Ws = W + a_root * joint_vel[..., col:col + 1]
        Vs = V + _cross(W, origin - P)
      elif typ == PRISMATIC:
        a_root = quat_apply(Qo, axis.expand_as(P))
        Qs = Qo
        Ps = origin + a_root * joint_pos[..., col:col + 1]
        Ws = W
        Vs = V + _cross(W, Ps - P) + a_root * joint_vel[..., col:col + 1]
      else:
        Qs, Ps, Ws, Vs = Qo, origin, W, V + _cross(W, origin - P)
      pos.append(Ps)
      quat.append(Qs)
      lin.append(Vs)
      ang.append(Ws)

    rows = self._rows
    pos_b = torch.stack([pos[r] for r in rows], -2)
    quat_b = torch.stack([quat[r] for r in rows], -2)
    lin_b = torch.stack([lin[r] for r in rows], -2)
    ang_b = torch.stack([ang[r] for r in rows], -2)

    rq = root_quat / root_quat.norm(dim=-1, keepdim=True)
    rq = rq.unsqueeze(-2).expand_as(quat_b)
    w_r = root_ang_vel.unsqueeze(-2)
    pos_rel_w = quat_apply(rq, pos_b)
    pos_w = root_pos.unsqueeze(-2) + pos_rel_w
    quat_w = quat_mul(rq, quat_b)
    ang_w = w_r + quat_apply(rq, ang_b)
    lin_w = root_lin_vel.unsqueeze(-2) + _cross(w_r.expand_as(pos_rel_w), pos_rel_w) + quat_apply(rq, lin_b)
    return pos_w, quat_w, lin_w, ang_w


def alpha5_fk(body_names: list[str], joint_names: list[str], **kw) -> ForwardKinematics:
  """rov_alpha5.scn 的正运动学，自带 TCP 附着点 "tcp"。"""
  return ForwardKinematics(ALPHA5_SCN, body_names, joint_names, sites=ALPHA5_SITES, **kw)
