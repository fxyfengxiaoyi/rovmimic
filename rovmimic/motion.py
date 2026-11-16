"""参考轨迹加载器，字段与 mjlab 的 MotionLoader 相同，SuperMimic 可以直接使用。

  motion = ROVMotion("data/refs/usim_train_v1", device="cuda")
  motion.joint_pos / joint_vel (T, 7)，root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_w，
  motion.motion_ends (C,)，motion.time_step_total，motion.fk，
  motion.body_state(frames) -> 参考轨迹在这些帧上各连杆 / 附着点的世界系位姿和速度（正运动学现算，不预存）

连杆列表 BODY_NAMES = 奖励跟踪的连杆 TRACK_BODIES + 艇体检查用的其余点。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rovmimic import servo as S
from rovmimic.fk import ALPHA5_SCN, ALPHA5_SITES, ForwardKinematics
from rovmimic.terms import HULL_CHECK_BODIES, HULL_SITES

# 奖励跟踪的连杆：航行器 + 每个臂关节之后的第一个连杆 + TCP。第 0 个必须是航行器（根）。
TRACK_BODIES = ("base_link", "m2_1_1_link", "m2_joint_link", "m2_1_2_link", "ee_base_link", "tcp")
BODY_NAMES = TRACK_BODIES + tuple(b for b in HULL_CHECK_BODIES if b not in TRACK_BODIES)
# 用切片选连杆（不用列表索引，便于 torch.compile / CUDA graph）
TRACK = slice(0, len(TRACK_BODIES))     # 跟踪的连杆，0 是航行器
ARM_TRACK = slice(1, len(TRACK_BODIES))  # 跟踪的臂上连杆（含 TCP），姿势项只比这些
TCP_IDX = BODY_NAMES.index("tcp")
HULL = slice(2, len(BODY_NAMES))         # 艇体检查点
assert set(BODY_NAMES[HULL]) == set(HULL_CHECK_BODIES)
FIELDS = ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")


def make_fk(device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32) -> ForwardKinematics:
  return ForwardKinematics(ALPHA5_SCN, list(BODY_NAMES), list(S.JOINT_NAMES), sites={**ALPHA5_SITES, **HULL_SITES},
                           dtype=dtype, device=device)


class ROVMotion:
  def __init__(self, refs_dir: str | Path, device: str | torch.device = "cpu", num_clips: int | None = None):
    d = Path(refs_dir)
    joint_names = tuple(json.loads((d / "joint_names.json").read_text()))
    if joint_names != S.JOINT_NAMES:
      raise ValueError(f"{d}: joint order {joint_names} != {S.JOINT_NAMES}")
    ends = np.load(d / "motion_ends.npy").astype(np.int64)
    names = json.loads((d / "motion_names.json").read_text())
    if num_clips is not None:
      ends, names = ends[:num_clips], names[:num_clips]
    T = int(ends[-1])
    self.refs_dir = str(d)
    self.motion_names = names
    self.device = torch.device(device)
    for f in FIELDS:
      arr = np.load(d / f"{f}.npy", mmap_mode="r")[:T]
      setattr(self, f, torch.from_numpy(np.array(arr, dtype=np.float32)).to(self.device))  # 读进内存
    self.root_quat_w = self.root_quat_w / self.root_quat_w.norm(dim=-1, keepdim=True)
    self.motion_ends = torch.from_numpy(ends).to(self.device)
    self.time_step_total = T
    self.body_names = list(BODY_NAMES)
    self.fk = make_fk(self.device)

  @property
  def num_clips(self) -> int:
    return int(self.motion_ends.numel())

  def body_state(self, frames: torch.Tensor):
    """(E,) 帧号 -> 参考轨迹的 (pos (E,R,3), quat (E,R,4), lin_vel (E,R,3), ang_vel (E,R,3))，R = len(BODY_NAMES)。"""
    return self.fk(self.root_pos_w[frames], self.root_quat_w[frames], self.joint_pos[frames],
                   self.root_lin_vel_w[frames], self.root_ang_vel_w[frames], self.joint_vel[frames])

  def full_state(self, frames: torch.Tensor) -> torch.Tensor:
    """参考轨迹在这些帧的 full_state (E, 27)，布局见 rovmimic/state.py。"""
    return torch.cat([self.root_pos_w[frames], self.root_quat_w[frames], self.joint_pos[frames],
                      self.root_lin_vel_w[frames], self.root_ang_vel_w[frames], self.joint_vel[frames]], -1)

  def clip_start(self, frames: torch.Tensor) -> torch.Tensor:
    clip = torch.searchsorted(self.motion_ends, frames, right=True)
    starts = torch.cat([self.motion_ends.new_zeros(1), self.motion_ends[:-1]])
    return starts[clip.clamp(max=self.num_clips - 1)]
