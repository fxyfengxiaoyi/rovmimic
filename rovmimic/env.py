"""Stonefish 跟踪环境：多进程 Stonefish（rovmimic/vec_sim.py）+ 参考轨迹，
对外提供 SuperMimic 训练循环用到的 mjlab ManagerBasedRlEnv 接口（同时充当 RslRlVecEnvWrapper）。

每步的顺序与 mjlab 相同：
  1. 策略动作 -> 推进器设定值 + 舵机目标（rovmimic/servo.py），推进一个控制步（0.02 s，2 个物理步）；
  2. 用这一步之后的状态和参考帧 time_steps[t] 算奖励和失败判定（rovmimic/tracking.py，与世界模型同一个函数）；
  3. 失败、片段结束、超时的环境重置：从参考轨迹按回报采样一帧 f，状态设为参考 + 小扰动（RSI）；
  4. 所有环境 time_steps += 1（刚重置的变成 f + 1）；
  5. 返回观测（刚重置的环境返回重置后的观测）。

环境保存 Stonefish 的真实状态（含舵机步末角速度读数），对外的 full_state 里关节角速度是上一步平均速度。
观测：
  full_state (27)  真实状态，布局见 rovmimic/state.py
  actor (36)       传感器读数（不含噪声，噪声由 SuperMimic 统一加）+ 上一步动作
  goal (230)       参考窗口（用真实位置算，只用来确定维数；策略的目标由训练循环用带误差的位置另算）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from rovmimic import servo as S
from rovmimic import state as ST
from rovmimic.fk import quat_mul
from rovmimic.motion import ROVMotion
from rovmimic.tracking import (RewardTermCfg, TerminationTermCfg, default_rewards,
                               default_terminations, evaluate, fail_flags)


@dataclass
class SensorNoiseCfg:
  """策略看到的读数误差（见 readme/system_overview.md 1.7）。前 6 项是均匀噪声的半宽，每步独立。"""
  projected_gravity: float = 0.02
  ang_vel: float = 0.05        # rad/s
  dvl: float = 0.02            # m/s
  joint_pos: float = 0.005     # rad
  joint_vel: float = 0.05      # rad/s
  actions: float = 0.0
  depth: float = 0.01          # 深度计 (m)，每步独立
  drift_init: float = 0.05     # 水平定位：回合开始时的偏差，均匀分布半宽 (m)
  drift_step: float = 0.002    # 水平定位：每步随机游走的标准差 (m)
  drift_max: float = 0.15      # 水平定位：偏差上限 (m)
  wm_drift: float = 0.1        # 世界模型想象时，每段的水平偏差（段内不变），均匀分布半宽 (m)

  def proprio_half_widths(self, nq: int = 7, na: int = S.NUM_ACTIONS) -> list[float]:
    """与 [sensor(23), last_action(13)] 逐维对应的噪声半宽。"""
    per_term = [self.projected_gravity, self.ang_vel, self.dvl, self.joint_pos, self.joint_vel]
    out = [h for h, w in zip(per_term, ST.sensor_widths(nq), strict=True) for _ in range(w)]
    return out + [self.actions] * na


@dataclass
class ResetNoiseCfg:
  """RSI：在参考帧上加的均匀扰动（半宽）。"""
  pos: float = 0.05            # m，xyz
  rp: float = 0.05             # rad，横滚、俯仰
  yaw: float = 0.1             # rad
  lin_vel: float = 0.05        # m/s
  ang_vel: float = 0.05        # rad/s
  arm_joint: float = 0.05      # rad，臂关节（夹爪不加）


@dataclass
class StonefishTrackingEnvCfg:
  refs_dir: str = "data/refs/usim_train_v1"
  num_clips: int | None = None          # 只用前 n 条片段（调试用）
  num_envs: int = 4096                  # 与原版相同（2026-11-16 实测 1024 / 4096 两档每小时更新次数相近，4096 每次更新的新数据多 4 倍）
  num_workers: int = field(default_factory=lambda: min(64, os.cpu_count() or 1))  # 4096 环境：64 进程 150 ms/步，32 进程 180 ms/步
  phys_hz: float = 100.0
  ctrl_hz: float = 50.0
  decimation: int = 2                   # 每个控制步的物理步数 = phys_hz / ctrl_hz
  episode_length_s: float = 10.0
  sampling_mode: str = "external"       # external（SuperMimic 按回报采样）/ uniform / start
  rewards: dict[str, RewardTermCfg] = field(default_factory=default_rewards)
  terminations: dict[str, TerminationTermCfg] = field(default_factory=default_terminations)
  events: dict = field(default_factory=dict)  # 不做域随机化
  sensors: SensorNoiseCfg = field(default_factory=SensorNoiseCfg)
  reset: ResetNoiseCfg = field(default_factory=ResetNoiseCfg)


class MotionCommand:
  """mjlab MotionCommand 里 SuperMimic 用到的部分。"""

  def __init__(self, motion: ROVMotion, num_envs: int, sampling_mode: str, step_dt: float, device):
    self.motion = motion
    self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.bin_count = int(motion.time_step_total // (0.2 / step_dt)) + 1  # 与 mjlab 相同
    self.cfg = SimpleNamespace(sampling_mode=sampling_mode)
    self.motion_anchor_body_index = 0
    self.device = device
    self._external_probs = torch.full((self.bin_count,), 1.0 / self.bin_count, device=device)

  def set_sampling_distribution(self, probs: torch.Tensor) -> None:
    probs = probs.detach().to(self.device, dtype=torch.float).clamp_min(0.0)
    total = probs.sum()
    if probs.numel() != self.bin_count or not torch.isfinite(total) or total <= 0:
      return
    self._external_probs = probs / total

  def sample_frames(self, n: int) -> torch.Tensor:
    motion, T = self.motion, self.motion.time_step_total
    mode = self.cfg.sampling_mode
    if mode == "external":  # 与 mjlab MotionCommand._sample_bins 相同
      total = T - 2
      b = torch.multinomial(self._external_probs, n, replacement=True).clamp_(max=self.bin_count - 1)
      lo = (b * total) // self.bin_count
      span = (((b + 1) * total) // self.bin_count - lo).clamp_(min=1)
      frames = (lo + (torch.rand(n, device=self.device) * span).long()).clamp_(max=total)
    elif mode == "uniform":
      frames = torch.randint(0, T - 1, (n,), device=self.device)
    elif mode == "start":
      starts = torch.cat([motion.motion_ends.new_zeros(1), motion.motion_ends[:-1]])
      frames = starts[torch.randint(0, motion.num_clips, (n,), device=self.device)]
    else:
      raise ValueError(f"unknown sampling_mode {mode!r}")
    on_last = torch.isin(frames, motion.motion_ends - 1)  # 不从片段最后一帧开始
    return torch.where(on_last, frames - 1, frames)


class TerminationManager:
  def __init__(self, num_envs: int, device):
    z = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.terminated, self.time_outs = z.clone(), z.clone()
    self._terms: dict[str, torch.Tensor] = {}

  def get_term(self, name: str) -> torch.Tensor:
    return self._terms[name]


class _Single:
  def __init__(self, term):
    self._term = term

  def get_term(self, name: str):
    return self._term


class StonefishTrackingEnv:
  def __init__(self, cfg: StonefishTrackingEnvCfg, device: str = "cpu"):
    from rovmimic.vec_sim import VecSim

    if abs(cfg.phys_hz / cfg.ctrl_hz - cfg.decimation) > 1e-9:
      raise ValueError("decimation must equal phys_hz / ctrl_hz")
    self.cfg = cfg
    self.device = torch.device(device)
    self.unwrapped = self
    self.num_envs = cfg.num_envs
    self.num_actions = S.NUM_ACTIONS
    self.step_dt = 1.0 / cfg.ctrl_hz
    self.max_episode_length = int(round(cfg.episode_length_s / self.step_dt))
    self.motion = ROVMotion(cfg.refs_dir, self.device, cfg.num_clips)
    self.fk = self.motion.fk
    self.sim = VecSim(cfg.num_envs, cfg.num_workers, phys_hz=cfg.phys_hz, ctrl_hz=cfg.ctrl_hz)
    if tuple(self.sim.joint_names) != S.JOINT_NAMES or tuple(self.sim.servo_names) != S.SERVO_NAMES:
      raise ValueError("scenario joint order differs from rovmimic/servo.py")
    if abs(self.sim.ctrl_dt - self.step_dt) > 1e-9:
      raise ValueError(f"sim ctrl_dt {self.sim.ctrl_dt} != {self.step_dt}")

    N = self.num_envs
    self.motion_command = MotionCommand(self.motion, N, cfg.sampling_mode, self.step_dt, self.device)
    self.command_manager = _Single(self.motion_command)
    self.action_manager = SimpleNamespace(action=torch.zeros(N, S.NUM_ACTIONS, device=self.device))
    self.termination_manager = TerminationManager(N, self.device)
    self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=self.device)
    self.pos_drift = torch.zeros(N, 3, device=self.device)
    self._x = np.zeros((N, self.sim.nx))  # Stonefish 真实状态
    self._vj = torch.zeros(N, len(S.JOINT_NAMES), device=self.device)  # 上一步平均关节速度
    self._obs: TensorDict | None = None
    self.reset()

  # ------------------------------------------------------------------ 状态

  def full_state(self) -> torch.Tensor:
    x = torch.from_numpy(self._x).to(self.device, torch.float32)
    return torch.cat([x[:, 0:3], x[:, 3:7], x[:, 13:20], x[:, 7:10], x[:, 10:13], self._vj], -1)

  def position_noise(self) -> torch.Tensor:
    """这一步的位置测量误差 (N, 3)：水平是回合内缓慢漂移的偏差，竖直是深度计的小噪声。"""
    sc = self.cfg.sensors
    z = (torch.rand(self.num_envs, device=self.device) * 2 - 1) * sc.depth
    return torch.cat([self.pos_drift[:, :2], z[:, None]], -1)

  def _compute_obs(self) -> TensorDict:
    full = self.full_state()
    rep = ST.represent(full)
    actor = torch.cat([ST.sensor(rep), self.action_manager.action], -1)
    goal = ST.goal(self.motion, self.motion_command.time_steps, full[:, :3], full[:, 3:7])
    return TensorDict({"full_state": full, "actor": actor, "goal": goal}, batch_size=[self.num_envs],
                      device=self.device)

  # ------------------------------------------------------------------ 重置

  def _reset_idx(self, ids: torch.Tensor, frames: torch.Tensor | None = None) -> None:
    n = int(ids.numel())
    if n == 0:
      return
    rc, dev = self.cfg.reset, self.device
    frames = self.motion_command.sample_frames(n) if frames is None else frames.to(dev, torch.long)
    ref = self.motion.full_state(frames).double()
    pos, quat, q, lin, ang, vj = ST.split_full(ref)

    def u(*shape, h):
      return (torch.rand(*shape, device=dev, dtype=torch.float64) * 2 - 1) * h

    pos = pos + u(n, 3, h=rc.pos)
    rpy = torch.cat([u(n, 2, h=rc.rp), u(n, 1, h=rc.yaw)], -1)
    cr, sr = torch.cos(rpy[:, 0] / 2), torch.sin(rpy[:, 0] / 2)
    cp, sp = torch.cos(rpy[:, 1] / 2), torch.sin(rpy[:, 1] / 2)
    cy, sy = torch.cos(rpy[:, 2] / 2), torch.sin(rpy[:, 2] / 2)
    dq = torch.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                      cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], -1)
    quat = quat_mul(dq, quat)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    lin = lin + u(n, 3, h=rc.lin_vel)
    ang = ang + u(n, 3, h=rc.ang_vel)
    q = q.clone()
    lo = torch.tensor([S.LIMITS[j][0] for j in S.ARM], device=dev, dtype=torch.float64)
    hi = torch.tensor([S.LIMITS[j][1] for j in S.ARM], device=dev, dtype=torch.float64)
    q[:, 0:4] = torch.minimum(torch.maximum(q[:, 0:4] + u(n, 4, h=rc.arm_joint), lo), hi)

    ids_np = ids.cpu().numpy()
    # sfenv 状态：[pos, quat, lin_w, ang_w, q, qd]；qd 用参考的上一步平均速度
    self._x[ids_np] = torch.cat([pos, quat, lin, ang, q, vj], -1).cpu().numpy()
    self._vj[ids] = vj.float()
    self.motion_command.time_steps[ids] = frames + 1
    self.episode_length_buf[ids] = 0
    self.action_manager.action[ids] = 0.0
    sc = self.cfg.sensors
    drift = (torch.rand(n, 3, device=dev) * 2 - 1) * sc.drift_init
    drift[:, 2] = 0.0
    self.pos_drift[ids] = drift

  @torch.inference_mode()
  def reset(self):
    self._reset_idx(torch.arange(self.num_envs, device=self.device))
    self._obs = self._compute_obs()
    return self._obs, {}

  @torch.inference_mode()
  def reset_to(self, frames: torch.Tensor):
    """所有环境从指定的参考帧开始（评估用；扰动仍按 cfg.reset，评估时可设为 0）。"""
    if frames.numel() != self.num_envs:
      raise ValueError(f"need {self.num_envs} frames, got {frames.numel()}")
    self._reset_idx(torch.arange(self.num_envs, device=self.device), frames)
    self._obs = self._compute_obs()
    return self._obs, {}

  # ------------------------------------------------------------------ 推进

  def get_observations(self) -> TensorDict:
    return self._obs

  @torch.inference_mode()
  def step(self, actions: torch.Tensor):
    """与 mjlab 一样在 inference_mode 里推进；环境的持久状态张量只做原地更新。"""
    cfg, dt = self.cfg, self.step_dt
    a = actions.detach().to(self.device, torch.float32)
    prev = self.action_manager.action.clone()
    q = self._x[:, 13:20].copy()
    x_new = self.sim.step(self._x, S.sim_action(a.cpu().double().numpy(), q))
    finite_np = np.isfinite(x_new).all(1)
    x_new[~finite_np] = self._x[~finite_np]
    self._vj.copy_(torch.from_numpy((x_new[:, 13:20] - q) / dt))
    self._x = x_new
    self.action_manager.action.copy_(a)
    self.episode_length_buf += 1

    cmd = self.motion_command
    t = cmd.time_steps
    full = self.full_state()
    reward, terms, errors = evaluate(full, a, prev, self.fk, self.motion.body_state(t), self.motion.joint_pos[t],
                                     cfg.rewards, dt)
    finite = torch.from_numpy(finite_np).to(self.device)
    reward = torch.where(finite, reward, torch.zeros_like(reward))
    fails = fail_flags(errors, cfg.terminations)
    fails["nonfinite"] = ~finite
    terminated = torch.stack(list(fails.values()), -1).any(-1)
    motion_end = torch.isin(t + 1, self.motion.motion_ends)
    time_out = self.episode_length_buf >= self.max_episode_length
    time_outs = motion_end | time_out
    tm = self.termination_manager
    tm.terminated.copy_(terminated)
    tm.time_outs.copy_(time_outs)
    tm._terms = {"motion_end": motion_end, "time_out": time_out, **fails}
    dones = terminated | time_outs

    log = {f"Reward/{k}": v.mean() for k, v in terms.items()}
    log.update({f"Metrics/err_{k}": errors[k].mean() for k in ("anchor_pos", "anchor_ori", "tcp_pos")})
    log["Metrics/hull_depth_max"] = errors["hull"].max()
    log.update({f"Termination/{k}": v.float().mean() for k, v in tm._terms.items()})
    # 后几项是每个环境这一步的结果（重置之前），评估用：state 是这一步之后的 full_state，ref_t 是对比的参考帧
    extras = {"time_outs": time_outs, "log": log, "errors": errors, "fails": fails, "reward_terms": terms,
              "state": full, "ref_t": t.clone(), "terminated": terminated.clone()}

    cmd.time_steps += 1
    self._reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))
    sc = cfg.sensors
    self.pos_drift[:, :2] = (self.pos_drift[:, :2] + torch.randn_like(self.pos_drift[:, :2]) * sc.drift_step
                             ).clamp(-sc.drift_max, sc.drift_max)
    self._obs = self._compute_obs()
    return self._obs, reward, dones.long(), extras

  def close(self):
    self.sim.close()

