"""世界模型里的“想象”环境（策略在这里用 BPTT 学）：在 SuperMimic 的 MultiTaskWorldModelEnv 上改三处。

  _wm_step      奖励和软失败判定用 rovmimic/tracking.py（与仿真环境同一个函数），需要参考帧的关节目标
  _update_goal  世界模型 / 价值网络用真实位置算目标；策略的目标用带测量误差的位置算
  _on_reset     每段想象开始时抽一个水平定位偏差（段内不变，0.64 s 内漂移可以忽略）
"""

from __future__ import annotations

import torch
from supermimic.envs.multi_task_world_model_env import MultiTaskWorldModelEnv

from rovmimic import state as ST
from rovmimic import tracking as TR


class ROVWorldModelEnv(MultiTaskWorldModelEnv):
  def __init__(self, *, env, **kwargs) -> None:
    super().__init__(env=env, **kwargs)
    self._rewards = env.cfg.rewards
    self._terminations = env.cfg.terminations
    self._sensors = env.cfg.sensors
    self._dt = env.step_dt

  def _measured_aux(self) -> torch.Tensor:
    B, dev = self.aux_state.shape[0], self.aux_state.device
    depth = (torch.rand(B, 1, device=dev) * 2 - 1) * self._sensors.depth
    noise = torch.cat([self._pos_drift, depth], -1)
    return torch.cat([self.aux_state[:, :3] + noise, self.aux_state[:, 3:7]], -1)

  def _on_reset(self, batch) -> None:
    B, dev = batch.shape[0], batch["state"].device
    self._pos_drift = (torch.rand(B, 2, device=dev) * 2 - 1) * self._sensors.wm_drift
    super()._on_reset(batch)

  def _update_goal(self) -> None:
    t = self.motion_times
    self._ref_body_state = self.motion.body_state(t)
    self._ref_joint_pos = self.motion.joint_pos[t]
    self._wm_goal_obs, _ = self.representation.compute_wm_goal(self.motion, t, self.aux_state)
    _, self._actor_goal_obs = self.representation.compute_wm_goal(self.motion, t, self._measured_aux())
    self._build_task_goals()

  def _wm_step(self, rep_state, aux_state, action, last_action, sim_ctx):
    next_aux, next_rep = self.dynamics(rep_state, action, aux_state, sim_ctx)
    next_joint_state = self.representation.joint_state(next_rep)
    reward, _, errors = TR.evaluate(ST.full_from(next_aux, next_rep), action, last_action, self.representation.fk,
                                    self._ref_body_state, self._ref_joint_pos, self._rewards, self._dt)
    term_prob = TR.fail_prob(errors, self._terminations, self._termination_softness)
    return next_aux, next_rep, next_joint_state, term_prob, reward
