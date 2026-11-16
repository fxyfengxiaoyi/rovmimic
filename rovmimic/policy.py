"""推理：只用策略网络（actor.pt）和参考轨迹，从传感器读数和位姿算出动作。世界模型、价值网络、奖励都不用。

  pol = ROVPolicy.load("data/runs/logs/<task>/<时间>")        # 目录里要有 actor.pt 和 config.json（训练自动写）
  goal = pol.goal(motion, t, pos_meas, quat_meas)               # 230 维参考窗口，用测量的位姿算
  action = pol(obs, goal)                                       # obs 36 = [sensor 23, 上一步动作 13]，输出 13 维
  cmd = to_sim_command(action, q)                               # 15 维 Stonefish 指令（8 推进器 + 7 舵机目标）

各向量的布局见 readme/pipeline.md 1.2–1.4 和第 4 节。输入归一化统计在网络里，直接喂原始读数。

导出成不依赖本仓库的单文件（输入 obs (B, 36)、goal (B, 230)，输出 action (B, 13)）：
  python -I rovmimic/policy.py export <run_dir 或 checkpoints/it_XXXXXXX> [--onnx]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WS = Path(__file__).resolve().parents[1]
if str(WS) not in sys.path:
  sys.path.insert(0, str(WS))

from rovmimic import servo as S  # noqa: E402
from rovmimic import state as ST  # noqa: E402

OBS_DIM = 23 + S.NUM_ACTIONS  # sensor + 上一步动作
ACTOR_KEYS = ("latent_dim", "actor_hidden_dims", "latent_bc_weight", "history_length", "slot_mix")


def _runner_cfg(run_dir: Path | None) -> dict:
  """策略网络的结构参数：优先读训练时写的 config.json，没有就用 ROVRunnerCfg 的默认值。"""
  from rovmimic.runner import ROVRunnerCfg

  base = ROVRunnerCfg()
  cfg = {k: getattr(base, k) for k in ACTOR_KEYS}
  path = run_dir / "config.json" if run_dir is not None else None
  if path is not None and path.is_file():
    saved = json.loads(path.read_text())["runner_cfg"]
    cfg.update({k: saved[k] for k in ACTOR_KEYS if k in saved})
  elif path is not None:
    print(f"[WARN] {path} not found; using default actor sizes")
  cfg["actor_hidden_dims"] = tuple(cfg["actor_hidden_dims"])
  return cfg


def build_actor(cfg: dict, num_goals: int, device="cpu"):
  from supermimic.models import MultiEncoderAE
  from supermimic.tasks.task_spec import TaskSpec

  if cfg["history_length"] != 1:
    raise ValueError("ROV policy uses history_length=1")
  task = TaskSpec(name="motion", goal_dim=num_goals, extract_training_goal=None)
  return MultiEncoderAE(obs_dim=OBS_DIM, action_dim=S.NUM_ACTIONS, task_specs=[task], slot_mix=cfg["slot_mix"],
                        latent_dim=cfg["latent_dim"], hidden_dims=cfg["actor_hidden_dims"],
                        latent_bc_weight=cfg["latent_bc_weight"], history_length=1).to(device)


class ActorNet(nn.Module):
  """MultiEncoderAE 的推理前向，去掉 TensorDict，便于 TorchScript / ONNX 导出：
  goal -> 归一化 -> 编码器 -> tanh 潜变量；[潜变量, 归一化 obs] -> 解码器 -> 动作。"""

  def __init__(self, actor):
    super().__init__()
    self.goal_norm = actor.goal_normalizers["motion"]
    self.encoder = actor.encoders["motion"]
    self.obs_norm = actor.obs_normalizer
    self.decoder = actor.decoder

  def forward(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    z = torch.tanh(self.encoder(self.goal_norm(goal)))
    h = self.obs_norm(obs.unsqueeze(1)).flatten(1)
    return self.decoder(torch.cat([z, h], -1))


class ROVPolicy:
  def __init__(self, net: ActorNet, device="cpu"):
    self.net = net.to(device).eval()
    self.device = torch.device(device)

  @classmethod
  def load(cls, path: str | Path, device="cpu") -> "ROVPolicy":
    """path：含 actor.pt 的目录（训练日志目录或 checkpoints/it_XXXXXXX），或 actor.pt 文件本身。"""
    path = Path(path)
    run_dir, actor_pt = (path, path / "actor.pt") if path.is_dir() else (path.parent, path)
    actor = build_actor(_runner_cfg(run_dir), ST.goal_dim(), device)
    actor.load_state_dict(torch.load(actor_pt, map_location=device))
    actor.eval()
    return cls(ActorNet(actor), device)

  @staticmethod
  def goal(motion, t: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """参考窗口：参考帧 t, t+5, …, t+45 相对当前（测量的）位姿，见 rovmimic/state.goal。"""
    return ST.goal(motion, t, pos, quat)

  @torch.no_grad()
  def __call__(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return self.net(obs.to(self.device, torch.float32), goal.to(self.device, torch.float32))


def to_sim_command(action: torch.Tensor | np.ndarray, q: np.ndarray) -> np.ndarray:
  """13 维动作 -> 15 维 Stonefish 指令：推进器 clip 到 [-1, 1]；臂关节目标 = q + 0.01·clip(a)；
  夹爪目标 = (clip(a) + 1)/2 × 0.015，两指 0.83 倍联动（rovmimic/servo.py）。q 是当前 7 个关节角。"""
  a = action.detach().cpu().double().numpy() if torch.is_tensor(action) else np.asarray(action, np.float64)
  return S.sim_action(a, np.asarray(q, np.float64))


def export(run_dir: str, onnx: bool = False) -> list[Path]:
  pol = ROVPolicy.load(run_dir, "cpu")
  d = Path(run_dir) if Path(run_dir).is_dir() else Path(run_dir).parent
  obs, goal = torch.randn(4, OBS_DIM), torch.randn(4, ST.goal_dim())
  ref = pol(obs, goal)
  out = []
  ts = torch.jit.trace(pol.net, (obs, goal))
  ts_path = d / "policy.ts"
  ts.save(str(ts_path))
  err = (torch.jit.load(str(ts_path))(obs, goal) - ref).abs().max().item()
  print(f"[export] {ts_path}  max diff vs actor {err:.1e}")
  out.append(ts_path)
  if onnx:
    onnx_path = d / "policy.onnx"
    torch.onnx.export(pol.net, (obs, goal), str(onnx_path), input_names=["obs", "goal"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "goal": {0: "batch"}, "action": {0: "batch"}},
                      dynamo=False)
    feeds = {"obs": obs.numpy(), "goal": goal.numpy()}
    try:
      import onnxruntime as ort

      o = ort.InferenceSession(str(onnx_path)).run(None, feeds)[0]
    except ImportError:  # 没装 onnxruntime 时用 onnx 自带的参考实现核对
      from onnx.reference import ReferenceEvaluator

      o = ReferenceEvaluator(str(onnx_path)).run(None, feeds)[0]
    print(f"[export] {onnx_path}  max diff vs actor {np.abs(o - ref.numpy()).max():.1e}")
    out.append(onnx_path)
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  e = sub.add_parser("export")
  e.add_argument("run_dir")
  e.add_argument("--onnx", action="store_true")
  args = ap.parse_args()
  export(args.run_dir, args.onnx)
