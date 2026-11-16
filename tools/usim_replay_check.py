"""回放一致性检查：在无界面 Stonefish 里用 USIM 记录的指令开环回放，和记录的传感器比较。

python -I usim_replay_check.py --robot nav|alpha5 [--leads 0,0.05,0.1] [--out res.json] ep1.parquet ...

两种评估：
  [短时多起点] 每隔 0.5 s 取一个起点 k，用第 k 帧的传感器构造初始状态（深度←压力，横滚/俯仰←比力，
              机体速度←DVL，角速度←IMU，偏航无关），回放之后 2 s 的指令，比较 DVL 速度、角速度、深度变化。
              对照组"保持"= 假设速度保持第 k 帧不变；仿真误差明显小于对照组才说明动力学对得上。
  [整条开环]   从第 0 帧开始回放整条 episode，比较速度曲线。开环会累积误差，主要看量级与形状。
抓取类任务只用夹爪闭合前的片段（闭合后抓着物体，物体质量/浮力不在我们的场景里）；
离海底 < 0.3 m 的起点也跳过（我们的场景没有海底，不考虑接触）。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "sfenv" / "python"))
import usim_lib as U  # noqa: E402

HORIZONS = (5, 10, 20)  # 帧（0.5 / 1 / 2 s）


def usable_frames(ep, robot):
  T = len(ep["pwm"])
  ok = ep["altitude"] >= 0.3
  if robot == "alpha5":
    closed = np.where(ep["q"][:, 0] > 1e-3)[0]
    if len(closed):
      ok[closed[0]:] = False
  return ok, T


def eval_multistart(sim, ep, robot, lead, H=max(HORIZONS)):
  A, sub = U.ctrl_actions(ep, robot, 50, lead)
  ok, T = usable_frames(ep, robot)
  errs = {h: {"sim": {"dvl": [], "gyro": [], "depth": []}, "hold": {"dvl": [], "gyro": [], "depth": []}}
          for h in HORIZONS}
  for k in range(0, T - H, 5):
    if not ok[k:k + H + 1].all():
      continue
    x0 = U.initial_state(ep, k, robot, sim.nq, sim.joint_names)
    X, _ = sim.rollout(x0, A[k * sub:(k + H) * sub])
    S = U.sensors_from_state(X[sub - 1::sub], sim.nq)  # 第 k+1 .. k+H 帧
    for h in HORIZONS:
      j = h - 1
      d_rec = (ep["pressure"][k + h] - ep["pressure"][k]) / U.PRESSURE_PER_M
      e = errs[h]
      e["sim"]["dvl"].append(S["dvl"][j] - ep["dvl"][k + h])
      e["sim"]["gyro"].append(S["gyro"][j] - ep["gyro"][k + h])
      e["sim"]["depth"].append(S["depth"][j] - x0[2] - d_rec)
      e["hold"]["dvl"].append(ep["dvl"][k] - ep["dvl"][k + h])
      e["hold"]["gyro"].append(ep["gyro"][k] - ep["gyro"][k + h])
      # 对照：深度按第 k 帧的垂向速度外推
      vz0 = (U.quat_to_mat(x0[3:7]) @ (ep["dvl"][k] * np.array([1, -1, -1])))[2]
      e["hold"]["depth"].append(vz0 * h / U.FPS - d_rec)
  return errs


def eval_openloop(sim, ep, robot, lead):
  A, sub = U.ctrl_actions(ep, robot, 50, lead)
  ok, T = usable_frames(ep, robot)
  n = int(np.argmin(ok)) if not ok.all() else T  # 第一个不可用帧之前
  if n < 20:
    return None
  x0 = U.initial_state(ep, 0, robot, sim.nq, sim.joint_names)
  X, _ = sim.rollout(x0, A[:(n - 1) * sub])
  S = U.sensors_from_state(X[sub - 1::sub], sim.nq)  # 第 1 .. n-1 帧
  out = {"frames": n, "seconds": n / U.FPS}
  for name, sec in (("5s", 5), ("10s", 10), ("all", None)):
    m = (n - 1) if sec is None else min(n - 1, sec * U.FPS)
    out[name] = {
      "dvl_rmse": np.sqrt(((S["dvl"][:m] - ep["dvl"][1:m + 1]) ** 2).mean(0)).round(4).tolist(),
      "gyro_rmse": np.sqrt(((S["gyro"][:m] - ep["gyro"][1:m + 1]) ** 2).mean(0)).round(4).tolist(),
      "dvl_rms_rec": np.sqrt((ep["dvl"][1:m + 1] ** 2).mean(0)).round(4).tolist(),
      "depth_err_end": float(S["depth"][m - 1] - x0[2] - (ep["pressure"][m] - ep["pressure"][0]) / U.PRESSURE_PER_M),
    }
  if robot == "alpha5":
    qs = U.sim_to_usim_joints(S["q"], sim.joint_names)
    out["joint_rmse_abcde"] = np.sqrt(((qs - ep["q"][1:n]) ** 2).mean(0)).round(4).tolist()
  return out


def summarize(all_errs):
  rows = {}
  for h in HORIZONS:
    r = {}
    for who in ("sim", "hold"):
      for key in ("dvl", "gyro", "depth"):
        v = np.concatenate([np.atleast_2d(np.array(e[h][who][key])).reshape(len(e[h][who][key]), -1)
                            for e in all_errs if len(e[h][who][key])], 0)
        r[f"{who}_{key}"] = np.sqrt((v ** 2).mean(0)).round(4).tolist()
        r["n"] = len(v)
    rows[h] = r
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--robot", choices=["nav", "alpha5"], required=True)
  p.add_argument("--leads", default="0.0")
  p.add_argument("--out", default=None)
  p.add_argument("episodes", nargs="+")
  args = p.parse_args()

  import sfenv
  sim = sfenv.Sim(str(U.SCENARIOS[args.robot]), U.DATA_DIR, 100.0, 50.0)
  eps = [U.load_episode(f) for f in args.episodes]
  eps = [e for e in eps if U.robot_for_task(e["task"]) == args.robot]
  print(f"robot={args.robot}: {len(eps)} episodes, tasks {sorted({e['task'] for e in eps})}")
  results = {}
  for lead in [float(x) for x in args.leads.split(",")]:
    errs = [eval_multistart(sim, e, args.robot, lead) for e in eps]
    s = summarize(errs)
    results[f"lead={lead}"] = {"multistart": s}
    print(f"\n== lead {lead:.2f} s  短时多起点 RMSE（sim = 仿真回放，hold = 速度保持不变的对照）")
    for h in HORIZONS:
      r = s[h]
      print(f"  {h / U.FPS:.1f} s (n={r['n']}): dvl sim {r['sim_dvl']} hold {r['hold_dvl']} | "
            f"gyro sim {r['sim_gyro']} hold {r['hold_gyro']} | depth sim {r['sim_depth']} hold {r['hold_depth']}")
  lead0 = float(args.leads.split(",")[0])
  ol = {}
  print(f"\n== 整条开环回放（lead {lead0:.2f} s）")
  for e in eps:
    r = eval_openloop(sim, e, args.robot, lead0)
    ol[e["episode"]] = r
    if r is None:
      continue
    extra = f" joints[a..e] {r['joint_rmse_abcde']}" if "joint_rmse_abcde" in r else ""
    print(f"  ep{e['episode']:5d} task{e['task']} {r['seconds']:5.1f}s | 前10s dvl rmse {r['10s']['dvl_rmse']} "
          f"(记录 rms {r['10s']['dvl_rms_rec']}) gyro {r['10s']['gyro_rmse']} | "
          f"全程 dvl {r['all']['dvl_rmse']} 深度终点误差 {r['all']['depth_err_end']:+.2f} m{extra}")
  results["openloop"] = ol
  if args.out:
    Path(args.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
  main()
