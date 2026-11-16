"""检查 USIM（LeRobot v2.1）单个 episode 的字段、取值范围，以及 target_pos 和实际运动的关系。"""

import sys

import numpy as np
import pandas as pd

STATE = {"joint_pos": (0, 5), "pwm": (5, 13), "joint_v": (13, 18), "dvl_v": (18, 21), "imu_av": (21, 24),
         "imu_la": (24, 27), "pressure": (27, 28), "dvl_h": (28, 29)}

for path in sys.argv[1:]:
  df = pd.read_parquet(path)
  s = np.stack(df["observation.state"].to_numpy())
  a = np.stack(df["action"].to_numpy())
  tp = np.stack(df["target_pos"].to_numpy())
  ts = df["timestamp"].to_numpy()
  print(f"== {path.split('/')[-1]}  frames={len(df)}  task_index={df['task_index'].iloc[0]}  "
        f"dt={np.diff(ts).mean():.4f}±{np.diff(ts).std():.1e}  columns={list(df.columns)}")
  for k, (i, j) in STATE.items():
    x = s[:, i:j]
    print(f"  state.{k:9s} min {np.round(x.min(0), 3)}  max {np.round(x.max(0), 3)}")
  print(f"  action.joint_pos min {np.round(a[:, :5].min(0), 3)} max {np.round(a[:, :5].max(0), 3)}")
  print(f"  action.pwm       min {np.round(a[:, 5:].min(0), 1)} max {np.round(a[:, 5:].max(0), 1)}")
  print(f"  target_pos       min {np.round(tp.min(0), 3)} max {np.round(tp.max(0), 3)}")
  # action 与下一帧 state 的关系：action.pwm 是否等于 state.pwm[t+1]？action.joint_pos 与 state.joint_pos[t+1]？
  print(f"  |action.pwm[t] - state.pwm[t+1]| mean {np.abs(a[:-1, 5:] - s[1:, 5:13]).mean():.3f};"
        f"  |action.pwm[t] - state.pwm[t]| mean {np.abs(a[:, 5:] - s[:, 5:13]).mean():.3f}")
  print(f"  |action.q[t] - state.q[t+1]| mean {np.abs(a[:-1, :5] - s[1:, :5]).mean():.4f};"
        f"  |action.q[t] - state.q[t]| mean {np.abs(a[:, :5] - s[:, :5]).mean():.4f}")
  # target_pos（机体系平移）与 DVL 速度 × dt 的比值：≈1 说明是下一帧实际位移，>>1 说明是远处航点。
  dt = 0.1
  disp = s[:, 18:21] * dt
  nz = np.linalg.norm(tp[:, :3], axis=1) > 1e-6
  print(f"  frames with nonzero target_pos: {nz.sum()}/{len(df)}")
  if nz.any():
    r = np.linalg.norm(tp[nz, :3], axis=1) / np.maximum(np.linalg.norm(disp[nz], axis=1), 1e-6)
    print(f"  |target_pos.xyz| / |dvl_v*dt|  median {np.median(r):.2f}  p10 {np.percentile(r, 10):.2f}  p90 {np.percentile(r, 90):.2f}")
    print(f"  |target_pos.xyz| median {np.median(np.linalg.norm(tp[nz, :3], axis=1)):.3f} m; |dvl_v| median {np.median(np.linalg.norm(s[nz, 18:21], axis=1)):.3f} m/s")
  print("  first 3 rows: state", np.round(s[:3], 3).tolist())
  print("                action", np.round(a[:3], 3).tolist(), " target", np.round(tp[:3], 3).tolist())
