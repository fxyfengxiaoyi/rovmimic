# Stonefish 无界面基准结果（2026-11-13）

机器：2× Xeon Gold 6430（64 物理核 / 128 线程），测试时他人负载 ~20。每进程 `OMP_NUM_THREADS=1`，控制 50 Hz，随机平滑动作，每 500 控制步重置一次。

## 单进程吞吐（控制步/秒；括号内为实时倍数）

| 场景 | 物理 100 Hz | 200 Hz | 500 Hz | 1000 Hz | 重置耗时 |
|---|---|---|---|---|---|
| rov_only（8 推进器） | 3788 (76×) | 3807 | 3544 (71×) | 3338 | 1–2 µs |
| rov_arm（仓库自带 robot_arm） | 2963 (59×) | 2552 | 1804 (36×) | 1216 | 3–4 µs |
| rov_alpha5（U0 实际用的 Alpha5+Robotiq，臂段圆柱代理*） | 2531 (51×) | 2049 (41×) | 1187 (24×) | – | 3–4 µs |

\* 上表 rov_alpha5 的数字是用圆柱代理臂段测的。2026-11-14 拿到 Reach 授权后，已换成原始网格（u0env `pkg_install.py` 生成的 OBJ），代理版保留为 `rov_alpha5_proxy.scn`。
  - 原始网格：物理 100 Hz 下 2685 步/s，重置误差 1e-17，阶跃响应与代理版一致。
  - 静止俯仰配平：原始网格在机械臂 q=0 时为 -2.0°，在 USIM 工作姿态（b=c=d=0.5）时为 -4.3°；代理版分别为 -1.2° 和 -2.7°。两者都是中性浮力。

初始化 ~1.0 s；RestartScenario+StartSimulation 重建 ~0.9–1.0 s。仿真时间与期望值逐位一致（每次 StepSimulation 恰好一个子步）。

## 多进程扩展（rov_alpha5，物理 100 Hz，每进程 300 s 仿真时间）

| 进程数 | 总吞吐（控制步/秒） | 单进程均值 |
|---|---|---|
| 1 | 2 687 | 2 687 |
| 8 | 21 325 | 2 666 |
| 32 | 58 921 | 1 841 |
| 64 | 56 726 | 886 |

32 进程后饱和（他人负载 + 超线程 + 全核睿频下降）。

## 状态重置（btMultiBody 层直接读写基座位姿/速度 + 关节 q/qd）

- rov_only、rov_alpha5：在 100/200/500 Hz 下，重置到轨迹中任意 s_k 后用同样动作重放 100 步，与参考轨迹误差 ≤1e-15（姿态一项 ~1e-8 是 acos 的数值下限）；同一状态重置两次结果逐位一致；任意构造状态写入后读回误差为 0。
- rov_arm：100 Hz 精确；500 Hz 下首步误差 1e-16，之后按 ~16/s 指数放大到 1e-2（模型本身刚性+混沌，关节速度达 9–27 rad/s），不是重置接口的问题。

## 动力学观察

- 流体力按 `round(sps/50)` 分频，固定 50 Hz 更新。物理 100 Hz + 控制 50 Hz 时，每个控制步恰好一次流体力更新。
- 推进器设定值 0.5 时每个约 5 N。纵荡阶跃（4 推进器共 18.8 N）：0.5 s 0.041 → 1 s 0.077 → 2 s 0.134 → 5 s 0.233 → 10 s 0.285 m/s；垂荡（17.6 N）10 s 0.342 m/s。时间常数 3–4 s。
- 零推力 5 s：rov_alpha5 中性浮力（漂移 0.000 m）；rov_only 略正浮力（10 s 上浮 0.12 m）；rov_arm 下沉 3.4 m。
- 洋流：`Ocean` 构造时 `currentsEnabled=false`，库里没有任何地方调用 `EnableCurrents()`，必须手动开启。开启后 uniform 洋流可在运行时修改：0.3 / 0.6 m/s 洋流下，10 s 后航行器速度分别为 0.28 / 0.57 m/s。

## 复现

在仓库根目录、`stonefish` 环境里（安装见 `README.md`）。场景文件已在仓库里，不要重跑 `gen_scenarios.py`：
网格缺失时它会静默换成圆柱代理。

```bash
conda activate stonefish
cmake -S stonefish_bench -B stonefish_bench/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$PWD/third_party/stonefish_install;$CONDA_PREFIX" && make -C stonefish_bench/build
cd stonefish_bench
DATA=../third_party/u0env/ros_ws/src/stonefish_bluerov2/data/
export OMP_NUM_THREADS=1 SF_SERVO_LIMITS=$(python3 -I servo_limits.py scenarios/rov_alpha5.scn)
./build/sf_bench throughput scenarios/rov_alpha5.scn $DATA 100 50 60 500
./build/sf_bench reset      scenarios/rov_alpha5.scn $DATA 100 50
./build/sf_bench current    scenarios/rov_alpha5.scn $DATA 100 50
```
