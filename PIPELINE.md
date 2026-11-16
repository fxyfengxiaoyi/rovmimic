# SuperMimic → Stonefish：各工具的命令

安装见 `README.md`。下面的命令都在仓库根目录、conda 环境 `rovmimic` 里运行，并设置 `OMP_NUM_THREADS=1`。
读取 USIM 数据的脚本一律用 `python -I` 运行。

| 目录 | 内容 |
|---|---|
| `stonefish_bench/` | 无界面场景（`scenarios/`，由 `gen_scenarios.py` 从 u0env 生成，已在仓库里）、C++ 吞吐 / 重置基准（`sf_bench`），结果见 `RESULTS.md` |
| `sfenv/` | Stonefish 的 Python 模块（pybind11），提供精确的状态读写、`rollout`、`step_batch`（分时复用）；一个进程只能有一个 `Sim` |
| `tools/` | USIM 下载和读取（`usim_lib.py`）、回放一致性检查、参考轨迹生成与检查、世界模型离线检查、评估可视化、训练日志解析 |
| `rovmimic/` | SuperMimic 适配：仿真环境、世界模型、奖励、训练循环、正运动学、推理、评估，各文件见 `readme/system_overview.md` 4.6 |
| `scripts/` | 新机器的安装脚本 |
| `data/usim_lowdim/` | USIM 低维部分（meta + parquet，不含视频） |
| `data/refs/` | 生成的参考轨迹（mjlab 打包格式） |
| `data/runs/`、`data/eval/` | 训练和评估输出 |

## 场景

| 场景 | 来源 | 用途 |
|---|---|---|
| `rov_nav.scn` | u0env `robots/bluerov2.scn` | USIM 导航类任务（goto / scan / inspect / follow）实际用的机器人，无机械臂 |
| `rov_alpha5.scn` | u0env `robots/bluerov2_alpha5_robotiq.scn` | USIM 抓取 / 搬运类任务；训练用。Alpha5 臂段网格按 Reach 授权从 `alpha_description` 取得，用 u0env 的 `tools/dataprocess/pkg_install.py` 转成 OBJ，与原模型一致 |
| `rov_alpha5_proxy.scn` | 同上 | 5 个臂段换成圆柱代理，只用于对比（质心和主轴与原模型不同） |

场景引用的网格在 `third_party/u0env/ros_ws/src/stonefish_bluerov2/data/`（`rovmimic/paths.py`，可用环境变量覆盖）。

Alpha5 网格受 Reach Robotics 许可证约束，只在本组内部使用，不要提交或分发：
- 由 `scripts/fetch_alpha5_meshes.sh` 生成：取 alpha 仓库的 `alpha_description`，放到 u0env 的 `ros_ws/src/description_alpha/origin/`，在 `stonefish` 环境里转换；
- 生成的 OBJ 在 u0env 的 `data/alpha5/` 下，脚本按 `scripts/alpha5_meshes.md5` 与开发机逐字节核对。
  网格转换做了简化，Stonefish 又按网格几何算臂段的质心和惯量，所以网格不同，动力学就不同。

u0env 和本仓库的 `.gitignore` 都排除了这些文件。

## 编译

由 `scripts/build_stonefish.sh` 完成（见 `README.md` 第 2 节）：在 `stonefish` 环境里编 Stonefish（装到 `third_party/stonefish_install`），
再按 `rovmimic` 环境的 Python 3.13 编 `sfenv`（输出 `sfenv/python/sfenv.cpython-313-*.so`）。
- `stonefish` 环境不能删：sfenv 运行时从它加载 SDL2、GL、freetype、libstdc++。
- make ≥ 4.4 默认用 fifo jobserver，而 pybind11 默认开 LTO，conda 编译器的 LTO 链接不认 fifo，脚本里改用 `--jobserver-style=pipe`。

自检：

```bash
python -I sfenv/tests/test_sfenv.py stonefish_bench/scenarios/rov_alpha5.scn third_party/u0env/ros_ws/src/stonefish_bluerov2/data/
```

## USIM 回放一致性检查

```bash
python -I tools/usim_replay_check.py --robot nav    --out results/replay_nav.json    data/usim_lowdim/test/data/chunk-000/*.parquet
python -I tools/usim_replay_check.py --robot alpha5 --out results/replay_alpha5.json data/usim_lowdim/test/data/chunk-000/*.parquet
```

## 参考轨迹

`scripts/prepare_data.sh` 会下载 USIM 并生成训练集和测试集。单独运行：

```bash
python -I tools/fetch_usim_lowdim.py
python -I tools/usim_make_refs.py --data data/usim_lowdim --split train --arm usim,random --workers 32 --out data/refs/usim_train_v1
python -I tools/check_refs.py data/refs/usim_train_v1
```

生成是确定性的：在 `rovmimic` 环境里重新生成，与开发机上的数据逐字节相同。训练集 32 进程约 5 分钟。

输出字段：
- `root_pos_w`、`root_quat_w`（wxyz）、`root_lin_vel_w`、`root_ang_vel_w`：base_link 原点系，NED 世界系。
- `joint_pos`、`joint_vel`：7 个关节，顺序见 `joint_names.json`。`joint_vel` 是上一步的平均速度 (q_k − q_{k−1})/dt，每条片段第一帧为 0（`rovmimic/servo.py`）。
- `motion_ends.npy`、`motion_names.json`。
- `actions.npy`：8 个推进器设定值 + 7 个舵机目标。
- `ee_pos_w`、`ee_quat_w`：TCP 位姿。
  - TCP 是 `ee_base_link`（MoveIt 规划末端）沿局部 +z（夹爪接近方向）偏移 7.5 cm 的点，在两指指垫之间。
  - 这是 U0 抓取规划器让物体中心所处的位置：由 USIM 抓取时刻反推，中位数 7.4 cm。
- `meta.json`：每条片段的来源，以及被跳过的 episode 和原因。

参考轨迹统一是 50 Hz，与控制频率一致。

## 正运动学

`rovmimic/fk.py` 直接读取场景文件里的关节链，从基座状态和关节状态算出各连杆、TCP 在世界系的位姿和速度。
世界模型只预测状态，奖励里的连杆 / 末端项都由它算出，梯度也经过它传回策略。

```bash
python -I rovmimic/tests/test_fk.py data/refs/usim_test_v1
```

## SuperMimic 训练、评估

测试：

```bash
python -I rovmimic/tests/test_terms.py        # 舵机公式、超限、艇体侵入
python -I rovmimic/tests/test_vec_sim.py 256 8
python -I rovmimic/tests/test_env.py          # 仿真环境执行参考动作，应逐帧复现参考轨迹
python -I rovmimic/tests/test_wm.py           # 世界模型公式、奖励、梯度、目标
TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python -I rovmimic/tests/test_runner_smoke.py
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python -I rovmimic/tests/test_runner_smoke.py cuda
```

世界模型离线检查 / 预训练（用参考轨迹，约 4 分钟，显存 < 0.5 GB；正式训练默认不用，见 `readme/pipeline.md` 3.1）：

```bash
CUDA_VISIBLE_DEVICES=0 python -I tools/fit_wm_refs.py --iters 600 --out data/wm_refs/run1
```

训练（先确认 GPU 空闲显存 ≥ 16 GB；默认就是原版规模，见 `readme/first_run.md`）：

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 nohup python -I rovmimic/train.py > data/runs/train_first.log 2>&1 &
python -I tools/plot_train_log.py data/runs/train_first.log
```

评估、可视化、导出：

```bash
python -I rovmimic/eval.py --policy <存档目录> --out data/eval/<名字>            # 加 --clean 关掉噪声和扰动
python -I rovmimic/eval.py --baseline open_loop --out data/eval/open_loop
python -I tools/compare_evals.py data/eval/*
python -I tools/visualize_eval.py data/eval/<名字> --video --clips 0 1
python -I rovmimic/policy.py export <存档目录> --onnx
```

仿真吞吐（2026-11-16，机器上有其他负载）：
- 8 进程约 1.9 万环境步/s，32 进程约 4.2 万。
- 4096 个环境时，加上奖励、观测和策略前向，每个真实步 64 进程 150 ms，32 进程 180 ms。
