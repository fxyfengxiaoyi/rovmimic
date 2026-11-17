# rovmimic：水下机器人全身轨迹跟踪（SuperMimic → Stonefish + BlueROV2 + Alpha5）

把 [SuperMimic](https://github.com/HybridRobotics/supermimic)（学习的可微世界模型 + BPTT 训练全身跟踪策略）从 MuJoCo / Unitree G1
移植到水下：仿真器 [Stonefish](https://github.com/patrykcieslak/stonefish)，机器人是 BlueROV2 Heavy + Reach Alpha5 机械臂 + Robotiq Hand-E 夹爪，
参考轨迹由 USIM 数据集在 Stonefish 里重放生成。策略同时控制 8 个推进器、4 个臂关节和夹爪，跟踪航行器位姿 + 机械臂关节 + 末端（TCP）。

| 文档 | 内容 |
|---|---|
| `readme/pipeline.md` | **方法 pipeline**：每个部件的输入输出、训练和推理流程、批量规模和实测耗时 |
| `readme/first_run.md` | **第一轮训练说明**：配置、步骤、监控、评估、验收标准、出问题怎么办 |
| `readme/system_overview.md` | 机器人系统构成、参考轨迹、世界模型设计和理由 |
| `PIPELINE.md` | 各个工具脚本的命令（数据管线、测试、离线检查） |
| `stonefish_bench/RESULTS.md` | Stonefish 吞吐、重置等基准 |
| `AGENTS.md` | 给代码代理（Codex 等）的工作约定 |

## 目录

```
rovmimic/          适配代码：仿真环境、世界模型、奖励、训练循环、推理、评估（各文件见 readme/system_overview.md 4.6）
  train.py         训练入口
  eval.py          Stonefish 闭环评估
  policy.py        推理接口 + 导出 TorchScript / ONNX
  tests/           测试
sfenv/             Stonefish 的 Python 模块（C++ / pybind11）
stonefish_bench/   无界面场景（scenarios/*.scn，训练用 rov_alpha5.scn）和 C++ 基准
tools/             USIM 下载、参考轨迹生成与检查、世界模型离线检查、评估可视化、训练日志解析
scripts/           新机器的安装脚本
readme/            设计和流程文档
# 以下不在仓库里，由脚本生成：
supermimic/  mjlab_research/          上游（fetch_deps.sh）
third_party/stonefish  third_party/u0env  third_party/stonefish_install
data/usim_lowdim  data/refs  data/runs  data/eval
```

## 1. 环境要求

- Linux x86_64，conda（miniconda 即可），git，能访问 GitHub、HuggingFace、download.pytorch.org。
- NVIDIA GPU：训练时**空闲显存 ≥ 16 GB**（原版规模实测峰值 14.3 GiB）。开发机是 RTX A6000。
  默认装的 PyTorch 是 cu130，需要驱动 ≥ 580；驱动更旧时设 `TORCH_INDEX`（见第 2 步）。
- CPU：仿真是多进程的 CPU 计算，默认 `min(64, 核数)` 个进程，**建议 ≥ 32 核**。核少会更慢，但不影响结果。
- 磁盘：约 10 GB（依赖和编译约 3 GB，数据 2 GB，每次训练的存档约 1 GB）。

## 2. 安装（新机器，按顺序执行）

```bash
git clone <本仓库地址> rovmimic && cd rovmimic

bash scripts/fetch_deps.sh        # 按固定版本拉 supermimic、mjlab_research、stonefish、u0env（都是公开仓库）
bash scripts/create_env.sh        # conda 环境 rovmimic：Python 3.13 + torch 2.14.1 + 上游包 + scripts/requirements.txt
#   驱动不支持 CUDA 13 时：TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/create_env.sh
bash scripts/build_stonefish.sh   # conda 环境 stonefish（编译器 + C++ 库，运行时也要用，不能删）；编 Stonefish 和 sfenv
#   系统 libstdc++ 较旧时（如 Ubuntu 22.04）它会自动给 rovmimic 环境设 LD_PRELOAD，见脚本末尾；之后要重新 conda activate
bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license   # Alpha5 网格，见下面的许可说明
#   有打包好的数据（第 3 节）时，先解压数据包，再用包里的网格，不用联网：
#   bash scripts/fetch_alpha5_meshes.sh --i-have-reach-license --from data/alpha5
```

**Alpha5 网格许可**：Alpha5 的网格来自 Reach Robotics 的专有包 `alpha_description`，使用前必须已有 Reach 授权，
生成的文件只限组内使用，**任何时候都不能提交到 git 或对外分发**（`.gitignore` 已排除）。脚本会把生成的网格与开发机逐字节核对：
Stonefish 按网格几何算臂段的质心和惯量，网格不同动力学就不同。

装好以后，下面的命令都在仓库根目录、`conda activate rovmimic` 之后运行。

## 3. 数据

```bash
bash scripts/prepare_data.sh      # 下载 USIM 低维部分（约 220 MB），生成训练 / 测试参考轨迹（约 1.7 GB，几分钟）
```

参考轨迹生成是确定性的，开发机上与 `rovmimic` 环境里重新生成的结果逐字节相同。
结果：`data/refs/usim_train_v1`（3469 条片段，38.5 h）和 `data/refs/usim_test_v1`（1037 条，11.5 h）。

**拿到了打包好的数据时**（`rovmimic_data_v1.tar.zst`，由开发机的 `scripts/pack_data.sh` 生成）就不用跑上面的脚本，直接解压到仓库根目录：

```bash
sha256sum -c rovmimic_data_v1.tar.zst.sha256                  # 在压缩包所在目录，核对传输是否完整
tar -I zstd -xf <路径>/rovmimic_data_v1.tar.zst -C .          # 在仓库根目录；没有 zstd 时 apt install zstd 或 conda install -c conda-forge zstd
sha256sum -c --quiet data/MANIFEST.sha256 && echo OK          # 逐文件核对
```

包里有参考轨迹、USIM 低维原始数据、世界模型离线拟合结果和 Alpha5 网格（`data/alpha5`，Reach 授权文件，只限组内使用）。
解压后按第 2 步最后一条命令（`--from data/alpha5`）把网格装到 u0env 里，脚本会按 md5 核对。
`fetch_deps.sh` 要在这之前跑过，因为网格要装进 `third_party/u0env`。

## 4. 测试（装好后先全部跑通）

```bash
export OMP_NUM_THREADS=1
python -I sfenv/tests/test_sfenv.py stonefish_bench/scenarios/rov_alpha5.scn third_party/u0env/ros_ws/src/stonefish_bluerov2/data/
python -I rovmimic/tests/test_fk.py               # 正运动学与 Stonefish 一致（1e-14）
python -I rovmimic/tests/test_terms.py            # 舵机公式、超限、艇体侵入
python -I rovmimic/tests/test_vec_sim.py 256 8    # 多进程仿真与单进程逐位相同
python -I rovmimic/tests/test_env.py              # 仿真环境执行参考动作，逐帧复现参考轨迹（位置误差 < 1e-6 m）
python -I rovmimic/tests/test_wm.py               # 世界模型公式、奖励、梯度、目标
TORCHDYNAMO_DISABLE=1 WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="" python -I rovmimic/tests/test_runner_smoke.py   # 训练循环（CPU，小网络）
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python -I rovmimic/tests/test_runner_smoke.py cuda                     # 同上，GPU + torch.compile + CUDA graph
```

## 5. 训练

**第一轮训练的完整步骤、监控项和验收标准见 `readme/first_run.md`。** 最简命令：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv   # 选空闲显存 ≥ 16 GB 的卡
mkdir -p data/runs
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 nohup python -I rovmimic/train.py > data/runs/train_first.log 2>&1 &
```

默认配置就是原版 SuperMimic 的规模：4096 个仿真环境，策略批 4096 × 想象 32 步，世界模型 10 × 2048 段 × 8 步，
5 万个真实步（约 2 万次更新）。开发机 A6000 上约 9 小时、显存约 15 GB。所有参数都能用命令行改，`python -I rovmimic/train.py --help` 查看。

输出在 `data/runs/logs/ROVMimic-BlueROV2-Alpha5/<时间>/`：
- `actor.pt`、`world_model.pt`、`critic.pt`：最新存档，每 1000 个真实步覆盖一次；
- `checkpoints/it_XXXXXXX/`：每 5000 个真实步另存一份；
- `config.json`：完整配置，推理和评估按它重建网络。

W&B 默认离线；训练过程用 `python -I tools/plot_train_log.py data/runs/train_first.log` 画曲线。
训练不能从中断处续跑，中断了只能重新开始。

## 6. 评估和可视化

```bash
# 闭环评估：测试集前 256 条片段，每条最多 60 s，带与训练相同的传感器噪声和初始扰动
python -I rovmimic/eval.py --policy data/runs/logs/ROVMimic-BlueROV2-Alpha5/<时间>/checkpoints/it_0045000 --out data/eval/it45k
python -I rovmimic/eval.py --policy <同上> --clean --out data/eval/it45k_clean          # 无噪声、无扰动：看精度上限
python -I rovmimic/eval.py --baseline open_loop --out data/eval/open_loop               # 基线：开环重放参考指令
# 画图：汇总图、每条片段的曲线；--video 再渲染 3D 动画（mp4）
python -I tools/visualize_eval.py data/eval/it45k --video --clips 0 1 2 3
```

`metrics.json` 里的主要指标：完成率（整段没触发失败的比例）、归一化回报（每步奖励之和 / 满分）、
航行器位置 / 姿态、TCP、臂关节、夹爪误差（先对每条片段取平均，再给中位数和 90% 分位）。

## 7. 推理接口

```python
from rovmimic.policy import ROVPolicy, to_sim_command
pol = ROVPolicy.load("data/runs/logs/ROVMimic-BlueROV2-Alpha5/<时间>")   # 读 actor.pt + config.json
goal = pol.goal(motion, t, pos_meas, quat_meas)  # 230 维参考窗口：参考帧 t..t+45 相对测量位姿
action = pol(obs, goal)                          # obs 36 = [重力方向 3, 角速度 3, DVL 3, 关节角 7, 关节角速度 7, 上一步动作 13]
cmd = to_sim_command(action, q)                  # 8 个推进器设定值 + 7 个舵机目标
```

导出成不依赖本仓库的单文件：`python -I rovmimic/policy.py export <存档目录> --onnx`，生成 `policy.ts`（TorchScript）和 `policy.onnx`，
输入 `obs (B, 36)`、`goal (B, 230)`，输出 `action (B, 13)`。各量的定义见 `readme/pipeline.md` 第 1、4 节。

## 8. 注意事项

- 读取下载数据的脚本一律用 `python -I` 运行；脚本里先 `import torch` 再 `import sfenv`。
- 共享 GPU：先看空闲显存再启动，**不要结束别人的进程**。
- Stonefish 场景文件 `stonefish_bench/scenarios/*.scn` 已在仓库里，不需要重新生成。
  如果重跑 `gen_scenarios.py`，Alpha5 网格缺失时它会静默换成圆柱代理，动力学就变了。
- 路径都相对仓库根目录；u0env 或网格放在别处时，用环境变量 `ROVMIMIC_U0ENV` / `ROVMIMIC_STONEFISH_DATA` / `ROVMIMIC_USIM` 指过去（见 `rovmimic/paths.py`）。

## 上游和许可

- SuperMimic、mjlab_research：HybridRobotics，见各自仓库（`fetch_deps.sh` 固定的版本）。
- Stonefish：Patryk Cieślak，GPL-3.0。
- u0env：BlueROV2 / Robotiq 模型和 Alpha5 转换脚本，Apache-2.0；其中 Alpha5 相关文件受 Reach Robotics 许可约束（见上）。
- USIM 数据集：HuggingFace `Vincent2025hello/usim`。
