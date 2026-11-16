# 第一轮训练说明

第一轮的目的是在**原版规模、完整训练集**上验证整套方法：策略能否在 Stonefish 里闭环跟住参考轨迹（航行器 + 机械臂 + 夹爪），
并且明显好于开环重放。同时摸清训练速度、显存和各指标的走势，作为后续长训练和调参的依据。

这一轮**不追求**最优精度，也不做学习率衰减、域随机化、DVL 降频 / 丢数、航向误差、真机部署（见第 6 节）。

安装、数据和测试见 `README.md` 第 1–4 节，方法细节见 `pipeline.md`。下文命令都在仓库根目录、`conda activate rovmimic` 之后运行。

---

## 1. 配置

全部是代码默认值，不需要额外参数。

| 项 | 值 |
|---|---|
| 参考轨迹 | 训练 `data/refs/usim_train_v1`（3469 条，38.5 h）；评估 `data/refs/usim_test_v1`（1037 条，11.5 h） |
| 仿真 | 4096 个环境，`min(64, 核数)` 个进程；回合最长 10 s（500 步） |
| 训练长度 | 5 万个真实步（所有环境各走一步算一个）；每真实步 0.4 次更新，共约 2 万次更新 |
| 世界模型更新 | 每次 10 个小批 × 2048 段 × 8 步；Adam，学习率 2e-4；从零开始，不用离线预训练 |
| 策略更新 | 每次 4096 个起点 × 想象 32 步，BPTT；actor-critic（价值网络 5 轮 × 4 个小批）；学习率 2e-4（恒定） |
| 采集 | 策略输出 + N(0, 0.1) 探索噪声；传感器噪声、位置测量误差、初始扰动都开（`system_overview.md` 1.7） |
| 存档 | 每 1000 步覆盖 `actor.pt` 等；每 5000 步另存到 `checkpoints/it_XXXXXXX/`（共 11 份，含第 0 步和最后的 49000 步） |
| 资源（开发机 RTX A6000 实测） | 显存峰值约 15 GB；约 1.6 s/次更新，全程约 9 h |

不用世界模型预训练的原因见 `pipeline.md` 3.1。

**换一种 GPU 时的时间估计**：大部分时间花在 GPU 更新上（A6000 上约 75%）。更快的卡会成比例缩短，例如 H100 的更新约快 6 倍，全程约 2–3 h。
CPU 核少时采集变慢：4096 环境、64 进程时每个真实步约 150 ms。

---

## 2. 步骤

### 2.1 前置检查（约 15 分钟）

1. `README.md` 第 4 节的测试全部通过，包括 GPU 冒烟测试（`test_runner_smoke.py cuda`）。
2. 选卡：`nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv`，空闲显存 ≥ 16 GB。
   最好选计算也空闲的卡，被别人占满算力时会慢好几倍。
3. 记下机器信息：GPU 型号、CPU 核数（`nproc`）。

### 2.2 基线评估（约 10 分钟）

```bash
python -I rovmimic/eval.py --baseline open_loop --clean --out data/eval/open_loop_clean   # 检查评估本身：应完成率 100%、误差 ≈ 0
python -I rovmimic/eval.py --baseline open_loop --out data/eval/open_loop                 # 基线：带初始扰动开环重放
```

开环重放就是把参考轨迹里记录的指令原样发出去。无扰动时它和参考完全重合。
加上与训练相同的初始扰动后，它没有反馈，会越漂越远：开发机上 8 条片段、20 s，完成率 50%，航行器位置误差中位数 22 cm。
策略要明显好于它，才说明学到了闭环跟踪。

### 2.3 试跑（约 15 分钟）

```bash
mkdir -p data/runs
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=<卡号> python -I rovmimic/train.py --runner.max-iterations 500 \
  --task-id ROVMimic-dryrun > data/runs/dryrun.log 2>&1
python -I tools/plot_train_log.py data/runs/dryrun.log
```

检查：
- 能正常跑完。前约 5 分钟是 `torch.compile` 和 CUDA graph 预热，日志不动是正常的。
- 每块日志（10 个真实步，4 次更新）的耗时，开发机参考值：`Collection time` 约 1.5 s、`WM Learning time` 约 2.1 s、`Policy Learning time` 约 2.7 s。
  明显更慢时检查 GPU 是否被别人占用、CPU 核数，并在汇报里写明。
- 显存峰值约 15 GB（训练时用 `nvidia-smi` 看）。

### 2.4 正式训练

```bash
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=<卡号> nohup python -I rovmimic/train.py > data/runs/train_first.log 2>&1 &
grep "Logging to" data/runs/train_first.log      # 存档目录：data/runs/<这里打印的相对路径>
```

下文用 `RUN` 指这个存档目录，例如 `RUN=data/runs/logs/ROVMimic-BlueROV2-Alpha5/2026-11-20_10-00-00`。

### 2.5 过程监控（每 1–2 小时一次）

```bash
python -I tools/plot_train_log.py data/runs/train_first.log     # -> data/runs/train_first.png、.csv
```

| 日志项 | 含义 | 预期走势 |
|---|---|---|
| `Loss/state` 及 `Loss/root_*` | 世界模型 8 步预测误差，已除以名义误差（1 表示与名义误差相当） | 前约 1000 步快速下降，之后保持低位、平稳 |
| `Policy/reward` | 策略更新时想象回报的平均值（含价值网络的末端估计，按步长换算），只看走势 | 上升 |
| `Mean reward`、`Mean episode length` | 真实仿真里每个回合的总奖励（满分 80）和长度（最长 500 步） | 上升；长度接近 500 说明很少失败 |
| `Termination/anchor_pos`、`anchor_ori`、`hull` | 每步因位置、姿态、侵入艇体而失败的环境比例 | 下降 |
| `Metrics/err_anchor_pos`、`err_tcp_pos` | 仿真里航行器、TCP 的平均误差（m） | 下降 |
| `Policy/grad_norm`、`Policy/value_function` | 策略梯度范数，价值网络损失 | 平稳，不爆炸 |

**出现以下情况就停下并汇报**：任何项出现 NaN / inf；`Loss/state` 在 2000 步之后持续上升；1 万步后 `Mean episode length` 完全不涨；
显存不足；耗时是 2.3 节参考值的 3 倍以上。

### 2.6 中间评估（每份快照约 5 分钟，可以和训练同时跑）

```bash
for it in 0005000 0010000 0015000 0020000; do
  python -I rovmimic/eval.py --policy $RUN/checkpoints/it_$it --num-workers 16 --out data/eval/first/it_$it
done
python -I tools/compare_evals.py data/eval/open_loop data/eval/first/it_*
```

评估默认用测试集前 256 条片段，每条最多 60 s，带训练时的噪声和扰动。
和训练同时跑时，用 `--num-workers 16` 少占 CPU。训练卡显存不够时加 `--device cpu`，会慢一些。

### 2.7 最终评估（训练结束后，约 30 分钟）

最后一份快照是 `it_0049000`。

```bash
F=$RUN/checkpoints/it_0049000
python -I rovmimic/eval.py --policy $F --out data/eval/first/it_0049000                         # 带噪声，256 条 × 60 s
python -I rovmimic/eval.py --policy $F --clean --out data/eval/first/final_clean                # 无噪声：精度
python -I rovmimic/eval.py --policy $F --num-clips 0 --max-seconds 0 --out data/eval/first/final_full   # 全部 1037 条，整段
python -I tools/compare_evals.py data/eval/open_loop_clean data/eval/open_loop data/eval/first/* --out data/eval/first/table.md
python -I tools/visualize_eval.py data/eval/first/it_0049000 --video --clips 0 1 2 3
python -I tools/visualize_eval.py data/eval/first/final_clean --video --clips 0 1
python -I tools/visualize_eval.py data/eval/open_loop --clips 0 1 2 3
python -I rovmimic/policy.py export $F --onnx
```

---

## 3. 验收标准（初步，可以调）

带噪声时，策略看到的水平位置带有缓慢漂移的测量误差（设计上最大 ±0.15 m），它会跟着测量值走。
所以带噪声的评估只看稳定性，精度看无噪声（`--clean`）的评估。

| 项 | 用哪个评估 | 标准 |
|---|---|---|
| 稳定性 | 带噪声，256 条 × 60 s | 完成率 ≥ 80%，且明显高于开环基线 |
| 回报 | 带噪声 | 归一化回报均值 ≥ 0.7（开环基线约 0.6） |
| 航行器精度 | 无噪声 | 位置误差中位数 ≤ 5 cm，姿态 ≤ 0.05 rad |
| 末端精度 | 无噪声 | TCP 误差中位数 ≤ 5 cm |
| 机械臂 | 无噪声 | 臂关节误差中位数 ≤ 0.03 rad，夹爪 ≤ 2 mm |
| 趋势 | 各快照 | 指标随训练基本变好，后期不崩 |

都达标：说明方法可行，下一步做更长的训练和第 6 节的项目。
部分达标：按第 4 节和失败原因分布定位问题。

---

## 4. 出问题怎么办

先汇报，不要自行大改设计。可以先试的参数：

| 现象 | 可能原因 | 先试 |
|---|---|---|
| 显存不足 | 卡被占，或卡太小 | 换卡；或策略批减半（显存约减半）：`runner.policy:actor-critic --runner.policy.batch-size 2048` |
| 采集很慢 | CPU 核少或被占 | `--env.num-workers <核数>`，或 `--env.num-envs 2048` |
| `Loss/*` 不降，或出现 NaN | 世界模型学不动 | 停下汇报，附 `train_first.png` |
| 想象回报涨、真实回报不涨 | 策略在利用世界模型的误差 | 汇报 `Loss/*` 和 `Termination/*` 曲线；可以试降学习率：`--runner.wm.lr 1e-4 runner.policy:actor-critic --runner.policy.lr 1e-4` |
| `hull` 失败占多数 | 机械臂撞艇体 | 汇报 `Reward/hull`、`Termination/hull` 曲线和几段视频 |
| 训练中断 | — | 不能续跑，只能重新开始 |

命令行注意：
- 改策略学习器的参数，必须写出子命令 `runner.policy:actor-critic`，放在其他参数之后。
- 不要用 `runner.policy:bptt`：那是不带价值网络的版本，论文里短想象步长时训不出来。

---

## 5. 时间预算（A6000）

| 步骤 | 用时 |
|---|---|
| 安装 + 数据（新机器） | 约 1 h，主要是下载和编译 |
| 测试 | 约 15 min |
| 基线评估 + 试跑 | 约 25 min |
| 正式训练 | 约 9 h |
| 中间 + 最终评估 | 约 1 h，中间评估可以和训练重叠 |

---

## 6. 这一轮不做的

- 学习率余弦衰减：论文里有，开源代码没有，前 2 万次更新影响很小。
- 域随机化（质量、浮力、推力）、DVL 5 Hz 与丢数、航向误差：见 `pipeline.md` 第 5 节。
- 真机部署：推理接口和导出已经有（`rovmimic/policy.py`），ROS 节点还没写。

---

## 7. 汇报内容

1. 机器：GPU 型号、CPU 核数；试跑实测的每块耗时；正式训练的总用时和显存峰值。
2. 训练曲线：`data/runs/train_first.png`，并简述走势和任何异常。
3. 评估表：`data/eval/first/table.md`，包括开环基线两行、各快照、最终无噪声和全测试集。
4. 对照第 3 节逐项写出是否达标。
5. 图和视频的路径：`summary.png`、2–4 个 `clip_XX.mp4`，并描述典型的失败情形。
6. 导出的 `policy.ts`、`policy.onnx` 的路径。
