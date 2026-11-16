# 给代码代理的工作约定

先读 `README.md`（安装、测试、训练、评估命令）和 `readme/first_run.md`（第一轮训练要做什么、看什么、怎么汇报）。
方法细节在 `readme/pipeline.md`，设计理由在 `readme/system_overview.md`。

## 环境

- Python 代码都在 conda 环境 `rovmimic` 里跑（Python 3.13，torch 2.14）。`stonefish` 环境只用来编译和转换网格，不要删：sfenv 运行时从它加载 C++ 库。
- 在仓库根目录运行；用 `python -I`；设 `OMP_NUM_THREADS=1`（仿真是多进程，每进程单线程）。
- 脚本里先 `import torch` 再 `import sfenv`。用 VecSim（多进程 spawn）的代码必须写成文件运行，不能 `python - <<EOF`。

## 必须遵守

- **Alpha5 网格（`alpha_description`、`data/alpha5/*.obj`）受 Reach Robotics 许可约束：永远不要提交到 git、不要上传到任何地方。**
  `.gitignore` 已排除；提交前用 `git status` 确认没有 `.obj`/`.stl`/`alpha5` 相关文件。
- `data/`（USIM 数据、参考轨迹、训练和评估输出）不进 git。
- 共享 GPU：启动前用 `nvidia-smi` 看空闲显存（训练需要 ≥ 16 GB）；不要结束别人的进程。
- 不要改 `supermimic/`、`mjlab_research/`、`third_party/` 里的上游代码；需要改行为时在 `rovmimic/` 里继承或替换（参考 `rovmimic/runner.py`）。
- 不要重新生成 `stonefish_bench/scenarios/*.scn`，也不要改 `scripts/alpha5_meshes.md5`；它们决定仿真动力学，必须和参考轨迹一致。
- 改了 `rovmimic/` 的代码，提交前至少跑 `README.md` 第 4 节里的 `test_terms`、`test_env`、`test_wm` 和 CPU 冒烟测试。

## 代码约定

- 两空格缩进，行宽 120。注释和文档用中文，标识符用英文。
- 状态、观测、动作的布局集中在 `rovmimic/state.py` 和 `rovmimic/servo.py`；奖励和失败判定在 `rovmimic/tracking.py`，仿真和世界模型共用。
- 路径一律相对仓库根目录，从 `rovmimic/paths.py` 取；不要写绝对路径。
- 改了设计（奖励、观测、动作、规模），同步更新 `readme/pipeline.md`。

## 汇报

训练和评估的汇报格式见 `readme/first_run.md` 第 7 节；结果文件放在 `data/runs/`、`data/eval/` 下，汇报里写清路径。
