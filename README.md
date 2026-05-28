# openpi_agibot

本项目基于 Physical Intelligence 的 [openpi](README_openpi.md) 做了面向 Agibot G2（左臂单手）的端到端落地：从自采数据转换、`pi0.5` LoRA 微调、离线评估，到真机在线推理闭环与一键启动脚本。

如果你只是想快速复现实机推理，优先看：
- [agi_bot/pipeline_guide.md](agi_bot/pipeline_guide.md)

如果你想看实验过程与最终结论摘要，参考：
- [agi_bot/experiment_log.md](agi_bot/experiment_log.md)

## 你能在这里复现什么

- 数据：Agibot 自采数据 → LeRobot 数据集（两路图像 + 8D 状态/动作）
- 训练：`pi0.5 + 8D(quat xyzw) + LoRA` 微调配置与训练入口
- 离线验证：用训练数据对比 `预测动作` vs `GT`（位置/姿态/夹爪）
- 真机闭环：控制器实时采集图像与状态 → 本机推理 → 回传末端位姿与夹爪 → 控制器执行
- 一键联动：推理前先复位（回到采集第一帧附近），再启动连续推理；也提供一键停止

## 任务设定（当前主线）

- 任务：抓取四个铝型条中最左边的
- Prompt：`grasp the leftmost aluminum profile among the four aluminum profiles`
- 输入图像（两路）：
  - `head_color`
  - `hand_left`
- 状态/动作（8D，quat 必须为 xyzw）：
  - `x, y, z, qx, qy, qz, qw, gripper`
- 夹爪：监督与部署均使用二值 `{0, 1}`（阈值 `0.5`）

## 快速开始（本机离线）

本仓库使用 `uv` 管理 Python 依赖（见 [pyproject.toml](pyproject.toml)），Python 版本要求 `>=3.11`。

```bash
cd /path/to/openpi_agibot
uv sync
```

### 1) 数据转换（Agibot → LeRobot）

```bash
uv run python agi_bot/convert_agibot_data_to_lerobot.py \
  --data-dir agi_bot/agi_data/g2_data \
  --repo-name g2_leftmost_aluminum_profile_grasp_pi05_quat \
  --output-root agi_bot/lerobot_datasets \
  --task 'grasp the leftmost aluminum profile among the four aluminum profiles'
```

### 2) 离线评估（用训练数据验证 checkpoint）

```bash
uv run python agi_bot/evaluate_training_samples.py \
  --config-name pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4 \
  --checkpoint-dir checkpoints/pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4/agibot_g2_leftmost_aluminum_profile_grasp_pi05_quat_lora_a100x4/20000 \
  --dataset-root agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat \
  --prompt "grasp the leftmost aluminum profile among the four aluminum profiles" \
  --num-episodes 3 \
  --frames-per-episode 1 \
  --save-output agi_bot/infer_results/offline_eval_training_samples_20000.json
```

完整离线验证与指标说明见：
- [agi_bot/pipeline_guide.md](agi_bot/pipeline_guide.md)

## 真机在线推理闭环（Agibot G2 左臂）

真机闭环由两部分组成：
- 本机：推理服务（加载微调后的 `pi0.5` checkpoint）
- 控制器：采集图像/状态并执行动作（桥接客户端）

### 一键启动（推荐）

在本机执行（会自动：启动推理服务 → 同步脚本到控制器 → 先复位 → 再启动连续推理）：

```bash
cd /path/to/openpi_agibot
export ROBOT_PASSWORD='<your-robot-password>'
bash scripts/start_pi05_g2_full_inference.sh
```

一键停止：

```bash
cd /path/to/openpi_agibot
export ROBOT_PASSWORD='<your-robot-password>'
bash scripts/stop_pi05_g2_full_inference.sh
```

日志查看与参数覆盖方式见：
- [agi_bot/pipeline_guide.md](agi_bot/pipeline_guide.md)

## 代码结构（与复现相关的最小集合）

- 真机闭环
  - `agi_bot/vla_bridge_server.py`：本机推理服务
  - `agi_bot/vla_bridge_client_g2_left.py`：控制器侧桥接客户端（左臂单手）
  - `agi_bot/reset_left_home.py`：控制器侧复位脚本（内部复用控制器已验证的 `move_to_home()` 路径）
  - `agi_bot/remote_run_full_inference.sh`：控制器侧“复位 + 启动推理”的启动器
  - `scripts/start_pi05_g2_full_inference.sh` / `scripts/stop_pi05_g2_full_inference.sh`：本机一键启停入口
- 数据与离线验证
  - `agi_bot/convert_agibot_data_to_lerobot.py`：数据转换
  - `agi_bot/evaluate_training_samples.py` / `agi_bot/evaluate_episode_replay.py`：离线评估
  - `agi_bot/pipeline_guide.md`：当前有效主流程文档
  - `agi_bot/experiment_log.md`：精简实验记录

## 致谢

- 上游项目：Physical Intelligence 的 openpi（见 [README_openpi.md](README_openpi.md)）
- 数据格式/工具链：LeRobot
- 机器人控制：Agibot GDK（运行在控制器侧）

## License

本仓库许可证以根目录 `LICENSE` 为准；上游相关许可证参考 `LICENSE_GEMMA.txt` 与 openpi 原始仓库说明。
