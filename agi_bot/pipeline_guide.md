# Agibot `pi0.5` 端到端流程（数据 → 训练 → 离线验证 → 真机推理）

本仓库当前只保留已验证可复用主线：`pi0.5 + 8D(quat xyzw) + LoRA`，用于 Agibot（左臂单手）抓取任务。

参考入口：

- [README_openpi.md](file:///home/admin1/ct/openpi_agibot/README_openpi.md)
- [config.py](file:///home/admin1/ct/openpi_agibot/src/openpi/training/config.py)

## 1. 关键约定（必须一致）

### 1.1 任务与输入输出

- 任务：抓取四个铝型条中最左边的
- Prompt：`grasp the leftmost aluminum profile among the four aluminum profiles`
- 图像输入（两路）：
  - `head_color`
  - `hand_left`
- 状态/动作（8D，quat 必须 xyzw）：
  - `x, y, z, qx, qy, qz, qw, gripper`
- 夹爪监督与部署接口统一为二值：
  - `gripper ∈ {0, 1}`（阈值 `0.5`）

### 1.2 目录

- 原始数据：`agi_bot/agi_data/g2_data`
- LeRobot 数据集：`agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat`
- 训练配置名：`pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4`
- 推理默认 checkpoint：`20000`

## 2. 机器人连接与控制器侧约定（长期复用）

### 2.1 连接信息

- 控制器：`ssh <user>@<robot-host>`

### 2.2 控制器侧目录约定

- 控制器应用目录：`~/app`
- 控制器控制相关目录：`~/app/gdk`
- 桥接脚本放置目录：`/data/ct`

### 2.3 控制器侧环境

在控制器上运行桥接/复位相关脚本时使用：

```bash
source /home/agi/app/env.sh /home/agi/app
```

## 3. 数据转换（g2_data → LeRobot）

### 3.1 转换命令

```bash
uv run python agi_bot/convert_agibot_data_to_lerobot.py \
  --data-dir /home/admin1/ct/openpi_agibot/agi_bot/agi_data/g2_data \
  --repo-name g2_leftmost_aluminum_profile_grasp_pi05_quat \
  --output-root /home/admin1/ct/openpi_agibot/agi_bot/lerobot_datasets \
  --task 'grasp the leftmost aluminum profile among the four aluminum profiles'
```

### 3.2 最小验收

- 数据集目录存在：`agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat`
- 关键文件存在：
  - `conversion_report.json`
  - `meta/info.json`
  - `meta/tasks.jsonl`

## 4. 训练配置与归一化

### 4.1 当前主配置

- 配置名：`pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4`
- 位置：[config.py](file:///home/admin1/ct/openpi_agibot/src/openpi/training/config.py)
- 关键设定（摘要）：
  - `action_dim=8`
  - `action_horizon=10`
  - `batch_size=32`
  - `num_train_steps=30000`
  - `save_interval=10000`
  - LoRA 微调（冻结 `PaliGemma`，训练 `Action Expert`）

### 4.2 训练前必须做：compute_norm_stats

- `xyz` 参与统计归一化
- quat 不做分布拉伸，只做单位化与连续性修正
- `gripper` 保持二值语义

## 5. 离线推理验证（Offline Eval）

离线验证目的：用训练数据对比 `模型预测动作` vs `数据动作（GT）`，快速确认末端位姿与夹爪是否对得上。

### 5.1 推理 checkpoint 需要哪些文件

- 需要：`params/` + `assets/`
- 不需要：`train_state/`

### 5.2 训练样本抽样评估

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

### 5.3 Episode 回放评估

```bash
uv run python agi_bot/evaluate_episode_replay.py \
  --config-name pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4 \
  --checkpoint-dir checkpoints/pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4/agibot_g2_leftmost_aluminum_profile_grasp_pi05_quat_lora_a100x4/20000 \
  --dataset-root agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat \
  --prompt "grasp the leftmost aluminum profile among the four aluminum profiles" \
  --num-episodes 2 \
  --save-output agi_bot/infer_results/offline_eval_episode_replay_20000.json
```

### 5.4 当前 20000 checkpoint 的离线结论（摘要）

- 末端位置：episode mean 约 `0.79cm`（p95 约 `1.66cm`）
- 末端姿态：episode mean 约 `2.80°`
- 夹爪：episode accuracy ~ `99.6%`

## 6. 真机推理（已验证可用）

### 6.1 已验证参考脚本（控制器侧）

- `agi:/data/pi05_test/g2_data_collector_v2.py`

关键点（真机已验证）：

- 读取末端位姿：`TF.get_tf_from_base_link("arm_l_end_link")`
- 下发控制：`EndEffectorPose.group = kBothArms`（右臂保持当前位姿不动）
- 控制需要短时间窗重复下发 pose（类似 `servo_move()`），单次下发不足以稳定运动
- 推理前先显式张开夹爪更稳定
- 本任务闭合抓取用 `action-index=9`

### 6.2 一键启动（先复位，再推理）

在本机 `/home/admin1/ct/openpi_agibot` 下执行：

```bash
bash scripts/start_pi05_g2_full_inference.sh
```

运行前需要先导出控制器密码：

```bash
export ROBOT_PASSWORD='<your-robot-password>'
```

该命令会自动：

1. 启动/重启本机推理服务 `agi_bot/vla_bridge_server.py`
2. 同步桥接脚本到控制器 `/data/ct/`
3. 控制器侧先执行左臂复位（`agi_bot/reset_left_home.py`，内部调用 collector 的 `move_to_home()`）
4. 控制器侧后台启动连续推理客户端（`agi_bot/vla_bridge_client_g2_left.py`）

### 6.3 一键停止

```bash
bash scripts/stop_pi05_g2_full_inference.sh
```

### 6.4 查看日志

本机推理服务日志：

```bash
tail -f /home/admin1/ct/openpi_agibot/.runtime_logs/vla_bridge_server.log
```

控制器侧连续推理日志：

```bash
ssh <user>@<robot-host> 'tail -f /tmp/vla_full_run.log'
```

控制器侧最新一帧输出：

```bash
ssh <user>@<robot-host> 'cat /data/ct/vla_bridge_last.json'
```

## 7. 新数据复用模板（最短路径）

1. 原始数据放入：`agi_bot/agi_data/<new_dataset_name>`
2. 转换：`convert_agibot_data_to_lerobot.py`
3. 输出到：`agi_bot/lerobot_datasets/<repo_name>`
4. 在 [config.py](file:///home/admin1/ct/openpi_agibot/src/openpi/training/config.py) 复制当前配置并修改：
   - `repo_id`
   - `default_prompt`
   - `TrainConfig.name`
   - `--exp-name`
5. 训练前执行：`compute_norm_stats`
6. 启训：`scripts/train.py`
