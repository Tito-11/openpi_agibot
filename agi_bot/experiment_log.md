# Agibot `pi0.5` 实验记录（精简版）

本文件只保留关键里程碑与“最终有效结论”。具体可复用流程以 [pipeline_guide.md](file:///home/admin1/ct/openpi_agibot/agi_bot/pipeline_guide.md) 为准。

## 1. 最终结论（当前有效）

- 端到端链路已打通：数据转换 → 训练 → 离线验证 → 真机在线推理闭环
- 任务：抓取四个铝型条中最左边的
- 表示：`8D = xyz + quat(xyzw) + gripper(0/1)`
- 推理默认 checkpoint：`20000`
- 真机抓取闭合使用：`action-index=9`
- 真机已观测到：在线推理成功抓住并回退（非数据重放）

## 2. 数据与配置（当前主线）

- 原始数据：`agi_bot/agi_data/g2_data`
- LeRobot 数据集：`agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat`
- Prompt：`grasp the leftmost aluminum profile among the four aluminum profiles`
- 训练配置：`pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4`
- 数据转换摘要：
  - `51` episodes
  - `frames_aligned_total = 10362`
  - `frames_kept_total = 7263`
  - `idle_removed_ratio = 0.2991`

## 3. 训练与 checkpoint

- 训练摘要（配置级别）：
  - `action_horizon=10`
  - `batch_size=32`
  - `num_train_steps=30000`
  - `save_interval=10000`
  - LoRA 微调（冻结 `PaliGemma`，训练 `Action Expert`）
- 本轮训练末尾未落盘 `30000`，推理优先使用 `20000`
- 推理 checkpoint 精简策略：仅保留 `params/` + `assets/`（删除 `train_state/`）

## 4. 离线推理验证（用训练数据）

- 抽样评估（training samples）结论：末端位姿误差在厘米级以内，夹爪准确率高（详见输出 JSON）
  - `agi_bot/infer_results/offline_eval_training_samples_20000.json`
- episode 回放评估结论（2 episodes）：位置 mean ~ `0.79cm`，姿态 mean ~ `2.80°`，夹爪 accuracy ~ `99.6%`
  - `agi_bot/infer_results/offline_eval_episode_replay_20000.json`

## 5. 真机联动（关键问题与最终修正）

- 关键控制路径以控制器脚本为准：
  - `agi:/data/pi05_test/g2_data_collector_v2.py`
- 真机控制必要条件（已验证）：
  - TF：`arm_l_end_link`（右臂固定并用 `kBothArms` 下发）
  - pose 需要短时间窗重复下发（类似 `servo_move()`）
- 夹爪闭合问题：
  - `action-index=1..7` 无闭合预测
  - `action-index=9` 才稳定触发闭合与抓取
- 启动稳定性：
  - 推理前显式张开夹爪更稳定
  - 一键启动必须“先复位再推理”（已修复 env 初始化导致的早退，以及远端日志路径变量冲突）
