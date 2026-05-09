# Agibot VLA 模型 (OpenPI) 端到端实战全流程指南

本文档记录了从原始机器人数据到模型微调，再到推理部署的完整闭环流程。适用于对 Physical Intelligence (π0 / π0.5) 模型进行微调和物理机器人落地的开发者。

## 阶段一：数据清洗与格式转换 (Data Cleaning & Conversion)

在这一阶段，我们将 Agibot 采集的原始 8 维数据（XYZ + 四元数 + 夹爪）进行深度清洗，并转换为模型性能上限更高的 10 维（XYZ + 6D旋转矩阵 + 夹爪）LeRobot 数据集。

### 1.1 数据清洗与 6D 旋转映射
脚本会自动完成两件核心任务：
- **Idle 帧过滤**：计算相邻两帧的物理位移，如果静止超过 7 帧，将自动将其剔除。防止模型出现“学发呆”的卡死现象。
- **6D 旋转连续化**：利用 `scipy` 将带有数学缺陷的四元数转换为连续且欧几里得平坦的 6D 旋转矩阵。

**执行命令**：
```bash
uv run agi_bot/convert_agibot_data_to_lerobot.py --data_dir agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2
```
*输出：将在 `agi_bot/test_converted_data/agibot_routeB` 目录下生成纯净的 LeRobot 格式数据。*

---

## 阶段二：计算归一化参数 (Norm Stats Calculation)

为了加速模型收敛并统一不同物理动作的量纲，需要对前 9 维连续数据进行 Z-score 均值方差归一化。
*注意：脚本会自动遍历整个数据集，夹爪维度的参数虽然会被计算，但由于我们在 `TrainConfig` 中的设定，夹爪在训练时将被豁免。*

**执行命令**：
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_agibot
```

---

## 阶段三：模型配置与训练 (Model Config & Training)

这一步决定了模型的学习策略和性能表现。我们在 `src/openpi/training/config.py` 中的 `pi0_agibot` 进行了以下关键设定：

### 3.1 核心训练策略
- **冻结 VLM 主干防遗忘**：`paligemma_variant="gemma_2b"`。彻底冻结 2B 参数的视觉语言模型，只为 300M 的 `Action Expert` 开启 LoRA 微调。这保留了模型极强的“世界常识”，防止在单一抓取任务中过拟合。
- **全空间 Delta 解耦**：`_transforms.StateActionToDelta(action_dim=9)`。前 9 维（位置+6D旋转）预测相对增量（Delta），极大增强平滑度与抗扰动泛化能力；第 10 维（夹爪）保留绝对值（Absolute）预测。

### 3.2 启动训练 (建议环境: Docker)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_agibot --exp-name=agibot_routeB_lora_tuning --overwrite
```
*训练监控：强烈建议通过 W&B (Weights & Biases) 实时监控 Loss 下降曲线。*

---

## 阶段四：物理部署与推理 (Inference & Deployment)

训练完成后，使用提取出的 Checkpoint 进行真机闭环推理测试。OpenPI 框架的输出管道（Output Pipeline）会自动完成**反归一化 (Unnormalize)** 和 **反 Delta 加法 (Absolute Reconstruction)**。

### 4.1 推理端手动还原逻辑 (`run_inference.py`)
在最终下发给硬件控制器前，推理脚本会自动拦截模型吐出的 10 维浮点数，并执行以下物理对齐：
- **Gram-Schmidt 正交化**：将模型吐出的 6D 浮点数无损、正交地还原回合法的 3x3 旋转矩阵，并转回控制器需要的 4 维四元数。
- **夹爪硬切分二值化**：将连续浮点数夹爪指令按照 `> 0.5` 阈值强制二值化为 `1.0`（闭合）和 `0.0`（张开）。

### 4.2 执行推理测试
```bash
uv run agi_bot/run_inference.py
```
*验证：观察控制台输出的 8 维数组，确认机械臂是否能够平滑、准确地执行抓取动作。*
