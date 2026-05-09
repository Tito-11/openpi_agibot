# Agibot VLA 模型 (OpenPI) 端到端实战全流程指南

本文档记录了从原始机器人数据到模型微调，再到推理部署的完整闭环流程。适用于对 Physical Intelligence (π0 / π0.5) 模型进行微调和物理机器人落地的开发者。

## 阶段一：数据清洗与格式转换 (Data Cleaning & Conversion)

在这一阶段，我们将 Agibot 采集的原始 8 维数据（XYZ + 四元数 + 夹爪）进行深度清洗，并转换为模型性能上限更高的 10 维（XYZ + 6D旋转矩阵 + 夹爪）LeRobot 数据集。

### 1.1 核心数据清洗标准：Idle 帧过滤 (Idle Frame Filtering)

对于像 OpenPI (π0.5) 这样的 VLA（视觉-语言-动作）模型，数据的质量直接决定了模型泛化和执行的上限。在真实的机器人遥操作（Teleoperation）或数据采集中，通常会包含大量的 **Idle 帧（静止帧）**。如果把这些 Idle 帧喂给模型，会导致模型变得“迟钝”或“卡死”，并在推理时停滞不前。

利用我们在 `convert_agibot_data_to_lerobot.py` 中的逻辑，脚本会自动完成以下核心任务：
- **阈值判定**：如果相邻两帧之间的关节/末端位置变化小于设定的微小阈值（如 `1e-3`），则认为该帧是 Idle 帧。
- **Idle 帧过滤**：如果连续静止的帧数超过阈值（如 7 帧），则将这部分完全剔除。防止模型出现“学发呆”的卡死现象。这是因为 `action_horizon` 通常是 10，过滤长静止段能防止模型输出全是静止的 action chunk。
- **最小有效动作段筛选**：剔除极短片段（如不到 16 帧），防止手抖引起的噪音。
- **6D 旋转连续化**：利用 `scipy` 将带有数学缺陷的四元数转换为连续且欧几里得平坦的 6D 旋转矩阵。

**执行命令**：
```bash
uv run agi_bot/convert_agibot_data_to_lerobot.py --data_dir agi_bot/data/cartesian_grasp_routeB_5pt_clean_v2
```
*输出：将在 `agi_bot/test_converted_data/agibot_routeB` 目录下生成纯净的 LeRobot 格式数据。模型学习到的将是“看到目标 -> 果断出击”的高效映射。*

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
- **全空间 Delta 解耦**：通过 `_transforms.DeltaActions(mask=...)` 对前 9 维（位置+6D旋转）预测相对增量（Delta），极大增强平滑度与抗扰动泛化能力；第 10 维（夹爪）保留绝对值（Absolute）预测。
- **免除不必要的归一化**：在数据流生成时自动利用 `norm_stats` 修改策略，完全绕开了对 6D 旋转矩阵以及夹爪二值化维度的归一化处理（只对前3维的 XYZ 进行 Z-score）。

### 3.2 启动训练 (建议环境: Docker)
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_agibot --exp-name=agibot_routeB_lora_tuning --overwrite
```
*训练监控：强烈建议通过 W&B (Weights & Biases) 实时监控 Loss 下降曲线。*

---

## 阶段四：物理部署与推理 (Inference & Deployment)

训练完成后，使用提取出的 Checkpoint 进行真机闭环推理测试。OpenPI 框架的输出管道（Output Pipeline）会自动完成**反归一化 (Unnormalize)** 和 **反 Delta 加法 (Absolute Reconstruction)**。

### 4.1 推理端手动还原逻辑
在最终下发给硬件控制器前，推理脚本会自动拦截模型吐出的 10 维浮点数，并执行以下物理对齐：
- **Gram-Schmidt 正交化**：将模型吐出的 6D 浮点数无损、正交地还原回合法的 3x3 旋转矩阵，并转回控制器需要的 4 维四元数。
- **夹爪硬切分二值化**：将连续浮点数夹爪指令按照 `> 0.5` 阈值强制二值化为 `1.0`（闭合）和 `0.0`（张开）。

### 4.2 简易推理测试脚本 (Dummy Test)
我们在 `agi_bot/run_inference.py` 内提供了一个脱离物理机器人的纯推理解构脚本。当您的模型（如经过了2000或30000次迭代生成了 Checkpoint）落盘后，通过本脚本即能够验证 JAX XLA Graph 的编译情况以及模型的 `Input -> Output` 维数：

```bash
# 您可以直接在刚才配置好依赖的 Python (.venv) 环境中运行：
uv run agi_bot/run_inference.py
```
*验证：观察控制台输出的 8 维数组，确认机械臂是否能够平滑、准确地执行抓取动作。*

**脚本当中做了什么？**
1. 加载了 `pi0_agibot` 配置，该配置通过 `ModelTransformFactory` 和 `AgibotInputs` 解析原图并做 Tensor Pad（32D）。
2. 从训练侧将指定的 Checkpoint 参数通过 Orbax/BasePyTreeHandler 唤入显存。
3. 随机生成伪造的 RGB Array (`['observation/image']` 和 `['observation/wrist_image']`) 还有 8 维状态输入给模型。
4. 调用 `.infer(example)` 执行 VLM（PaliGemma）的 Transformer 预测；并返回 `['actions']`，由于设置了 `AgibotOutputs`，它截取并退回原来的 **8维动作数据**。

### 4.3 对接实际机器人的实网 Inference

要将模型部署在真实的 Agibot 上进行运动，您需要接入其实际相机的 RealSense 数据流以及 TCP/UDP 发送执行命令给关节控制器。

```python
import cv2
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

# 1. 载入您的最终权重
config = _config.get_config("pi0_agibot")
checkpoint_dir = "checkpoints/pi0_agibot/agibot_routeB_lora_tuning/step_30000"
policy = _policy_config.create_trained_policy(config, checkpoint_dir, default_prompt="grasp_bottle")

# 2. 机器人控制主循环
while robot_is_active:
    # A. 获取传感器实时数据
    img_head = get_agibot_camera_0()  # numpy array (240,320,3) RGB
    img_wrist = get_agibot_camera_1() # numpy array (240,320,3) RGB
    current_state = get_agibot_joints() # numpy array (8,)
    
    # B. 打包前传给 Pi0
    obs = {
        "observation/image": img_head,
        "observation/wrist_image": img_wrist,
        "observation/state": current_state
    }
    
    # C. 执行网络推理获取动作块
    # actions_chunk = np.ndarray (10, 8), 代表未来 10 个 timestep 的动作预测
    result = policy.infer(obs) 
    actions_chunk = result["actions"]
    
    # D. 动作平滑与控制下发
    # 根据 Action Chunking 策略，您可以选择执行首个 step，也可以通过 EMA 进行平滑
    execute_joint_positions(actions_chunk[0]) 

```

由于 OpenPI 支持以 **Websocket** 或 **REST** 启动为后备服务端，您也可以复用 `scripts/serve_policy.py`：

```bash
# 启动常驻开放服务，宿主机开启 8000 端口
uv run scripts/serve_policy.py Checkpoint --config=pi0_agibot --dir=checkpoints/pi0_agibot/agibot_routeB_lora_tuning --port 8000 --default-prompt "grasp_bottle"
```
随后您的机器人边缘计算板使用 WebSocket 连接该端口发送 JPEG 并接收动作矩阵即可！
