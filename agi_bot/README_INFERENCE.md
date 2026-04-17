# Agibot 结合 OpenPI 推理实机部署指南

此文档旨在指导您在模型训练（`agibot_routeB_lora_tuning`）完成之后，如何使用 JAX/OpenPI 从检查点（Checkpoint）进行推理跑在实际的智元机器人上。

## 1. 简易推理测试脚本 (Dummy Test)
我们在 `agi_bot/run_inference.py` 内提供了一个脱离物理机器人的纯推理解构脚本。  
当您的模型（如经过了2000或30000次迭代生成了 Checkpoint）落盘后，通过本脚本即能够验证 JAX XLA Graph 的编译情况以及模型的 `Input -> Output` 维数：

```bash
# 您可以直接在刚才配置好依赖的 Python (.venv) 环境中运行：
uv run agi_bot/run_inference.py
```

### 脚本当中做了什么？
1. 加载了 `pi0_agibot` 配置，该配置通过 `ModelTransformFactory` 和 `AgibotInputs` 解析原图并做 Tensor Pad（32D）。
2. 从训练侧将指定的 Checkpoint 参数通过 Orbax/BasePyTreeHandler 唤入显存。
3. 随机生成伪造的 RGB Array (`['observation/image']` 和 `['observation/wrist_image']`) 还有 8 维状态输入给模型。
4. 调用 `.infer(example)` 执行 VLM（PaliGemma）的 Transformer 预测；并返回 `['actions']`，由于设置了 `AgibotOutputs`，它截取并退回原来的 **8维动作数据**。

## 2. 对接实际机器人的实网 Inference

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

