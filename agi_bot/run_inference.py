import os
import logging
import numpy as np
import jax

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 [Agibot] Initializing inference test script...")
    # 1. 载入先前配置好的 pi0_agibot 设置
    config = _config.get_config("pi0_agibot")
    
    # 2. 定位到当前正在训练的 Checkpoint 目录
    checkpoint_dir = "checkpoints/pi0_agibot/agibot_routeB_lora_tuning"
    
    if not os.path.exists(checkpoint_dir):
         print(f"⚠️ Checkpoint directory not found at {checkpoint_dir}. Please wait for the first 2000 steps to save!")
         # 就算没有当前微调权重，我们也可以尝试加载 base 权重作为演示
         checkpoint_dir = os.path.expanduser("~/.cache/openpi/openpi-assets/checkpoints/pi0_base")
         print(f"🔄 Falling back to base checkpoint {checkpoint_dir} for demonstration.")
    
    print(f"📦 Loading policy from {checkpoint_dir} ...")
    policy = _policy_config.create_trained_policy(
        config, 
        checkpoint_dir, 
        default_prompt="grasp_bottle"
    )
    
    print("🖼️ Constructing dummy observation for Agibot (camera_0 and camera_1)...")
    # Agibot 的原始输入需要符合 agibot_policy.py 里 AgibotInputs 定义的 keys。
    # 原始的分辨率为 (240, 320, 3)，并且状态为 8 维浮点数。
    example = {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
    }

    print("🧠 Running policy inference. (First run will compile JAX computational graph, might take a while)...")
    result = policy.infer(example)
    
    print("✅ Inference successful!")
    print("➡️ Generated actions block shape:", result["actions"].shape)
    print("➡️ Sample output (first timestep action):", result["actions"][0])

if __name__ == "__main__":
    main()
