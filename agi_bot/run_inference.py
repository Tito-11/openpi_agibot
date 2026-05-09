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
    # 原始的分辨率为 (240, 320, 3)，并且现在状态为 10 维浮点数（含 6D 旋转）。
    example = {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/state": np.random.rand(10).astype(np.float32),
    }

    # C. Execute model inference to get action chunks
    print("🧠 Running policy inference. (First run will compile JAX computational graph, might take a while)...")
    result = policy.infer(example)
    
    # actions_chunk shape: (horizon, action_dim) -> e.g. (10, 10)
    actions_chunk = result["actions"]
    
    # D. Post-process the output
    # 1. Binarize the gripper output (10th dimension, index 9)
    # Since Flow Matching produces continuous values, we threshold the gripper at 0.5.
    actions_chunk[:, 9] = np.where(actions_chunk[:, 9] > 0.5, 1.0, 0.0)
    
    # 2. Convert 6D Rotation back to Quaternion
    # actions_chunk[:, 3:9] are the 6D rotation components
    from scipy.spatial.transform import Rotation as R
    
    def ortho6d_to_matrix(x6d):
        # x6d shape: (N, 6)
        x_raw = x6d[:, 0:3]
        y_raw = x6d[:, 3:6]
        
        # Normalize x
        x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
        # Gram-Schmidt to find orthogonal y
        z = np.cross(x, y_raw)
        z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        y = np.cross(z, x)
        
        # Stack into (N, 3, 3) matrix
        matrix = np.stack([x, y, z], axis=-1)
        return matrix
        
    rot_matrices = ortho6d_to_matrix(actions_chunk[:, 3:9])
    quats_scipy = R.from_matrix(rot_matrices).as_quat() # returns [x, y, z, w]
    
    # Convert scipy's [x, y, z, w] back to Agibot's expected [w, x, y, z]
    quats_agibot = np.stack([quats_scipy[:, 3], quats_scipy[:, 0], quats_scipy[:, 1], quats_scipy[:, 2]], axis=1)
    
    # Reconstruct the final 8-dim action for the robot
    # [x, y, z, qw, qx, qy, qz, gripper]
    final_actions_for_robot = np.concatenate([
        actions_chunk[:, 0:3],
        quats_agibot,
        actions_chunk[:, 9:10]
    ], axis=-1)
    
    print("✅ Inference complete! Extracted Action Chunk (Converted back to 8D with Quaternion):")
    print(final_actions_for_robot)
    
    print("✨ First step action to execute on robot:", final_actions_for_robot[0])

if __name__ == "__main__":
    main()
