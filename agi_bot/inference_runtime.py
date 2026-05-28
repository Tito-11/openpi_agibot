import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
try:
    from quat_math import quat_xyzw_to_output, reorder_quat_xyzw, rot6d_interleaved_to_rot_matrix, rot_matrix_to_rot6d_interleaved
except ImportError:
    from agi_bot.quat_math import (
        quat_xyzw_to_output,
        reorder_quat_xyzw,
        rot6d_interleaved_to_rot_matrix,
        rot_matrix_to_rot6d_interleaved,
    )


DEFAULT_CONFIG = "pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4"
DEFAULT_CHECKPOINT = (
    "checkpoints/pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4/"
    "agibot_g2_leftmost_aluminum_profile_grasp_pi05_quat_lora_a100x4/20000"
)
DEFAULT_PROMPT = "grasp the leftmost aluminum profile among the four aluminum profiles"


def load_image_rgb_uint8(image_path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def load_state_from_values(
    values: list[float] | np.ndarray,
    *,
    expected_state_dim: int = 8,
    state_quat_order: str = "xyzw",
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    state = np.asarray(values, dtype=np.float32)
    if expected_state_dim == 8:
        if state.shape == (8,):
            state = state.astype(np.float32).copy()
            state[3:7] = reorder_quat_xyzw(state[3:7], state_quat_order)
            state[3:7] = state[3:7] / np.clip(np.linalg.norm(state[3:7]), 1e-8, None)
            state[7] = 1.0 if state[7] > gripper_threshold else 0.0
            return state
        if state.shape == (10,):
            pos = state[:3]
            rot_matrix = rot6d_interleaved_to_rot_matrix(state[3:9])[0]
            quat_xyzw = R.from_matrix(rot_matrix).as_quat().astype(np.float32)
            gripper = np.array([1.0 if state[9] > gripper_threshold else 0.0], dtype=np.float32)
            return np.concatenate([pos.astype(np.float32), quat_xyzw, gripper], axis=0)
        raise ValueError(f"Expected 8-dim or 10-dim state for 8-dim model input, got shape {state.shape}.")

    if expected_state_dim == 10:
        if state.shape == (10,):
            state = state.astype(np.float32)
            state[9] = 1.0 if state[9] > gripper_threshold else 0.0
            return state
        if state.shape == (8,):
            pos = state[:3]
            quat = reorder_quat_xyzw(state[3:7], state_quat_order)
            quat = quat / np.clip(np.linalg.norm(quat), 1e-8, None)
            rot_matrix = R.from_quat(quat).as_matrix()
            rot6d = rot_matrix_to_rot6d_interleaved(rot_matrix[None, ...])[0]
            gripper = np.array([1.0 if state[7] > gripper_threshold else 0.0], dtype=np.float32)
            return np.concatenate([pos.astype(np.float32), rot6d, gripper], axis=0)
        raise ValueError(f"Expected 8-dim or 10-dim state for 10-dim model input, got shape {state.shape}.")

    raise ValueError(f"Unsupported expected state dim: {expected_state_dim}")


def load_state_from_args(
    *,
    state_arg: str | None,
    state_json_path: str | None,
    expected_state_dim: int = 8,
    state_quat_order: str = "xyzw",
    gripper_threshold: float = 0.5,
) -> np.ndarray:
    if state_json_path:
        values = json.loads(Path(state_json_path).read_text())
    elif state_arg:
        values = [float(v.strip()) for v in state_arg.split(",") if v.strip()]
    else:
        raise ValueError("Provide either state_arg or state_json_path.")
    return load_state_from_values(
        values,
        expected_state_dim=expected_state_dim,
        state_quat_order=state_quat_order,
        gripper_threshold=gripper_threshold,
    )


def get_train_config(config_name: str):
    return _config.get_config(config_name)


def get_action_dim(config_name: str) -> int:
    return int(get_train_config(config_name).model.action_dim)


def create_policy(
    *,
    config_name: str = DEFAULT_CONFIG,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT,
    prompt: str = DEFAULT_PROMPT,
):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    train_config = get_train_config(config_name)
    return _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=prompt,
    )


def infer_actions_chunk(
    policy,
    *,
    head_image: np.ndarray,
    hand_left_image: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, dict]:
    observation = {
        "images": {
            "head_color": head_image,
            "hand_left_color": hand_left_image,
        },
        "state": state,
    }
    result = policy.infer(observation)
    policy_actions = np.asarray(result["actions"], dtype=np.float32)
    return policy_actions, result.get("policy_timing", {})


def convert_actions_to_robot(
    actions_chunk: np.ndarray,
    *,
    gripper_threshold: float = 0.5,
    output_quat_order: str = "xyzw",
) -> np.ndarray:
    processed = np.asarray(actions_chunk, dtype=np.float32).copy()
    if processed.ndim != 2:
        raise ValueError(f"Expected actions chunk shape [N, D], got {processed.shape}")
    if processed.shape[1] == 8:
        quats_xyzw = processed[:, 3:7]
        quat_norms = np.clip(np.linalg.norm(quats_xyzw, axis=1, keepdims=True), 1e-8, None)
        quats_out = quat_xyzw_to_output(quats_xyzw / quat_norms, output_quat_order)
        gripper = np.where(processed[:, 7:8] > gripper_threshold, 1.0, 0.0)
        return np.concatenate([processed[:, 0:3], quats_out, gripper], axis=-1)
    if processed.shape[1] == 10:
        processed[:, 9] = np.where(processed[:, 9] > gripper_threshold, 1.0, 0.0)
        rot_matrices = rot6d_interleaved_to_rot_matrix(processed[:, 3:9])
        quats_xyzw = R.from_matrix(rot_matrices).as_quat()
        quats_out = quat_xyzw_to_output(quats_xyzw, output_quat_order)
        return np.concatenate([processed[:, 0:3], quats_out, processed[:, 9:10]], axis=-1)
    raise ValueError(f"Expected actions chunk shape [N, 8] or [N, 10], got {processed.shape}")


def first_action_to_robot_xyzw(first_action_10d: np.ndarray, *, gripper_threshold: float = 0.5) -> np.ndarray:
    action = np.asarray(first_action_10d, dtype=np.float32)
    if action.shape not in {(8,), (10,)}:
        raise ValueError(f"Expected first 8D or 10D action, got shape {action.shape}.")
    robot_action = convert_actions_to_robot(
        action[None, :],
        gripper_threshold=gripper_threshold,
        output_quat_order="xyzw",
    )[0]
    if not np.all(np.isfinite(robot_action)):
        raise ValueError("Controller action contains NaN or Inf.")
    quat_norm = np.linalg.norm(robot_action[3:7])
    if not np.isclose(quat_norm, 1.0, atol=1e-3):
        raise ValueError(f"Controller quaternion norm is invalid: {quat_norm}")
    return robot_action.astype(np.float32)
