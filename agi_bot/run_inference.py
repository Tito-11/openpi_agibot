import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
try:
    from quat_math import (
        quat_xyzw_to_output,
        reorder_quat_xyzw,
        rot6d_interleaved_to_rot_matrix,
        rot_matrix_to_rot6d_interleaved,
    )
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference for the Agibot pi0 checkpoint.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="Training config name.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT, help="Checkpoint step directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for inference.")
    parser.add_argument("--head-image", required=True, help="Path to head_color image.")
    parser.add_argument("--hand-left-image", required=True, help="Path to hand_left image.")
    parser.add_argument(
        "--state",
        help="Comma separated 8D or 10D state.",
    )
    parser.add_argument("--state-json", help="JSON file containing a list of 8 or 10 floats.")
    parser.add_argument(
        "--state-quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used when the provided state is raw 8D.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize gripper values for input state and output actions.",
    )
    parser.add_argument(
        "--output-quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used in saved and printed robot actions.",
    )
    parser.add_argument(
        "--save-output",
        help="Optional JSON path to save both 10-dim and converted 8-dim actions.",
    )
    return parser.parse_args()


def _load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _load_state(args: argparse.Namespace, *, expected_state_dim: int) -> np.ndarray:
    if args.state_json:
        values = json.loads(Path(args.state_json).read_text())
    elif args.state:
        values = [float(v.strip()) for v in args.state.split(",") if v.strip()]
    else:
        raise ValueError("Provide either --state or --state-json.")

    state = np.asarray(values, dtype=np.float32)
    if expected_state_dim == 8 and state.shape == (8,):
        state = state.astype(np.float32).copy()
        state[3:7] = reorder_quat_xyzw(state[3:7], args.state_quat_order)
        state[3:7] = state[3:7] / np.clip(np.linalg.norm(state[3:7]), 1e-8, None)
        state[7] = 1.0 if state[7] > args.gripper_threshold else 0.0
        return state
    if expected_state_dim == 8 and state.shape == (10,):
        pos = state[:3]
        rot_matrix = rot6d_interleaved_to_rot_matrix(state[3:9])[0]
        quat_xyzw = R.from_matrix(rot_matrix).as_quat().astype(np.float32)
        gripper = np.array([1.0 if state[9] > args.gripper_threshold else 0.0], dtype=np.float32)
        return np.concatenate([pos.astype(np.float32), quat_xyzw, gripper], axis=0)
    if expected_state_dim == 10 and state.shape == (10,):
        state = state.astype(np.float32).copy()
        state[9] = 1.0 if state[9] > args.gripper_threshold else 0.0
        return state
    if expected_state_dim == 10 and state.shape == (8,):
        pos = state[:3]
        quat = reorder_quat_xyzw(state[3:7], args.state_quat_order)
        quat = quat / np.clip(np.linalg.norm(quat), 1e-8, None)
        rot_matrix = R.from_quat(quat).as_matrix()
        rot6d = rot_matrix_to_rot6d_interleaved(rot_matrix[None, ...])[0]
        gripper = np.array([1.0 if state[7] > args.gripper_threshold else 0.0], dtype=np.float32)
        return np.concatenate([pos.astype(np.float32), rot6d, gripper], axis=0)
    raise ValueError(f"Expected 8-dim or 10-dim state, got shape {state.shape}.")


def _convert_actions_to_robot(
    actions_chunk: np.ndarray, *, gripper_threshold: float, output_quat_order: str
) -> np.ndarray:
    processed = np.array(actions_chunk, copy=True)
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


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=args.prompt,
    )

    observation = {
        "images": {
            "head_color": _load_image(args.head_image),
            "hand_left_color": _load_image(args.hand_left_image),
        },
        "state": _load_state(args, expected_state_dim=int(train_config.model.action_dim)),
    }

    result = policy.infer(observation)
    policy_actions = np.asarray(result["actions"], dtype=np.float32)
    robot_actions_8d = _convert_actions_to_robot(
        policy_actions,
        gripper_threshold=args.gripper_threshold,
        output_quat_order=args.output_quat_order,
    )

    print(f"Policy actions [horizon,{policy_actions.shape[1]}]:")
    print(policy_actions)
    print(f"\n8D robot actions [x,y,z,quat({args.output_quat_order}),gripper]:")
    print(robot_actions_8d)
    print("\nFirst action to execute:")
    print(robot_actions_8d[0])

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": args.config_name,
            "checkpoint_dir": str(checkpoint_dir),
            "prompt": args.prompt,
            "output_quat_order": args.output_quat_order,
            "gripper_threshold": args.gripper_threshold,
            "policy_action_dim": int(policy_actions.shape[1]),
            "policy_actions": policy_actions.tolist(),
            "actions_8d": robot_actions_8d.tolist(),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved output to {output_path}")


if __name__ == "__main__":
    main()
