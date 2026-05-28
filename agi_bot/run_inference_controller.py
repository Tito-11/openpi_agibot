import argparse
import json
from pathlib import Path

from inference_runtime import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_PROMPT,
    create_policy,
    first_action_to_robot_xyzw,
    get_action_dim,
    infer_actions_chunk,
    load_image_rgb_uint8,
    load_state_from_args,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run controller-ready inference and emit only the first 8D robot action in xyzw order."
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG, help="Training config name.")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT, help="Checkpoint step directory.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt used for inference.")
    parser.add_argument("--head-image", required=True, help="Path to head_color image.")
    parser.add_argument("--hand-left-image", required=True, help="Path to hand_left image.")
    parser.add_argument("--state", help="Comma separated 8D or 10D state.")
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
        help="Threshold used to binarize gripper values for input state and output action.",
    )
    parser.add_argument(
        "--save-output",
        help="Optional JSON path to save the controller-ready first action and metadata.",
    )
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    action_dim = get_action_dim(args.config_name)
    policy = create_policy(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        prompt=args.prompt,
    )
    state = load_state_from_args(
        state_arg=args.state,
        state_json_path=args.state_json,
        expected_state_dim=action_dim,
        state_quat_order=args.state_quat_order,
        gripper_threshold=args.gripper_threshold,
    )
    policy_actions, policy_timing = infer_actions_chunk(
        policy,
        head_image=load_image_rgb_uint8(args.head_image),
        hand_left_image=load_image_rgb_uint8(args.hand_left_image),
        state=state,
    )
    first_policy_action = policy_actions[0]
    controller_action_8d = first_action_to_robot_xyzw(
        first_policy_action,
        gripper_threshold=args.gripper_threshold,
    )

    # Print a single JSON array for easy controller-side piping.
    print(json.dumps(controller_action_8d.tolist(), ensure_ascii=True))

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_name": args.config_name,
            "checkpoint_dir": str(Path(args.checkpoint_dir)),
            "prompt": args.prompt,
            "state_dim": int(state.shape[0]),
            "gripper_threshold": args.gripper_threshold,
            "policy_action_dim": int(policy_actions.shape[1]),
            "policy_action_first": first_policy_action.tolist(),
            "action_8d_xyzw_first": controller_action_8d.tolist(),
            "policy_timing": policy_timing,
        }
        output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
