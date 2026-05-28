import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from quat_math import reorder_quat_xyzw, rot6d_interleaved_to_rot_matrix, rot_matrix_to_rot6d_interleaved
except ImportError:
    from agi_bot.quat_math import reorder_quat_xyzw, rot6d_interleaved_to_rot_matrix, rot_matrix_to_rot6d_interleaved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Agibot state for current inference configs.")
    parser.add_argument(
        "--state",
        help="Comma separated state. Supports 8D quat state or 10D rot6d state.",
    )
    parser.add_argument(
        "--state-json",
        help="JSON file containing a list of 8 or 10 floats.",
    )
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used by raw 8D state.",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize the gripper value.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the prepared state.",
    )
    parser.add_argument(
        "--target-dim",
        type=int,
        choices=[8, 10],
        default=8,
        help="Target state dimension. Use 8 for current g2 pi0.5 inference.",
    )
    return parser.parse_args()


def _load_values(args: argparse.Namespace) -> np.ndarray:
    if args.state_json:
        values = json.loads(Path(args.state_json).read_text())
    elif args.state:
        values = [float(v.strip()) for v in args.state.split(",") if v.strip()]
    else:
        raise ValueError("Provide either --state or --state-json.")
    return np.asarray(values, dtype=np.float32)


def _raw8_to_model10(raw_state: np.ndarray, quat_order: str, gripper_threshold: float) -> np.ndarray:
    pos = raw_state[:3]
    quat = raw_state[3:7]
    gripper = np.array([1.0 if raw_state[7] > gripper_threshold else 0.0], dtype=np.float32)
    quat_xyzw = reorder_quat_xyzw(quat, quat_order)
    quat_xyzw = quat_xyzw / np.clip(np.linalg.norm(quat_xyzw), 1e-8, None)
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    rot6d = rot_matrix_to_rot6d_interleaved(rot_matrix[None, ...])[0]
    return np.concatenate([pos.astype(np.float32), rot6d, gripper], axis=0)


def _model10_to_raw8(state_10d: np.ndarray, gripper_threshold: float) -> np.ndarray:
    pos = state_10d[:3]
    rot_matrix = rot6d_interleaved_to_rot_matrix(state_10d[3:9])[0]
    quat_xyzw = R.from_matrix(rot_matrix).as_quat().astype(np.float32)
    gripper = np.array([1.0 if state_10d[9] > gripper_threshold else 0.0], dtype=np.float32)
    return np.concatenate([pos.astype(np.float32), quat_xyzw, gripper], axis=0)


def main() -> None:
    args = _parse_args()
    values = _load_values(args)

    if args.target_dim == 8:
        if values.shape == (8,):
            prepared_state = values.astype(np.float32).copy()
            prepared_state[3:7] = reorder_quat_xyzw(prepared_state[3:7], args.quat_order)
            prepared_state[3:7] = prepared_state[3:7] / np.clip(np.linalg.norm(prepared_state[3:7]), 1e-8, None)
            prepared_state[7] = 1.0 if prepared_state[7] > args.gripper_threshold else 0.0
        elif values.shape == (10,):
            prepared_state = _model10_to_raw8(values, args.gripper_threshold)
        else:
            raise ValueError(f"Expected 8D or 10D input, got shape {values.shape}.")
    else:
        if values.shape == (8,):
            prepared_state = _raw8_to_model10(values, args.quat_order, args.gripper_threshold)
        elif values.shape == (10,):
            prepared_state = values.astype(np.float32)
            prepared_state[9] = 1.0 if prepared_state[9] > args.gripper_threshold else 0.0
        else:
            raise ValueError(f"Expected 8D or 10D input, got shape {values.shape}.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared_state.tolist(), indent=2))

    print(f"Prepared {args.target_dim}D state:")
    print(prepared_state)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
