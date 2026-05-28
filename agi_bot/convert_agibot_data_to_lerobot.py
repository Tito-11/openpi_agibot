"""Convert Agibot episodes into a quaternion-based LeRobot dataset for pi0.5 training."""

import json
import math
import os
import shutil
from pathlib import Path

import cv2
import lerobot.common.datasets.lerobot_dataset as lerobot_ds
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_REPO_NAME = "grasp_bottle_test_pi05_quat"
DEFAULT_TASK = "put the bottle into the box"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "lerobot_datasets"


def _configure_lerobot_home(output_root: str | Path | None) -> Path:
    if output_root is None:
        root = Path(os.environ.get("HF_LEROBOT_HOME", DEFAULT_OUTPUT_ROOT)).expanduser().resolve()
    else:
        root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(root)
    lerobot_ds.HF_LEROBOT_HOME = root
    return root


def _detect_quaternion_order(meta: dict) -> str:
    action_fields = meta.get("action_spec", {}).get("fields", [])
    quat_fields = [field for field in action_fields if field.startswith("q")]
    if quat_fields == ["qx", "qy", "qz", "qw"]:
        return "xyzw"
    if quat_fields == ["qw", "qx", "qy", "qz"]:
        return "wxyz"
    raise ValueError(f"Unsupported quaternion field order in meta.json: {quat_fields}")


def _get_episode_task(meta: dict, override_task: str | None) -> str:
    if override_task:
        return override_task
    return str(meta.get("task") or DEFAULT_TASK)


def _read_video_rgb(video_path: Path, *, expected_hw: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    target_h, target_w = expected_hw
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        if frame.shape[:2] != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {video_path}")
    return np.stack(frames, axis=0)


def _reorder_quat_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    if quat_order == "xyzw":
        return quat
    if quat_order == "wxyz":
        return np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1)
    raise ValueError(f"Unsupported quaternion order: {quat_order}")


def _normalize_and_fix_quaternions(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).copy()
    norms = np.linalg.norm(quat_xyzw, axis=1)
    valid = np.isfinite(quat_xyzw).all(axis=1) & (norms > 1e-8)
    quat_xyzw[valid] = quat_xyzw[valid] / norms[valid, None]

    valid_indices = np.flatnonzero(valid)
    for idx in valid_indices[1:]:
        prev_idx = valid_indices[np.searchsorted(valid_indices, idx) - 1]
        if np.dot(quat_xyzw[idx], quat_xyzw[prev_idx]) < 0:
            quat_xyzw[idx] *= -1.0
    return quat_xyzw, valid, norms


def _convert_pose_sequence(
    data_array: np.ndarray,
    *,
    quat_order: str,
    gripper_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_array = np.asarray(data_array, dtype=np.float32)
    if data_array.ndim != 2 or data_array.shape[1] != 8:
        raise ValueError(f"Expected shape (N, 8), got {data_array.shape}")

    pos = data_array[:, :3].astype(np.float32)
    quat_raw = _reorder_quat_xyzw(data_array[:, 3:7], quat_order)
    quat_xyzw, valid_quat, quat_norms = _normalize_and_fix_quaternions(quat_raw)
    gripper = (data_array[:, 7] > gripper_threshold).astype(np.float32)[:, None]
    converted = np.concatenate([pos, quat_xyzw, gripper], axis=1).astype(np.float32)
    valid_rows = valid_quat & np.isfinite(pos).all(axis=1)
    return converted, valid_rows, quat_norms


def _quat_angle_deltas_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    if len(quat_xyzw) <= 1:
        return np.zeros((0,), dtype=np.float32)
    dots = np.sum(quat_xyzw[1:] * quat_xyzw[:-1], axis=1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    angles = 2.0 * np.arccos(dots)
    return np.rad2deg(angles).astype(np.float32)


def _compute_idle_keep_mask(
    state_8d: np.ndarray,
    *,
    translation_threshold_m: float,
    rotation_threshold_deg: float,
    min_idle_len: int,
) -> np.ndarray:
    num_steps = len(state_8d)
    keep_mask = np.ones(num_steps, dtype=bool)
    if num_steps <= 2:
        return keep_mask

    pos_delta = np.linalg.norm(state_8d[1:, :3] - state_8d[:-1, :3], axis=1)
    quat_delta = _quat_angle_deltas_deg(state_8d[:, 3:7])
    gripper_delta = np.abs(state_8d[1:, 7] - state_8d[:-1, 7])

    is_idle = np.concatenate(
        [
            [False],
            (pos_delta < translation_threshold_m)
            & (quat_delta < rotation_threshold_deg)
            & (gripper_delta < 0.5),
        ]
    )

    run_start = None
    for idx, idle in enumerate(is_idle):
        if idle and run_start is None:
            run_start = idx
        if not idle and run_start is not None:
            run_len = idx - run_start
            if run_len > min_idle_len:
                keep_mask[run_start + 1 : idx - 1] = False
            run_start = None
    if run_start is not None:
        run_len = num_steps - run_start
        if run_len > min_idle_len:
            keep_mask[run_start + 1 : num_steps - 1] = False
    return keep_mask


def _safe_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def main(
    data_dir: str,
    *,
    repo_name: str = DEFAULT_REPO_NAME,
    output_root: str | None = None,
    task: str | None = None,
    quat_order: str = "auto",
    gripper_threshold: float = 0.5,
    idle_translation_threshold_m: float = 5e-4,
    idle_rotation_threshold_deg: float = 1.0,
    min_idle_len: int = 12,
    min_valid_frames: int = 10,
    image_height: int = 240,
    image_width: int = 320,
    push_to_hub: bool = False,
):
    lerobot_home = _configure_lerobot_home(output_root)
    output_path = lerobot_home / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="agibot",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (image_height, image_width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    data_dir_path = Path(data_dir)
    episodes = sorted(d for d in data_dir_path.iterdir() if d.is_dir() and d.name.startswith("episode_"))

    report = {
        "repo_name": repo_name,
        "data_dir": str(data_dir_path.resolve()),
        "output_path": str(output_path.resolve()),
        "image_size": {"height": image_height, "width": image_width},
        "episodes_total": len(episodes),
        "episodes_converted": 0,
        "episodes_skipped": [],
        "episodes": [],
    }
    all_state_xyz = []
    all_state_quat_norms = []
    all_state_quat_delta_deg = []
    gripper_values = []
    total_aligned_steps = 0
    total_kept_steps = 0
    total_invalid_rows = 0
    total_idle_removed = 0

    for ep_dir in episodes:
        print(f"Processing {ep_dir.name}...")
        try:
            meta = json.loads((ep_dir / "meta.json").read_text())
            actions_raw = np.load(ep_dir / "actions.npy")
            states_raw = np.load(ep_dir / "states.npy")
            episode_quat_order = _detect_quaternion_order(meta) if quat_order == "auto" else quat_order
            episode_task = _get_episode_task(meta, task)
            expected_hw = (image_height, image_width)
            head_frames = _read_video_rgb(ep_dir / "head_color.mp4", expected_hw=expected_hw)
            hand_frames = _read_video_rgb(ep_dir / "hand_left.mp4", expected_hw=expected_hw)
        except Exception as exc:
            report["episodes_skipped"].append({"episode": ep_dir.name, "reason": f"load_error: {exc}"})
            print(f"  Skipping {ep_dir.name}: {exc}")
            continue

        aligned_steps = min(len(actions_raw), len(states_raw), len(head_frames), len(hand_frames))
        actions_raw = actions_raw[:aligned_steps]
        states_raw = states_raw[:aligned_steps]
        head_frames = head_frames[:aligned_steps]
        hand_frames = hand_frames[:aligned_steps]

        try:
            actions_8d, action_valid, action_quat_norms = _convert_pose_sequence(
                actions_raw,
                quat_order=episode_quat_order,
                gripper_threshold=gripper_threshold,
            )
            states_8d, state_valid, state_quat_norms = _convert_pose_sequence(
                states_raw,
                quat_order=episode_quat_order,
                gripper_threshold=gripper_threshold,
            )
        except Exception as exc:
            report["episodes_skipped"].append({"episode": ep_dir.name, "reason": f"pose_convert_error: {exc}"})
            print(f"  Skipping {ep_dir.name}: {exc}")
            continue

        valid_mask = action_valid & state_valid
        idle_keep_mask = _compute_idle_keep_mask(
            states_8d,
            translation_threshold_m=idle_translation_threshold_m,
            rotation_threshold_deg=idle_rotation_threshold_deg,
            min_idle_len=min_idle_len,
        )
        keep_mask = valid_mask & idle_keep_mask

        kept_steps = int(np.sum(keep_mask))
        invalid_removed = int(np.sum(~valid_mask))
        idle_removed = int(np.sum(valid_mask & ~idle_keep_mask))

        print(
            f"  aligned={aligned_steps}, kept={kept_steps}, invalid_removed={invalid_removed}, "
            f"idle_removed={idle_removed}, quat_order={episode_quat_order}"
        )
        if kept_steps < min_valid_frames:
            report["episodes_skipped"].append(
                {
                    "episode": ep_dir.name,
                    "reason": f"too_few_valid_frames: kept={kept_steps}",
                }
            )
            continue

        for idx in np.flatnonzero(keep_mask):
            dataset.add_frame(
                {
                    "image": head_frames[idx],
                    "wrist_image": hand_frames[idx],
                    "state": states_8d[idx],
                    "actions": actions_8d[idx],
                    "task": episode_task,
                }
            )
        dataset.save_episode()

        kept_states = states_8d[keep_mask]
        all_state_xyz.append(kept_states[:, :3])
        all_state_quat_norms.append(np.linalg.norm(kept_states[:, 3:7], axis=1))
        all_state_quat_delta_deg.append(_quat_angle_deltas_deg(kept_states[:, 3:7]))
        gripper_values.append(kept_states[:, 7])
        total_aligned_steps += aligned_steps
        total_kept_steps += kept_steps
        total_invalid_rows += invalid_removed
        total_idle_removed += idle_removed
        report["episodes_converted"] += 1
        report["episodes"].append(
            {
                "episode": ep_dir.name,
                "task": episode_task,
                "quat_order": episode_quat_order,
                "aligned_steps": aligned_steps,
                "kept_steps": kept_steps,
                "invalid_removed": invalid_removed,
                "idle_removed": idle_removed,
                "action_quat_norm": _safe_stats(action_quat_norms[action_valid]),
                "state_quat_norm": _safe_stats(state_quat_norms[state_valid]),
            }
        )

    if hasattr(dataset, "consolidate"):
        dataset.consolidate()

    xyz_concat = np.concatenate(all_state_xyz, axis=0) if all_state_xyz else np.zeros((0, 3), dtype=np.float32)
    quat_norm_concat = (
        np.concatenate(all_state_quat_norms, axis=0) if all_state_quat_norms else np.zeros((0,), dtype=np.float32)
    )
    quat_delta_concat = (
        np.concatenate(all_state_quat_delta_deg, axis=0)
        if all_state_quat_delta_deg
        else np.zeros((0,), dtype=np.float32)
    )
    gripper_concat = np.concatenate(gripper_values, axis=0) if gripper_values else np.zeros((0,), dtype=np.float32)

    report["summary"] = {
        "frames_aligned_total": total_aligned_steps,
        "frames_kept_total": total_kept_steps,
        "frames_invalid_removed_total": total_invalid_rows,
        "frames_idle_removed_total": total_idle_removed,
        "idle_removed_ratio": float(total_idle_removed / total_aligned_steps) if total_aligned_steps else None,
        "state_xyz_min": xyz_concat.min(axis=0).tolist() if len(xyz_concat) else None,
        "state_xyz_max": xyz_concat.max(axis=0).tolist() if len(xyz_concat) else None,
        "state_quat_norm": _safe_stats(quat_norm_concat),
        "state_quat_delta_deg": _safe_stats(quat_delta_concat),
        "gripper_open_count": int(np.sum(gripper_concat < 0.5)),
        "gripper_close_count": int(np.sum(gripper_concat > 0.5)),
    }
    report_path = output_path / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"\nConverted dataset saved to: {output_path}")
    print(f"Conversion report saved to: {report_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["agibot", "lerobot", "quaternion", "pi05"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
