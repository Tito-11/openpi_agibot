import argparse
import io
import json
from pathlib import Path

import numpy as np

from quat_math import quat_xyzw_to_output, rot6d_interleaved_to_rot_matrix


DEFAULT_CONFIG = "pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4"
DEFAULT_CHECKPOINT = (
    "checkpoints/pi05_agibot_g2_leftmost_aluminum_profile_grasp_a100x4/"
    "agibot_g2_leftmost_aluminum_profile_grasp_pi05_quat_lora_a100x4/20000"
)
DEFAULT_DATASET_ROOT = "agi_bot/lerobot_datasets/g2_leftmost_aluminum_profile_grasp_pi05_quat"
DEFAULT_PROMPT = "grasp the leftmost aluminum profile among the four aluminum profiles"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline inference evaluation on training samples.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-episodes", type=int, default=3, help="How many episodes to sample.")
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=1,
        help="How many frame positions to sample from each selected episode.",
    )
    parser.add_argument(
        "--output-quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used in saved robot-format actions.",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--save-output", help="Optional JSON path to save detailed evaluation results.")
    parser.add_argument("--plot-dir", help="Optional directory to save plots (PNG).")
    parser.add_argument(
        "--plot-json",
        help="Optional path to an existing evaluation JSON. When provided, skips inference and only generates plots.",
    )
    return parser.parse_args()


def _decode_image(image_dict: dict) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(image_dict["bytes"])).convert("RGB"), dtype=np.uint8)


def _sample_positions(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 1:
        return [max(length // 2, 0)]
    # Avoid only taking the first/last frame; sample inside the episode span.
    positions = np.linspace(0.2, 0.8, count)
    return sorted({min(length - 1, int(round(p * (length - 1)))) for p in positions})


def _select_episode_files(dataset_root: Path, num_episodes: int) -> list[Path]:
    files = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {dataset_root}")
    if num_episodes >= len(files):
        return files
    idxs = np.linspace(0, len(files) - 1, num_episodes, dtype=int)
    return [files[i] for i in idxs]


def _convert_actions_to_robot(actions_chunk: np.ndarray, *, gripper_threshold: float, quat_order: str) -> np.ndarray:
    action_dim = int(actions_chunk.shape[-1])
    if action_dim == 8:
        processed = np.array(actions_chunk, copy=True)
        processed[:, 7] = np.where(processed[:, 7] > gripper_threshold, 1.0, 0.0)
        quats_xyzw = processed[:, 3:7]
        quat_norms = np.clip(np.linalg.norm(quats_xyzw, axis=1, keepdims=True), 1e-8, None)
        quats_out = quat_xyzw_to_output(quats_xyzw / quat_norms, quat_order)
        return np.concatenate([processed[:, 0:3], quats_out, processed[:, 7:8]], axis=-1)
    if action_dim == 10:
        processed = np.array(actions_chunk, copy=True)
        processed[:, 9] = np.where(processed[:, 9] > gripper_threshold, 1.0, 0.0)
        rot_matrices = rot6d_interleaved_to_rot_matrix(processed[:, 3:9])
        from scipy.spatial.transform import Rotation as R

        quats_xyzw = R.from_matrix(rot_matrices).as_quat()
        quats_out = quat_xyzw_to_output(quats_xyzw, quat_order)
        return np.concatenate([processed[:, 0:3], quats_out, processed[:, 9:10]], axis=-1)
    raise ValueError(f"Unsupported action_dim={action_dim}, expected 8 or 10.")


def _rotation_error_deg(pred_action: np.ndarray, target_action: np.ndarray) -> float:
    pred_action = np.asarray(pred_action, dtype=np.float32)
    target_action = np.asarray(target_action, dtype=np.float32)
    if pred_action.shape[0] != target_action.shape[0]:
        raise ValueError(f"Action dim mismatch: pred={pred_action.shape[0]} target={target_action.shape[0]}")
    action_dim = int(pred_action.shape[0])
    if action_dim == 8:
        from scipy.spatial.transform import Rotation as R

        pred_rot = R.from_quat(pred_action[3:7]).as_matrix()
        tgt_rot = R.from_quat(target_action[3:7]).as_matrix()
        delta = pred_rot @ tgt_rot.T
        return float(np.degrees(R.from_matrix(delta).magnitude()))
    if action_dim == 10:
        from scipy.spatial.transform import Rotation as R

        pred_rot = rot6d_interleaved_to_rot_matrix(pred_action[3:9])[0]
        tgt_rot = rot6d_interleaved_to_rot_matrix(target_action[3:9])[0]
        delta = pred_rot @ tgt_rot.T
        return float(np.degrees(R.from_matrix(delta).magnitude()))
    raise ValueError(f"Unsupported action_dim={action_dim}, expected 8 or 10.")


def _quat_norm_error(quat_action_8d: np.ndarray) -> float:
    quat = np.asarray(quat_action_8d[3:7], dtype=np.float32)
    return float(abs(np.linalg.norm(quat) - 1.0))


def _default_plot_dir_for_json(json_path: Path) -> Path:
    return json_path.parent / "plots" / json_path.stem


def _ensure_matplotlib() -> None:
    import matplotlib


def _save_training_sample_plots(payload: dict, plot_dir: Path) -> list[Path]:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    results = payload.get("results", [])
    if not results:
        raise ValueError("No results found in JSON payload (expected key: 'results').")

    pos_l2 = np.asarray([r["position_l2"] for r in results], dtype=np.float32)
    rot_deg = np.asarray([r["rotation_error_deg"] for r in results], dtype=np.float32)
    infer_ms = np.asarray([r["infer_ms"] for r in results], dtype=np.float32)
    gripper_pred = np.asarray([r["gripper_pred"] for r in results], dtype=np.float32)
    gripper_target = np.asarray([r["gripper_target"] for r in results], dtype=np.float32)

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(pos_l2, bins=min(30, max(5, len(pos_l2))), color="#4C78A8", alpha=0.9)
    ax1.set_title("Position L2")
    ax1.set_xlabel("L2 (meters)")
    ax1.set_ylabel("count")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(rot_deg, bins=min(30, max(5, len(rot_deg))), color="#F58518", alpha=0.9)
    ax2.set_title("Rotation Error")
    ax2.set_xlabel("deg")
    ax2.set_ylabel("count")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(pos_l2, rot_deg, s=18, alpha=0.85, color="#54A24B")
    ax3.set_title("Position vs Rotation")
    ax3.set_xlabel("L2 (meters)")
    ax3.set_ylabel("deg")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(infer_ms, bins=min(30, max(5, len(infer_ms))), color="#B279A2", alpha=0.9)
    ax4.set_title("Infer Latency")
    ax4.set_xlabel("ms")
    ax4.set_ylabel("count")

    fig.tight_layout()
    out1 = plot_dir / "training_samples_overview.png"
    fig.savefig(out1, dpi=200)
    plt.close(fig)
    saved.append(out1)

    cm = np.zeros((2, 2), dtype=np.int32)
    pred01 = (gripper_pred > 0.5).astype(np.int32)
    tgt01 = (gripper_target > 0.5).astype(np.int32)
    for p, t in zip(pred01, tgt01, strict=False):
        cm[t, p] += 1

    fig2 = plt.figure(figsize=(4.8, 4.2))
    ax = fig2.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Gripper Confusion (target x pred)")
    ax.set_xlabel("pred")
    ax.set_ylabel("target")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center", color="black")
    fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.tight_layout()
    out2 = plot_dir / "training_samples_gripper_confusion.png"
    fig2.savefig(out2, dpi=200)
    plt.close(fig2)
    saved.append(out2)

    return saved


def _run_eval(args: argparse.Namespace) -> dict:
    import pandas as pd
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    checkpoint_dir = Path(args.checkpoint_dir)
    dataset_root = Path(args.dataset_root)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config_name),
        checkpoint_dir,
        default_prompt=args.prompt,
    )

    selected_files = _select_episode_files(dataset_root, args.num_episodes)
    results = []

    for parquet_file in selected_files:
        df = pd.read_parquet(parquet_file)
        row_positions = _sample_positions(len(df), args.frames_per_episode)
        for row_idx in row_positions:
            row = df.iloc[row_idx]
            observation = {
                "images": {
                    "head_color": _decode_image(row["image"]),
                    "hand_left_color": _decode_image(row["wrist_image"]),
                },
                "state": np.asarray(row["state"], dtype=np.float32),
            }
            model_result = policy.infer(observation)
            pred_chunk = np.asarray(model_result["actions"], dtype=np.float32)
            pred_first = pred_chunk[0]
            target_first = np.asarray(row["actions"], dtype=np.float32)

            pos_l2 = float(np.linalg.norm(pred_first[:3] - target_first[:3]))
            pos_l1 = float(np.mean(np.abs(pred_first[:3] - target_first[:3])))
            rot_deg = _rotation_error_deg(pred_first, target_first)
            gripper_idx = 7 if int(pred_first.shape[0]) == 8 else 9
            pred_gripper = float(pred_first[gripper_idx] > args.gripper_threshold)
            target_gripper = float(target_first[gripper_idx] > args.gripper_threshold)
            gripper_correct = bool(pred_gripper == target_gripper)

            pred_robot_first = _convert_actions_to_robot(
                pred_chunk[:1],
                gripper_threshold=args.gripper_threshold,
                quat_order=args.output_quat_order,
            )[0]
            target_robot_first = _convert_actions_to_robot(
                target_first[None, :],
                gripper_threshold=args.gripper_threshold,
                quat_order=args.output_quat_order,
            )[0]
            pred_robot_chunk = _convert_actions_to_robot(
                pred_chunk,
                gripper_threshold=args.gripper_threshold,
                quat_order=args.output_quat_order,
            )

            action_chunk_shape_ok = (
                pred_chunk.ndim == 2 and pred_chunk.shape[0] == 10 and pred_chunk.shape[1] in (8, 10)
            )
            robot_chunk_shape_ok = (
                pred_robot_chunk.ndim == 2 and pred_robot_chunk.shape[0] == 10 and pred_robot_chunk.shape[1] == 8
            )
            output_finite_ok = bool(np.all(np.isfinite(pred_chunk)) and np.all(np.isfinite(pred_robot_chunk)))
            gripper_binary_ok = bool(np.all(np.isin(pred_robot_chunk[:, 7], [0.0, 1.0])))
            quat_norm_error = _quat_norm_error(pred_robot_first)
            quat_norm_ok = quat_norm_error < 1e-3

            results.append(
                {
                    "episode_file": parquet_file.name,
                    "row_idx": int(row_idx),
                    "frame_index": int(row["frame_index"]),
                    "episode_index": int(row["episode_index"]),
                    "infer_ms": float(model_result["policy_timing"]["infer_ms"]),
                    "position_l2": pos_l2,
                    "position_l1_xyz": pos_l1,
                    "rotation_error_deg": rot_deg,
                    "gripper_pred": pred_gripper,
                    "gripper_target": target_gripper,
                    "gripper_correct": gripper_correct,
                    "action_chunk_shape_ok": action_chunk_shape_ok,
                    "robot_chunk_shape_ok": robot_chunk_shape_ok,
                    "output_finite_ok": output_finite_ok,
                    "gripper_binary_ok": gripper_binary_ok,
                    "quat_norm_error": quat_norm_error,
                    "quat_norm_ok": quat_norm_ok,
                    "pred_action_model_first": pred_first.tolist(),
                    "target_action_model": target_first.tolist(),
                    "pred_action_8d_first": pred_robot_first.tolist(),
                    "target_action_8d": target_robot_first.tolist(),
                }
            )
            print(
                f"[{parquet_file.name} row={row_idx}] "
                f"pos_l2={pos_l2:.4f}, rot_deg={rot_deg:.2f}, "
                f"gripper={'ok' if gripper_correct else 'bad'}, "
                f"format={'ok' if action_chunk_shape_ok and robot_chunk_shape_ok and output_finite_ok and gripper_binary_ok and quat_norm_ok else 'bad'}, "
                f"infer_ms={model_result['policy_timing']['infer_ms']:.1f}"
            )

    summary = {
        "num_samples": len(results),
        "mean_position_l2": float(np.mean([r["position_l2"] for r in results])) if results else None,
        "mean_position_l1_xyz": float(np.mean([r["position_l1_xyz"] for r in results])) if results else None,
        "mean_rotation_error_deg": float(np.mean([r["rotation_error_deg"] for r in results])) if results else None,
        "gripper_accuracy": float(np.mean([r["gripper_correct"] for r in results])) if results else None,
        "format_ok_rate": float(
            np.mean(
                [
                    r["action_chunk_shape_ok"]
                    and r["robot_chunk_shape_ok"]
                    and r["output_finite_ok"]
                    and r["gripper_binary_ok"]
                    and r["quat_norm_ok"]
                    for r in results
                ]
            )
        )
        if results
        else None,
        "mean_quat_norm_error": float(np.mean([r["quat_norm_error"] for r in results])) if results else None,
        "mean_infer_ms": float(np.mean([r["infer_ms"] for r in results])) if results else None,
    }

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "results": results}


def main() -> None:
    args = _parse_args()
    if args.plot_json:
        json_path = Path(args.plot_json)
        payload = json.loads(json_path.read_text())
        plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir_for_json(json_path)
        saved = _save_training_sample_plots(payload, plot_dir)
        print(f"\nSaved {len(saved)} plot(s) to {plot_dir}")
        return

    payload = _run_eval(args)

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved detailed results to {output_path}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        saved = _save_training_sample_plots(payload, plot_dir)
        print(f"\nSaved {len(saved)} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
