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
    parser = argparse.ArgumentParser(description="Run sequential replay evaluation on full training episodes.")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--episode-indices",
        help="Comma separated parquet episode indices to replay, e.g. 0,25,50. If omitted, representative episodes are chosen.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="How many representative episodes to replay when --episode-indices is not provided.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        help="Optional step cap per episode for faster debugging.",
    )
    parser.add_argument(
        "--output-quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order used for robot-format action dumps.",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--position-jitter-threshold", type=float, default=0.003)
    parser.add_argument("--rotation-jitter-threshold-deg", type=float, default=3.0)
    parser.add_argument("--save-output", help="Optional JSON path to save detailed replay results.")
    parser.add_argument("--plot-dir", help="Optional directory to save plots (PNG).")
    parser.add_argument(
        "--plot-json",
        help="Optional path to an existing replay evaluation JSON. When provided, skips inference and only generates plots.",
    )
    return parser.parse_args()


def _decode_image(image_dict: dict) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(image_dict["bytes"])).convert("RGB"), dtype=np.uint8)


def _select_episode_files(dataset_root: Path, args: argparse.Namespace) -> list[Path]:
    files = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root}")
    if args.episode_indices:
        selected = []
        for idx in [int(v.strip()) for v in args.episode_indices.split(",") if v.strip()]:
            path = dataset_root / "data" / "chunk-000" / f"episode_{idx:06d}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"Requested episode parquet not found: {path}")
            selected.append(path)
        return selected

    if args.num_episodes >= len(files):
        return files
    idxs = np.linspace(0, len(files) - 1, args.num_episodes, dtype=int)
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


def _rotation_step_deg(prev_action: np.ndarray, next_action: np.ndarray) -> float:
    prev_action = np.asarray(prev_action, dtype=np.float32)
    next_action = np.asarray(next_action, dtype=np.float32)
    if prev_action.shape[0] != next_action.shape[0]:
        raise ValueError(f"Action dim mismatch: prev={prev_action.shape[0]} next={next_action.shape[0]}")
    action_dim = int(prev_action.shape[0])
    if action_dim == 8:
        from scipy.spatial.transform import Rotation as R

        prev_rot = R.from_quat(prev_action[3:7]).as_matrix()
        next_rot = R.from_quat(next_action[3:7]).as_matrix()
        delta = next_rot @ prev_rot.T
        return float(np.degrees(R.from_matrix(delta).magnitude()))
    if action_dim == 10:
        from scipy.spatial.transform import Rotation as R

        prev_rot = rot6d_interleaved_to_rot_matrix(prev_action[3:9])[0]
        next_rot = rot6d_interleaved_to_rot_matrix(next_action[3:9])[0]
        delta = next_rot @ prev_rot.T
        return float(np.degrees(R.from_matrix(delta).magnitude()))
    raise ValueError(f"Unsupported action_dim={action_dim}, expected 8 or 10.")


def _quat_norm_error(quat_action_8d: np.ndarray) -> float:
    quat = np.asarray(quat_action_8d[3:7], dtype=np.float32)
    return float(abs(np.linalg.norm(quat) - 1.0))


def _episode_summary(step_results: list[dict], *, first_compile_ms: float) -> dict:
    pos_l2 = [r["position_l2"] for r in step_results]
    rot_deg = [r["rotation_error_deg"] for r in step_results]
    infer_ms = [r["infer_ms"] for r in step_results]
    format_ok = [
        r["action_chunk_shape_ok"]
        and r["robot_chunk_shape_ok"]
        and r["output_finite_ok"]
        and r["gripper_binary_ok"]
        and r["quat_norm_ok"]
        for r in step_results
    ]
    return {
        "num_steps": len(step_results),
        "mean_position_l2": float(np.mean(pos_l2)) if pos_l2 else None,
        "p95_position_l2": float(np.percentile(pos_l2, 95)) if pos_l2 else None,
        "mean_rotation_error_deg": float(np.mean(rot_deg)) if rot_deg else None,
        "p95_rotation_error_deg": float(np.percentile(rot_deg, 95)) if rot_deg else None,
        "gripper_accuracy": float(np.mean([r["gripper_correct"] for r in step_results])) if step_results else None,
        "false_gripper_triggers": int(sum(r["false_gripper_trigger"] for r in step_results)),
        "position_jitter_flags": int(sum(r["position_jitter_flag"] for r in step_results)),
        "rotation_jitter_flags": int(sum(r["rotation_jitter_flag"] for r in step_results)),
        "format_ok_rate": float(np.mean(format_ok)) if format_ok else None,
        "mean_quat_norm_error": float(np.mean([r["quat_norm_error"] for r in step_results])) if step_results else None,
        "mean_infer_ms": float(np.mean(infer_ms)) if infer_ms else None,
        "steady_mean_infer_ms": float(np.mean(infer_ms[1:])) if len(infer_ms) > 1 else None,
        "first_compile_infer_ms": float(first_compile_ms),
    }


def _default_plot_dir_for_json(json_path: Path) -> Path:
    return json_path.parent / "plots" / json_path.stem


def _ensure_matplotlib() -> None:
    import matplotlib


def _flatten_episode_steps(payload: dict) -> list[dict]:
    episodes = payload.get("episodes", [])
    if not episodes:
        raise ValueError("No episodes found in JSON payload (expected key: 'episodes').")
    flattened: list[dict] = []
    for ep in episodes:
        name = ep.get("episode_file", "episode_unknown")
        for s in ep.get("steps", []):
            s2 = dict(s)
            s2["_episode_file"] = name
            flattened.append(s2)
    return flattened


def _save_episode_replay_plots(payload: dict, plot_dir: Path) -> list[Path]:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    episodes = payload.get("episodes", [])
    for ep in episodes:
        name = ep.get("episode_file", "episode_unknown")
        steps = ep.get("steps", [])
        if not steps:
            continue
        pos_l2 = np.asarray([s["position_l2"] for s in steps], dtype=np.float32)
        rot_deg = np.asarray([s["rotation_error_deg"] for s in steps], dtype=np.float32)
        infer_ms = np.asarray([s["infer_ms"] for s in steps], dtype=np.float32)
        grip_p = np.asarray([s["gripper_pred"] for s in steps], dtype=np.float32)
        grip_t = np.asarray([s["gripper_target"] for s in steps], dtype=np.float32)
        jitter_pos = np.asarray([bool(s["position_jitter_flag"]) for s in steps], dtype=np.int32)
        jitter_rot = np.asarray([bool(s["rotation_jitter_flag"]) for s in steps], dtype=np.int32)
        false_grip = np.asarray([bool(s["false_gripper_trigger"]) for s in steps], dtype=np.int32)
        t = np.arange(len(steps), dtype=np.int32)

        fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t, pos_l2, color="#4C78A8", linewidth=1.2)
        ax1.set_title("Position L2")
        ax1.set_xlabel("step")
        ax1.set_ylabel("L2 (meters)")

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t, rot_deg, color="#F58518", linewidth=1.2)
        ax2.set_title("Rotation Error")
        ax2.set_xlabel("step")
        ax2.set_ylabel("deg")

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.step(t, (grip_t > 0.5).astype(np.int32), where="post", label="target", color="#54A24B")
        ax3.step(t, (grip_p > 0.5).astype(np.int32), where="post", label="pred", color="#E45756", alpha=0.85)
        ax3.set_yticks([0, 1])
        ax3.set_ylim(-0.2, 1.2)
        ax3.set_title("Gripper (target vs pred)")
        ax3.set_xlabel("step")
        ax3.legend(loc="upper right")

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(t, infer_ms, color="#B279A2", linewidth=1.2)
        ax4.set_title("Infer Latency")
        ax4.set_xlabel("step")
        ax4.set_ylabel("ms")

        fig.suptitle(name)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        out = plot_dir / f"{Path(name).stem}_curves.png"
        fig.savefig(out, dpi=200)
        plt.close(fig)
        saved.append(out)

        fig2 = plt.figure(figsize=(12, 3.2))
        ax = fig2.add_subplot(1, 1, 1)
        ax.step(t, jitter_pos, where="post", label="pos_jitter", color="#4C78A8")
        ax.step(t, jitter_rot, where="post", label="rot_jitter", color="#F58518")
        ax.step(t, false_grip, where="post", label="false_gripper", color="#E45756")
        ax.set_yticks([0, 1])
        ax.set_ylim(-0.2, 1.2)
        ax.set_title(f"{name} flags")
        ax.set_xlabel("step")
        ax.legend(loc="upper right", ncols=3)
        fig2.tight_layout()
        out2 = plot_dir / f"{Path(name).stem}_flags.png"
        fig2.savefig(out2, dpi=200)
        plt.close(fig2)
        saved.append(out2)

    flattened = _flatten_episode_steps(payload)
    pos_l2_all = np.asarray([s["position_l2"] for s in flattened], dtype=np.float32)
    rot_deg_all = np.asarray([s["rotation_error_deg"] for s in flattened], dtype=np.float32)
    infer_ms_all = np.asarray([s["infer_ms"] for s in flattened], dtype=np.float32)

    fig3 = plt.figure(figsize=(10, 7))
    ax1 = fig3.add_subplot(2, 2, 1)
    ax1.hist(pos_l2_all, bins=40, color="#4C78A8", alpha=0.9)
    ax1.set_title("Position L2 (all steps)")
    ax1.set_xlabel("L2 (meters)")
    ax1.set_ylabel("count")

    ax2 = fig3.add_subplot(2, 2, 2)
    ax2.hist(rot_deg_all, bins=40, color="#F58518", alpha=0.9)
    ax2.set_title("Rotation Error (all steps)")
    ax2.set_xlabel("deg")
    ax2.set_ylabel("count")

    ax3 = fig3.add_subplot(2, 2, 3)
    ax3.scatter(pos_l2_all, rot_deg_all, s=6, alpha=0.45, color="#54A24B")
    ax3.set_title("Position vs Rotation (all steps)")
    ax3.set_xlabel("L2 (meters)")
    ax3.set_ylabel("deg")

    ax4 = fig3.add_subplot(2, 2, 4)
    ax4.hist(infer_ms_all, bins=40, color="#B279A2", alpha=0.9)
    ax4.set_title("Infer Latency (all steps)")
    ax4.set_xlabel("ms")
    ax4.set_ylabel("count")

    fig3.tight_layout()
    out3 = plot_dir / "episode_replay_overview.png"
    fig3.savefig(out3, dpi=200)
    plt.close(fig3)
    saved.append(out3)

    return saved


def _run_replay_eval(args: argparse.Namespace) -> dict:
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

    episode_files = _select_episode_files(dataset_root, args)
    all_episode_results = []

    for parquet_file in episode_files:
        df = pd.read_parquet(parquet_file)
        if args.max_steps_per_episode is not None:
            df = df.iloc[: args.max_steps_per_episode]

        step_results = []
        prev_pred_action = None
        prev_target_action = None
        prev_pred_gripper = None
        prev_target_gripper = None
        first_compile_ms = None

        for row_idx, row in df.iterrows():
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

            pred_robot_chunk = _convert_actions_to_robot(
                pred_chunk,
                gripper_threshold=args.gripper_threshold,
                quat_order=args.output_quat_order,
            )
            pred_robot_first = pred_robot_chunk[0]
            target_robot_first = _convert_actions_to_robot(
                target_first[None, :],
                gripper_threshold=args.gripper_threshold,
                quat_order=args.output_quat_order,
            )[0]

            pos_l2 = float(np.linalg.norm(pred_first[:3] - target_first[:3]))
            rot_deg = _rotation_error_deg(pred_first, target_first)
            gripper_idx = 7 if int(pred_first.shape[0]) == 8 else 9
            pred_gripper = float(pred_first[gripper_idx] > args.gripper_threshold)
            target_gripper = float(target_first[gripper_idx] > args.gripper_threshold)
            gripper_correct = bool(pred_gripper == target_gripper)

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

            pred_step_pos_delta = None
            target_step_pos_delta = None
            pred_step_rot_delta_deg = None
            target_step_rot_delta_deg = None
            position_jitter_flag = False
            rotation_jitter_flag = False
            false_gripper_trigger = False

            if prev_pred_action is not None and prev_target_action is not None:
                pred_step_pos_delta = float(np.linalg.norm(pred_first[:3] - prev_pred_action[:3]))
                target_step_pos_delta = float(np.linalg.norm(target_first[:3] - prev_target_action[:3]))
                pred_step_rot_delta_deg = _rotation_step_deg(prev_pred_action, pred_first)
                target_step_rot_delta_deg = _rotation_step_deg(prev_target_action, target_first)
                position_jitter_flag = pred_step_pos_delta > (
                    target_step_pos_delta + args.position_jitter_threshold
                )
                rotation_jitter_flag = pred_step_rot_delta_deg > (
                    target_step_rot_delta_deg + args.rotation_jitter_threshold_deg
                )
                false_gripper_trigger = (
                    prev_pred_gripper is not None
                    and prev_target_gripper is not None
                    and pred_gripper != prev_pred_gripper
                    and target_gripper == prev_target_gripper
                )

            infer_ms = float(model_result["policy_timing"]["infer_ms"])
            if first_compile_ms is None:
                first_compile_ms = infer_ms

            step_results.append(
                {
                    "row_idx": int(row_idx),
                    "frame_index": int(row["frame_index"]),
                    "episode_index": int(row["episode_index"]),
                    "infer_ms": infer_ms,
                    "position_l2": pos_l2,
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
                    "pred_step_pos_delta": pred_step_pos_delta,
                    "target_step_pos_delta": target_step_pos_delta,
                    "pred_step_rot_delta_deg": pred_step_rot_delta_deg,
                    "target_step_rot_delta_deg": target_step_rot_delta_deg,
                    "position_jitter_flag": position_jitter_flag,
                    "rotation_jitter_flag": rotation_jitter_flag,
                    "false_gripper_trigger": false_gripper_trigger,
                    "pred_action_model_first": pred_first.tolist(),
                    "target_action_model": target_first.tolist(),
                    "pred_action_8d_first": pred_robot_first.tolist(),
                    "target_action_8d": target_robot_first.tolist(),
                }
            )

            prev_pred_action = pred_first
            prev_target_action = target_first
            prev_pred_gripper = pred_gripper
            prev_target_gripper = target_gripper

        episode_summary = _episode_summary(step_results, first_compile_ms=first_compile_ms or 0.0)
        episode_result = {
            "episode_file": parquet_file.name,
            "summary": episode_summary,
            "steps": step_results,
        }
        all_episode_results.append(episode_result)
        print(
            f"[{parquet_file.name}] steps={episode_summary['num_steps']} "
            f"pos_l2={episode_summary['mean_position_l2']:.4f} "
            f"rot_deg={episode_summary['mean_rotation_error_deg']:.2f} "
            f"gripper_acc={episode_summary['gripper_accuracy']:.3f} "
            f"false_gripper={episode_summary['false_gripper_triggers']} "
            f"pos_jitter={episode_summary['position_jitter_flags']} "
            f"rot_jitter={episode_summary['rotation_jitter_flags']} "
            f"steady_ms={episode_summary['steady_mean_infer_ms']:.2f}"
        )

    overall_summary = {
        "num_episodes": len(all_episode_results),
        "episodes": [r["episode_file"] for r in all_episode_results],
        "mean_position_l2": float(np.mean([r["summary"]["mean_position_l2"] for r in all_episode_results])),
        "mean_rotation_error_deg": float(
            np.mean([r["summary"]["mean_rotation_error_deg"] for r in all_episode_results])
        ),
        "mean_gripper_accuracy": float(np.mean([r["summary"]["gripper_accuracy"] for r in all_episode_results])),
        "total_false_gripper_triggers": int(sum(r["summary"]["false_gripper_triggers"] for r in all_episode_results)),
        "total_position_jitter_flags": int(sum(r["summary"]["position_jitter_flags"] for r in all_episode_results)),
        "total_rotation_jitter_flags": int(sum(r["summary"]["rotation_jitter_flags"] for r in all_episode_results)),
        "mean_format_ok_rate": float(np.mean([r["summary"]["format_ok_rate"] for r in all_episode_results])),
        "mean_quat_norm_error": float(np.mean([r["summary"]["mean_quat_norm_error"] for r in all_episode_results])),
        "mean_steady_infer_ms": float(np.mean([r["summary"]["steady_mean_infer_ms"] for r in all_episode_results])),
    }

    print("\nOverall Summary:")
    print(json.dumps(overall_summary, indent=2))
    return {"overall_summary": overall_summary, "episodes": all_episode_results}


def main() -> None:
    args = _parse_args()
    if args.plot_json:
        json_path = Path(args.plot_json)
        payload = json.loads(json_path.read_text())
        plot_dir = Path(args.plot_dir) if args.plot_dir else _default_plot_dir_for_json(json_path)
        saved = _save_episode_replay_plots(payload, plot_dir)
        print(f"\nSaved {len(saved)} plot(s) to {plot_dir}")
        return

    payload = _run_replay_eval(args)

    if args.save_output:
        output_path = Path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved replay results to {output_path}")

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        saved = _save_episode_replay_plots(payload, plot_dir)
        print(f"\nSaved {len(saved)} plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()
