import argparse
import base64
import json
import math
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agibot G2 left-arm bridge client for local VLA inference.")
    parser.add_argument("--server-url", required=True, help="Inference server URL, e.g. http://10.42.1.10:18080/infer")
    parser.add_argument("--loop-hz", type=float, default=2.0)
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Infer only, do not send robot control commands.")
    parser.add_argument("--camera-timeout-ms", type=float, default=1000.0)
    parser.add_argument("--request-timeout-s", type=float, default=10.0)
    parser.add_argument("--action-index", type=int, default=9)
    parser.add_argument("--state-quat-order", choices=["xyzw", "wxyz"], default="xyzw")
    parser.add_argument("--left-ee-frame-name", default="arm_l_end_link")
    parser.add_argument("--right-ee-frame-name", default="arm_r_end_link")
    parser.add_argument(
        "--request-left-arm-control",
        action="store_true",
        help="Request left arm VLA cartesian impedance control mode on startup.",
    )
    parser.add_argument("--control-mode-priority", type=int, default=150)
    parser.add_argument("--control-mode-safe-mode", choices=["normal", "reduced"], default="reduced")
    parser.add_argument("--control-mode-wait-s", type=float, default=1.0)
    parser.add_argument("--ready-pose", help="Optional 8D ready pose csv: x,y,z,qx,qy,qz,qw,gripper")
    parser.add_argument("--ready-pose-json", help="Optional JSON file containing 8D ready pose.")
    parser.add_argument(
        "--require-ready-pose",
        action="store_true",
        help="Require current left-arm state to stay near the configured ready pose before execution.",
    )
    parser.add_argument("--ready-pose-pos-tol-m", type=float, default=0.05)
    parser.add_argument("--ready-pose-rot-tol-deg", type=float, default=10.0)
    parser.add_argument("--pose-life-time", type=float, default=0.15)
    parser.add_argument(
        "--pose-stream-duration-s",
        type=float,
        default=0.5,
        help="Repeat the same cartesian pose command for this duration to match collector-style servo streaming.",
    )
    parser.add_argument(
        "--pose-stream-rate-hz",
        type=float,
        default=20.0,
        help="Rate used when repeating the same cartesian pose command.",
    )
    parser.add_argument("--max-translation-step-m", type=float, default=0.08)
    parser.add_argument("--max-rotation-step-deg", type=float, default=20.0)
    parser.add_argument(
        "--unsafe-raw-policy",
        choices=["block", "clip"],
        default="block",
        help="How to handle oversized raw predictions: block execution, or execute only the clipped command.",
    )
    parser.add_argument(
        "--execute-translation-step-m",
        type=float,
        default=0.02,
        help="Max translation actually executed per control cycle after safety filtering.",
    )
    parser.add_argument(
        "--execute-rotation-step-deg",
        type=float,
        default=5.0,
        help="Max rotation actually executed per control cycle after safety filtering.",
    )
    parser.add_argument(
        "--prediction-stable-pos-threshold-m",
        type=float,
        default=0.03,
        help="Raw predictions must stay within this delta across frames to count as stable.",
    )
    parser.add_argument(
        "--prediction-stable-rot-threshold-deg",
        type=float,
        default=8.0,
        help="Raw predictions must stay within this rotation delta across frames to count as stable.",
    )
    parser.add_argument(
        "--prediction-stable-frames",
        type=int,
        default=2,
        help="Number of consecutive similar raw predictions required before execution is allowed.",
    )
    parser.add_argument(
        "--gripper-debounce-frames",
        type=int,
        default=2,
        help="Consecutive same gripper predictions required before changing gripper command.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=2,
        help="Initial frames to observe only before any execution is allowed.",
    )
    parser.add_argument(
        "--current-gripper-close-threshold",
        type=float,
        default=-0.39,
        help="Current gripper actuator position > threshold means closed (1). Actuator range is approximately [-0.785, 0.0].",
    )
    parser.add_argument("--gripper-open-pos", type=float, default=-0.785)
    parser.add_argument("--gripper-close-pos", type=float, default=0.0)
    parser.add_argument(
        "--open-gripper-on-startup",
        action="store_true",
        help="Send an explicit open-gripper command before the main inference loop to match training-time initial state.",
    )
    parser.add_argument(
        "--startup-gripper-open-repeats",
        type=int,
        default=3,
        help="Number of repeated open-gripper commands sent on startup when --open-gripper-on-startup is enabled.",
    )
    parser.add_argument(
        "--startup-gripper-open-wait-s",
        type=float,
        default=0.5,
        help="Wait time between repeated startup open-gripper commands.",
    )
    parser.add_argument("--workspace-x-min", type=float)
    parser.add_argument("--workspace-x-max", type=float)
    parser.add_argument("--workspace-y-min", type=float)
    parser.add_argument("--workspace-y-max", type=float)
    parser.add_argument("--workspace-z-min", type=float)
    parser.add_argument("--workspace-z-max", type=float)
    parser.add_argument("--save-last-json", help="Optional path to save the latest bridge IO JSON.")
    return parser.parse_args()


def _rotation_delta_deg(quat_a_xyzw: np.ndarray, quat_b_xyzw: np.ndarray) -> float:
    quat_a = np.asarray(quat_a_xyzw, dtype=np.float32)
    quat_b = np.asarray(quat_b_xyzw, dtype=np.float32)
    quat_a = quat_a / np.clip(np.linalg.norm(quat_a), 1e-8, None)
    quat_b = quat_b / np.clip(np.linalg.norm(quat_b), 1e-8, None)
    dot = float(np.clip(abs(np.dot(quat_a, quat_b)), -1.0, 1.0))
    return float(math.degrees(2.0 * math.acos(dot)))


def _normalize_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    return quat / np.clip(np.linalg.norm(quat), 1e-8, None)


def _load_ready_pose(args: argparse.Namespace) -> np.ndarray | None:
    if args.ready_pose_json:
        values = json.loads(open(args.ready_pose_json, "r", encoding="utf-8").read())
    elif args.ready_pose:
        values = [float(v.strip()) for v in args.ready_pose.split(",") if v.strip()]
    else:
        return None
    ready = np.asarray(values, dtype=np.float32)
    if ready.shape != (8,):
        raise ValueError(f"Ready pose must be 8D, got shape {ready.shape}.")
    ready[3:7] = _normalize_quat(ready[3:7])
    ready[7] = 1.0 if ready[7] > 0.5 else 0.0
    return ready


def _slerp_quat(quat_from_xyzw: np.ndarray, quat_to_xyzw: np.ndarray, fraction: float) -> np.ndarray:
    q0 = _normalize_quat(quat_from_xyzw)
    q1 = _normalize_quat(quat_to_xyzw)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        out = q0 + fraction * (q1 - q0)
        return _normalize_quat(out)
    theta_0 = math.acos(dot)
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / max(sin_theta_0, 1e-8)
    s1 = sin_theta / max(sin_theta_0, 1e-8)
    return _normalize_quat((s0 * q0 + s1 * q1).astype(np.float32))


def _limit_pose_step(current_8d_xyzw: np.ndarray, target_8d_xyzw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    current = np.asarray(current_8d_xyzw, dtype=np.float32)
    target = np.asarray(target_8d_xyzw, dtype=np.float32)
    out = current.copy()

    pos_delta = target[:3] - current[:3]
    pos_norm = float(np.linalg.norm(pos_delta))
    if pos_norm > args.execute_translation_step_m > 0:
        pos_delta = pos_delta * (args.execute_translation_step_m / pos_norm)
    out[:3] = current[:3] + pos_delta

    rot_delta_deg = _rotation_delta_deg(current[3:7], target[3:7])
    if rot_delta_deg <= 1e-6 or args.execute_rotation_step_deg <= 0:
        out[3:7] = _normalize_quat(target[3:7])
    else:
        fraction = min(1.0, args.execute_rotation_step_deg / rot_delta_deg)
        out[3:7] = _slerp_quat(current[3:7], target[3:7], fraction)

    out[7] = float(target[7] > 0.5)
    return out.astype(np.float32)


class _SafetyFilter:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._ready_pose = _load_ready_pose(args)
        self._frame_count = 0
        self._prev_raw_action: np.ndarray | None = None
        self._stable_count = 0
        self._pending_gripper: float | None = None
        self._pending_gripper_count = 0
        self._commanded_gripper: float | None = None

    def _within_workspace(self, pose_8d_xyzw: np.ndarray) -> bool:
        x, y, z = [float(v) for v in pose_8d_xyzw[:3]]
        if self._args.workspace_x_min is not None and x < self._args.workspace_x_min:
            return False
        if self._args.workspace_x_max is not None and x > self._args.workspace_x_max:
            return False
        if self._args.workspace_y_min is not None and y < self._args.workspace_y_min:
            return False
        if self._args.workspace_y_max is not None and y > self._args.workspace_y_max:
            return False
        if self._args.workspace_z_min is not None and z < self._args.workspace_z_min:
            return False
        if self._args.workspace_z_max is not None and z > self._args.workspace_z_max:
            return False
        return True

    def _at_ready_pose(self, current_state_8d: np.ndarray) -> tuple[bool, float | None, float | None]:
        if self._ready_pose is None:
            return True, None, None
        pos_delta = float(np.linalg.norm(current_state_8d[:3] - self._ready_pose[:3]))
        rot_delta = _rotation_delta_deg(current_state_8d[3:7], self._ready_pose[3:7])
        ok = pos_delta <= self._args.ready_pose_pos_tol_m and rot_delta <= self._args.ready_pose_rot_tol_deg
        return ok, pos_delta, rot_delta

    def process(self, current_state_8d: np.ndarray, raw_action_8d: np.ndarray) -> dict:
        self._frame_count += 1
        current = np.asarray(current_state_8d, dtype=np.float32)
        raw = np.asarray(raw_action_8d, dtype=np.float32)

        raw_pos_delta = float(np.linalg.norm(raw[:3] - current[:3]))
        raw_rot_delta_deg = _rotation_delta_deg(raw[3:7], current[3:7])
        raw_safe = (
            raw_pos_delta <= self._args.max_translation_step_m
            and raw_rot_delta_deg <= self._args.max_rotation_step_deg
        )

        if self._prev_raw_action is None:
            self._stable_count = 1
            prediction_delta_pos = None
            prediction_delta_rot = None
        else:
            prediction_delta_pos = float(np.linalg.norm(raw[:3] - self._prev_raw_action[:3]))
            prediction_delta_rot = _rotation_delta_deg(raw[3:7], self._prev_raw_action[3:7])
            if (
                prediction_delta_pos <= self._args.prediction_stable_pos_threshold_m
                and prediction_delta_rot <= self._args.prediction_stable_rot_threshold_deg
                and float(raw[7] > 0.5) == float(self._prev_raw_action[7] > 0.5)
            ):
                self._stable_count += 1
            else:
                self._stable_count = 1
        self._prev_raw_action = raw.copy()

        raw_gripper = float(raw[7] > 0.5)
        current_gripper = float(current[7] > 0.5)
        if self._commanded_gripper is None:
            self._commanded_gripper = current_gripper
        if raw_gripper == self._commanded_gripper:
            self._pending_gripper = raw_gripper
            self._pending_gripper_count = self._args.gripper_debounce_frames
        else:
            if self._pending_gripper == raw_gripper:
                self._pending_gripper_count += 1
            else:
                self._pending_gripper = raw_gripper
                self._pending_gripper_count = 1
            if self._pending_gripper_count >= self._args.gripper_debounce_frames:
                self._commanded_gripper = raw_gripper

        warmed_up = self._frame_count > self._args.warmup_frames
        stable_enough = self._stable_count >= self._args.prediction_stable_frames
        within_workspace = self._within_workspace(raw)
        at_ready_pose, ready_pos_delta, ready_rot_delta = self._at_ready_pose(current)
        ready_gate_ok = at_ready_pose or not self._args.require_ready_pose
        execution_ready = warmed_up and stable_enough and (
            raw_safe or self._args.unsafe_raw_policy == "clip"
        ) and within_workspace and ready_gate_ok

        commanded = _limit_pose_step(current, raw, self._args)
        commanded[7] = self._commanded_gripper
        return {
            "commanded_action_8d_xyzw": commanded,
            "raw_safe": raw_safe,
            "raw_pos_delta_m": raw_pos_delta,
            "raw_rot_delta_deg": raw_rot_delta_deg,
            "prediction_delta_pos_m": prediction_delta_pos,
            "prediction_delta_rot_deg": prediction_delta_rot,
            "stable_count": self._stable_count,
            "warmed_up": warmed_up,
            "stable_enough": stable_enough,
            "within_workspace": within_workspace,
            "at_ready_pose": at_ready_pose,
            "ready_gate_ok": ready_gate_ok,
            "ready_pos_delta_m": ready_pos_delta,
            "ready_rot_delta_deg": ready_rot_delta,
            "execution_ready": execution_ready,
            "commanded_gripper": self._commanded_gripper,
            "pending_gripper": self._pending_gripper,
            "pending_gripper_count": self._pending_gripper_count,
            "unsafe_raw_policy": self._args.unsafe_raw_policy,
        }


def _encode_image_to_b64(image: Any, agibot_gdk: Any) -> str:
    if image is None:
        raise ValueError("Camera returned no image.")
    image_bytes = bytes(image.data)
    if image.encoding in {agibot_gdk.Encoding.JPEG, agibot_gdk.Encoding.PNG}:
        return base64.b64encode(image_bytes).decode("ascii")

    raise ValueError(
        f"Unsupported image encoding {image.encoding}. Current bridge expects JPEG/PNG camera stream from GDK."
    )


def _capture_left_state(robot: Any, tf: Any, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    pose = tf.get_tf_from_base_link(args.left_ee_frame_name)
    end_state = robot.get_end_state()["left_end_state"]
    if not end_state["end_states"]:
        raise ValueError("Left end effector state is empty.")
    gripper_raw = float(end_state["end_states"][0]["position"])
    gripper_binary = 1.0 if gripper_raw > args.current_gripper_close_threshold else 0.0

    state_8d = np.asarray(
        [
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
            pose.rotation.x,
            pose.rotation.y,
            pose.rotation.z,
            pose.rotation.w,
            gripper_binary,
        ],
        dtype=np.float32,
    )
    state_8d[3:7] = state_8d[3:7] / np.clip(np.linalg.norm(state_8d[3:7]), 1e-8, None)
    return state_8d, gripper_raw


def _infer(server_url: str, payload: dict, timeout_s: float) -> dict:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _apply_pose(robot: Any, tf: Any, agibot_gdk: Any, action_8d_xyzw: np.ndarray, args: argparse.Namespace) -> int:
    right_pose = tf.get_tf_from_base_link(args.right_ee_frame_name)
    cmd = agibot_gdk.EndEffectorPose()
    cmd.group = agibot_gdk.EndEffectorControlGroup.kBothArms
    cmd.life_time = args.pose_life_time
    cmd.left_end_effector_pose.position.x = float(action_8d_xyzw[0])
    cmd.left_end_effector_pose.position.y = float(action_8d_xyzw[1])
    cmd.left_end_effector_pose.position.z = float(action_8d_xyzw[2])
    cmd.left_end_effector_pose.orientation.x = float(action_8d_xyzw[3])
    cmd.left_end_effector_pose.orientation.y = float(action_8d_xyzw[4])
    cmd.left_end_effector_pose.orientation.z = float(action_8d_xyzw[5])
    cmd.left_end_effector_pose.orientation.w = float(action_8d_xyzw[6])
    cmd.right_end_effector_pose.position.x = float(right_pose.translation.x)
    cmd.right_end_effector_pose.position.y = float(right_pose.translation.y)
    cmd.right_end_effector_pose.position.z = float(right_pose.translation.z)
    cmd.right_end_effector_pose.orientation.x = float(right_pose.rotation.x)
    cmd.right_end_effector_pose.orientation.y = float(right_pose.rotation.y)
    cmd.right_end_effector_pose.orientation.z = float(right_pose.rotation.z)
    cmd.right_end_effector_pose.orientation.w = float(right_pose.rotation.w)
    repeats = max(1, int(args.pose_stream_duration_s * max(args.pose_stream_rate_hz, 1e-6)))
    period_s = 1.0 / max(args.pose_stream_rate_hz, 1e-6)
    ret = 0
    for _ in range(repeats):
        ret = int(robot.end_effector_pose_control(cmd))
        if repeats > 1:
            time.sleep(period_s)
    return ret


def _apply_gripper(robot: Any, agibot_gdk: Any, gripper_binary: float, args: argparse.Namespace) -> int:
    joint_states = agibot_gdk.JointStates()
    joint_states.group = "left_tool"
    joint_states.target_type = "omnipicker"
    joint_state = agibot_gdk.JointState()
    joint_state.position = float(args.gripper_close_pos if gripper_binary > 0.5 else args.gripper_open_pos)
    joint_states.states = [joint_state]
    joint_states.nums = len(joint_states.states)
    return int(robot.move_ee_pos(joint_states))


def _open_gripper_on_startup(robot: Any, agibot_gdk: Any, args: argparse.Namespace) -> None:
    repeats = max(1, int(args.startup_gripper_open_repeats))
    for _ in range(repeats):
        _apply_gripper(robot, agibot_gdk, 0.0, args)
        if args.startup_gripper_open_wait_s > 0:
            time.sleep(args.startup_gripper_open_wait_s)


def _request_left_arm_control(robot: Any, agibot_gdk: Any, args: argparse.Namespace) -> dict[str, Any]:
    mode = agibot_gdk.MotionControlMode()
    mode.input_source = agibot_gdk.INPUT_VLA
    mode.target = agibot_gdk.TARGET_LEFT_ARM
    mode.control_mode = agibot_gdk.CTRL_CARTESIAN_IMPEDANCE
    mode.safe_mode = (
        agibot_gdk.SAFE_REDUCED if args.control_mode_safe_mode == "reduced" else agibot_gdk.SAFE_NORMAL
    )
    mode.priority = int(args.control_mode_priority)
    ret = int(robot.set_control_mode(mode))
    if args.control_mode_wait_s > 0:
        time.sleep(args.control_mode_wait_s)
    status = robot.get_whole_body_status()
    return {
        "set_control_mode_ret": ret,
        "left_arm_control": bool(status.get("left_arm_control", False)),
        "left_arm_estop": bool(status.get("left_arm_estop", False)),
        "left_arm_error": int(status.get("left_arm_error", -1)),
        "status": status,
    }


def main() -> None:
    args = _parse_args()

    import agibot_gdk

    agibot_gdk.gdk_init()
    robot = agibot_gdk.Robot()
    camera = agibot_gdk.Camera()
    tf = agibot_gdk.TF()
    time.sleep(2.0)
    if args.request_left_arm_control:
        control_info = _request_left_arm_control(robot, agibot_gdk, args)
        print(json.dumps({"control_request": control_info}, ensure_ascii=True))
    if args.open_gripper_on_startup:
        _open_gripper_on_startup(robot, agibot_gdk, args)

    period_s = 1.0 / max(args.loop_hz, 1e-6)
    safety_filter = _SafetyFilter(args)

    while True:
        loop_start = time.time()
        try:
            head = camera.get_latest_image(agibot_gdk.CameraType.kHeadColor, args.camera_timeout_ms)
            hand_left = camera.get_latest_image(agibot_gdk.CameraType.kHandLeftColor, args.camera_timeout_ms)
            state_8d, gripper_raw = _capture_left_state(robot, tf, args)

            payload = {
                "state": state_8d.tolist(),
                "state_quat_order": args.state_quat_order,
                "action_index": args.action_index,
                "head_image_b64": _encode_image_to_b64(head, agibot_gdk),
                "hand_left_image_b64": _encode_image_to_b64(hand_left, agibot_gdk),
            }
            response = _infer(args.server_url, payload, args.request_timeout_s)
            if not response.get("ok", False):
                raise RuntimeError(f"Inference server error: {response}")

            raw_action = np.asarray(response["selected_action_8d_xyzw"], dtype=np.float32)
            safety_info = safety_filter.process(state_8d, raw_action)
            commanded_action = np.asarray(safety_info["commanded_action_8d_xyzw"], dtype=np.float32)
            cmd_pos_delta = float(np.linalg.norm(commanded_action[:3] - state_8d[:3]))
            cmd_rot_delta_deg = _rotation_delta_deg(commanded_action[3:7], state_8d[3:7])

            pose_ret = None
            gripper_ret = None
            if not args.dry_run:
                if not safety_info["execution_ready"]:
                    raise RuntimeError(
                        "Execution not ready: "
                        f"raw_safe={safety_info['raw_safe']} "
                        f"warmed_up={safety_info['warmed_up']} "
                        f"stable_enough={safety_info['stable_enough']} "
                        f"stable_count={safety_info['stable_count']}"
                    )
                pose_ret = _apply_pose(robot, tf, agibot_gdk, commanded_action, args)
                gripper_ret = _apply_gripper(robot, agibot_gdk, float(commanded_action[7]), args)

            log_payload = {
                "timestamp": time.time(),
                "state_8d_xyzw": state_8d.tolist(),
                "current_gripper_raw": gripper_raw,
                "pred_action_8d_xyzw": raw_action.tolist(),
                "commanded_action_8d_xyzw": commanded_action.tolist(),
                "raw_pos_delta_m": safety_info["raw_pos_delta_m"],
                "raw_rot_delta_deg": safety_info["raw_rot_delta_deg"],
                "cmd_pos_delta_m": cmd_pos_delta,
                "cmd_rot_delta_deg": cmd_rot_delta_deg,
                "raw_safe": safety_info["raw_safe"],
                "warmed_up": safety_info["warmed_up"],
                "stable_enough": safety_info["stable_enough"],
                "stable_count": safety_info["stable_count"],
                "within_workspace": safety_info["within_workspace"],
                "at_ready_pose": safety_info["at_ready_pose"],
                "ready_gate_ok": safety_info["ready_gate_ok"],
                "ready_pos_delta_m": safety_info["ready_pos_delta_m"],
                "ready_rot_delta_deg": safety_info["ready_rot_delta_deg"],
                "prediction_delta_pos_m": safety_info["prediction_delta_pos_m"],
                "prediction_delta_rot_deg": safety_info["prediction_delta_rot_deg"],
                "execution_ready": safety_info["execution_ready"],
                "commanded_gripper": safety_info["commanded_gripper"],
                "pending_gripper": safety_info["pending_gripper"],
                "pending_gripper_count": safety_info["pending_gripper_count"],
                "unsafe_raw_policy": safety_info["unsafe_raw_policy"],
                "pose_ret": pose_ret,
                "gripper_ret": gripper_ret,
                "policy_timing": response.get("policy_timing"),
                "bridge_latency_ms": response.get("receive_to_reply_ms"),
                "dry_run": args.dry_run,
            }
            print(json.dumps(log_payload, ensure_ascii=True))

            if args.save_last_json:
                with open(args.save_last_json, "w", encoding="utf-8") as f:
                    json.dump(log_payload, f, indent=2)
        except urllib.error.URLError as exc:
            print(json.dumps({"ok": False, "error": f"server_unreachable: {exc}"}, ensure_ascii=True))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=True))

        if args.run_once:
            break

        elapsed = time.time() - loop_start
        time.sleep(max(0.0, period_s - elapsed))


if __name__ == "__main__":
    main()
