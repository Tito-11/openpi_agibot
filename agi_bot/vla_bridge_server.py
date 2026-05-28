import argparse
import base64
import io
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agi_bot.inference_runtime import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_PROMPT,
    convert_actions_to_robot,
    create_policy,
    get_action_dim,
    infer_actions_chunk,
    load_state_from_values,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local VLA inference for the Agibot left-arm bridge.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--default-action-index",
        type=int,
        default=9,
        help="Default predicted horizon index to execute. Use 9 for this grasp task to include the learned close-gripper action.",
    )
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=240)
    parser.add_argument("--save-debug-dir", help="Optional directory to save received samples and outputs.")
    return parser.parse_args()


def _decode_image_b64(image_b64: str) -> np.ndarray:
    image_bytes = base64.b64decode(image_b64.encode("ascii"))
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    resized = pil_image.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _rotation_delta_deg(quat_a_xyzw: np.ndarray, quat_b_xyzw: np.ndarray) -> float:
    rot_a = R.from_quat(np.asarray(quat_a_xyzw, dtype=np.float32))
    rot_b = R.from_quat(np.asarray(quat_b_xyzw, dtype=np.float32))
    return float(np.degrees((rot_a * rot_b.inv()).magnitude()))


class _BridgeRuntime:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._action_dim = get_action_dim(args.config_name)
        self._policy = create_policy(
            config_name=args.config_name,
            checkpoint_dir=args.checkpoint_dir,
            prompt=args.prompt,
        )
        self._debug_dir = Path(args.save_debug_dir) if args.save_debug_dir else None
        if self._debug_dir is not None:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def infer(self, payload: dict) -> dict:
        start = time.time()
        head_rgb = _resize_image(
            _decode_image_b64(payload["head_image_b64"]),
            self._args.input_width,
            self._args.input_height,
        )
        hand_left_rgb = _resize_image(
            _decode_image_b64(payload["hand_left_image_b64"]),
            self._args.input_width,
            self._args.input_height,
        )

        state = load_state_from_values(
            payload["state"],
            expected_state_dim=self._action_dim,
            state_quat_order=payload.get("state_quat_order", "xyzw"),
            gripper_threshold=float(payload.get("gripper_threshold", self._args.gripper_threshold)),
        )

        policy_actions, policy_timing = infer_actions_chunk(
            self._policy,
            head_image=head_rgb,
            hand_left_image=hand_left_rgb,
            state=state,
        )
        action_index = int(payload.get("action_index", self._args.default_action_index))
        if action_index < 0 or action_index >= int(policy_actions.shape[0]):
            raise ValueError(
                f"Invalid action_index={action_index}, horizon={int(policy_actions.shape[0])}."
            )
        robot_actions = convert_actions_to_robot(
            policy_actions,
            gripper_threshold=float(payload.get("gripper_threshold", self._args.gripper_threshold)),
            output_quat_order="xyzw",
        )
        selected_action = np.asarray(robot_actions[action_index], dtype=np.float32)

        if not np.all(np.isfinite(selected_action)):
            raise ValueError("Predicted controller action contains NaN or Inf.")
        quat_norm = float(np.linalg.norm(selected_action[3:7]))
        if not np.isclose(quat_norm, 1.0, atol=1e-3):
            raise ValueError(f"Predicted quaternion norm is invalid: {quat_norm}")

        current_state_raw = np.asarray(payload["state"], dtype=np.float32)
        current_state_8d = load_state_from_values(
            current_state_raw,
            expected_state_dim=8,
            state_quat_order=payload.get("state_quat_order", "xyzw"),
            gripper_threshold=float(payload.get("gripper_threshold", self._args.gripper_threshold)),
        )
        pos_delta = float(np.linalg.norm(selected_action[:3] - current_state_8d[:3]))
        rot_delta_deg = _rotation_delta_deg(selected_action[3:7], current_state_8d[3:7])

        result = {
            "ok": True,
            "config_name": self._args.config_name,
            "checkpoint_dir": str(Path(self._args.checkpoint_dir)),
            "prompt": payload.get("prompt", self._args.prompt),
            "action_index": action_index,
            "state_dim": int(state.shape[0]),
            "policy_horizon": int(policy_actions.shape[0]),
            "policy_action_dim": int(policy_actions.shape[1]),
            "policy_timing": policy_timing,
            "receive_to_reply_ms": float((time.time() - start) * 1000.0),
            "current_state_8d_xyzw": current_state_8d.tolist(),
            "selected_action_8d_xyzw": selected_action.tolist(),
            "selected_action_position_delta_m": pos_delta,
            "selected_action_rotation_delta_deg": rot_delta_deg,
        }

        if self._debug_dir is not None:
            stamp = int(time.time() * 1000)
            Image.fromarray(head_rgb).save(self._debug_dir / f"{stamp}_head.png")
            Image.fromarray(hand_left_rgb).save(self._debug_dir / f"{stamp}_hand_left.png")
            (self._debug_dir / f"{stamp}_result.json").write_text(json.dumps(result, indent=2))

        return result


class _BridgeHandler(BaseHTTPRequestHandler):
    runtime: _BridgeRuntime = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/health", "/healthz"}:
            self._send_json(
                200,
                {
                    "ok": True,
                    "config_name": self.runtime._args.config_name,
                    "checkpoint_dir": str(Path(self.runtime._args.checkpoint_dir)),
                    "default_action_index": self.runtime._args.default_action_index,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json(404, {"ok": False, "error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Empty request body.")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = self.runtime.infer(payload)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    args = _parse_args()
    runtime = _BridgeRuntime(args)
    _BridgeHandler.runtime = runtime
    server = ThreadingHTTPServer((args.host, args.port), _BridgeHandler)
    print(
        f"Serving VLA bridge on http://{args.host}:{args.port} "
        f"(config={args.config_name}, checkpoint={args.checkpoint_dir}, action_index={args.default_action_index})"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
