import argparse
import importlib.util
import time

import agibot_gdk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset Agibot G2 left arm to the collector home pose.")
    parser.add_argument("--collector-path", default="/data/pi05_test/g2_data_collector_v2.py")
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--retry-wait-s", type=float, default=1.0)
    return parser.parse_args()


def _load_collector(collector_path: str):
    spec = importlib.util.spec_from_file_location("g2_collector", collector_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = _parse_args()
    collector = _load_collector(args.collector_path)
    agibot_gdk.gdk_init()
    robot = agibot_gdk.Robot()
    tf = agibot_gdk.TF()

    last_err = None
    ok = False
    for attempt in range(max(1, int(args.retry_count))):
        try:
            ok = bool(
                collector.move_to_home(
                    robot,
                    tf,
                    collector.DEFAULT_HOME_POSE,
                    collector.DEFAULT_HOME_GRIPPER,
                )
            )
            last_err = None
            break
        except RuntimeError as exc:
            last_err = exc
            time.sleep(max(0.0, float(args.retry_wait_s)))

    if last_err is not None and not ok:
        raise last_err

    print({"move_to_home_ok": ok})


if __name__ == "__main__":
    main()
