#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a right-arm action macro from the bimanual leader.")
    parser.add_argument("--output", type=Path, required=True, help="Path to write the macro JSON file.")
    parser.add_argument("--fps", type=int, default=30, help="Macro recording frequency.")
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID.")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
        help="Leader arm profile selector.",
    )
    parser.add_argument("--left_port", type=str, default="/dev/am_arm_leader_left")
    parser.add_argument("--right_port", type=str, default="/dev/am_arm_leader_right")
    parser.add_argument(
        "--mode",
        choices=["delta", "action"],
        default="delta",
        help="Save positions relative to the pose when 1 is pressed, or absolute positions.",
    )
    return parser.parse_args()


def right_arm_action(action: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in action.items() if key.startswith("right_") and key.endswith(".pos")}


def wait_for_start(events: dict) -> bool:
    events["start_recording"] = False
    events["stop_current_episode"] = False
    events["exit_early"] = False
    print("Move the right leader arm to the macro start pose, then press 1 to start recording.")
    while not events["start_recording"]:
        if events["stop_recording"]:
            return False
        time.sleep(0.05)
    events["start_recording"] = False
    return True


def main() -> None:
    args = parse_args()

    leader = BiSOLeader(
        BiSOLeaderConfig(
            left_arm_config=SOLeaderConfig(port=args.left_port, arm_profile=args.arm_profile),
            right_arm_config=SOLeaderConfig(port=args.right_port, arm_profile=args.arm_profile),
            id=args.leader_id,
        )
    )

    listener, events = init_keyboard_listener()
    if listener is None:
        raise RuntimeError("Macro recording requires keyboard listening.")

    frames: list[dict[str, dict[str, float]]] = []
    try:
        leader.connect()
        leader.disable_torque()

        if not wait_for_start(events):
            print("Macro recording aborted before start.")
            return

        base = right_arm_action(leader.get_action())
        if not base:
            raise RuntimeError("No right-arm leader action keys were read.")

        print("Recording macro. Move the right leader arm through the primitive, then press 2 to stop.")
        control_interval = 1.0 / args.fps
        while not events["stop_current_episode"]:
            if events["stop_recording"]:
                print("Macro recording aborted; not saving.")
                return

            start_t = time.perf_counter()
            current = right_arm_action(leader.get_action())
            if args.mode == "delta":
                values = {key: current[key] - base[key] for key in base}
                frames.append({"delta": values})
            else:
                frames.append({"action": current})

            precise_sleep(max(control_interval - (time.perf_counter() - start_t), 0.0))

        events["stop_current_episode"] = False
        events["exit_early"] = False

    finally:
        if leader.is_connected:
            leader.disconnect()
        if listener is not None:
            listener.stop()

    if not frames:
        raise RuntimeError("No macro frames were recorded.")

    payload = {
        "metadata": {
            "fps": args.fps,
            "mode": args.mode,
            "side": "right",
            "arm_profile": args.arm_profile,
            "num_frames": len(frames),
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(frames)} macro frames to {args.output}")


if __name__ == "__main__":
    main()
