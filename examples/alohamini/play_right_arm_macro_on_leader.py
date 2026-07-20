#!/usr/bin/env python3

import argparse
import contextlib
import time
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.scripts.lerobot_record import (
    _macro_entry_to_action,
    follower_to_leader_feedback,
    load_action_macro,
)
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.robot_utils import precise_sleep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a recorded right-arm macro on the leader arm only.")
    parser.add_argument("--right_arm_macro", type=Path, required=True, help="Macro JSON to play.")
    parser.add_argument("--fps", type=int, default=15, help="Playback control frequency.")
    parser.add_argument(
        "--speed_scale",
        type=float,
        default=0.25,
        help="Scale the macro displacement around the start pose. Use 1.0 for the recorded amplitude.",
    )
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID.")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="am-leader-6dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
        help="Leader arm profile selector.",
    )
    parser.add_argument("--left_port", type=str, default="/dev/am_arm_leader_left")
    parser.add_argument("--right_port", type=str, default="/dev/am_arm_leader_right")
    return parser.parse_args()


def wait_for_start(events: dict) -> bool:
    events["start_recording"] = False
    events["stop_recording"] = False
    events["exit_early"] = False
    print("Move the right leader arm to the macro start pose, release it, then press 1 to play.")
    while not events["start_recording"]:
        if events["stop_recording"]:
            return False
        time.sleep(0.05)
    events["start_recording"] = False
    return True


def scale_macro_frame(
    action: dict[str, float], base_action: dict[str, float], speed_scale: float
) -> dict[str, float]:
    return {
        key: base_action.get(key, value) + (value - base_action.get(key, value)) * speed_scale
        for key, value in action.items()
    }


def main() -> None:
    args = parse_args()
    macro = load_action_macro(args.right_arm_macro)
    if macro is None:
        raise RuntimeError("No macro was loaded.")

    leader = BiSOLeader(
        BiSOLeaderConfig(
            left_arm_config=SOLeaderConfig(port=args.left_port, arm_profile=args.arm_profile),
            right_arm_config=SOLeaderConfig(port=args.right_port, arm_profile=args.arm_profile),
            id=args.leader_id,
        )
    )

    listener, events = init_keyboard_listener()
    if listener is None:
        raise RuntimeError("Macro playback requires keyboard listening.")

    try:
        leader.connect()
        leader.disable_torque()

        if not wait_for_start(events):
            print("Playback aborted before start.")
            return

        base_action = {f"arm_{key}": value for key, value in leader.get_action().items()}
        print(
            f"Playing {len(macro)} frames from {args.right_arm_macro} at {args.fps} Hz "
            f"with speed_scale={args.speed_scale}."
        )
        print("Press ESC to abort.")

        leader.right_arm.enable_torque()
        control_interval = 1.0 / args.fps
        for index, frame in enumerate(macro):
            if events["stop_recording"]:
                print("Playback aborted.")
                break

            start_t = time.perf_counter()
            action = _macro_entry_to_action(frame, base_action)
            action = scale_macro_frame(action, base_action, args.speed_scale)
            feedback = follower_to_leader_feedback(action, side="right")
            if feedback:
                leader.send_feedback(feedback)

            if index % max(args.fps, 1) == 0:
                print(f"Frame {index}/{len(macro)}")

            precise_sleep(max(control_interval - (time.perf_counter() - start_t), 0.0))

    finally:
        if leader.is_connected:
            with contextlib.suppress(Exception):
                leader.right_arm.disable_torque()
            leader.disconnect()
        if listener is not None:
            listener.stop()


if __name__ == "__main__":
    main()
