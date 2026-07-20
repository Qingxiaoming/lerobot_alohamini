#!/usr/bin/env python3

"""Move calibrated AlohaMini follower arms smoothly to their center pose."""

from __future__ import annotations

import argparse
import time

from lerobot.robots.alohamini.config_lekiwi import LeKiwiConfig
from lerobot.robots.alohamini.lekiwi import LeKiwi


def _center_target(motor_names: list[str]) -> dict[str, float]:
    """Return center targets in the motors' normalized calibration coordinates."""
    return {name: 50.0 if name.endswith("_gripper") else 0.0 for name in motor_names}


def _smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _format_positions(positions: dict[str, float]) -> str:
    return "\n".join(f"  {name}: {value:.2f}" for name, value in positions.items())


def move_to_center(
    robot: LeKiwi,
    arm: str,
    duration_s: float,
    fps: float,
    current_limit_ma: float,
) -> None:
    selected: list[tuple[object, list[str]]] = []
    if arm in ("left", "both"):
        selected.append((robot.left_bus, robot.left_arm_motors))
    if arm in ("right", "both"):
        if robot.right_bus is None:
            raise RuntimeError("The right arm is not configured.")
        selected.append((robot.right_bus, robot.right_arm_motors))

    starts = [bus.sync_read("Present_Position", names) for bus, names in selected]
    targets = [_center_target(names) for _, names in selected]

    print("Current calibrated positions:")
    for positions in starts:
        print(_format_positions(positions))
    print("Center targets:")
    for positions in targets:
        print(_format_positions(positions))

    for bus, names in selected:
        bus.enable_torque(names)

    steps = max(1, round(duration_s * fps))
    period_s = 1.0 / fps
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        blend = _smoothstep(step / steps)
        for (bus, _), start, target in zip(selected, starts, targets, strict=True):
            goal = {name: start[name] + (target[name] - start[name]) * blend for name in target}
            bus.sync_write("Goal_Position", goal)

        robot.read_and_check_currents(limit_ma=current_limit_ma, print_currents=False)
        time.sleep(max(period_s - (time.perf_counter() - loop_start), 0.0))

    print("Follower arm movement completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoothly move calibrated AlohaMini follower arms to their center pose."
    )
    parser.add_argument(
        "--robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left_port", default="/dev/am_arm_follower_left")
    parser.add_argument("--right_port", default="/dev/am_arm_follower_right")
    parser.add_argument("--robot_id", default="AlohaMiniRobot")
    parser.add_argument("--duration", type=float, default=5.0, help="Movement duration in seconds.")
    parser.add_argument("--fps", type=float, default=30.0, help="Command update frequency.")
    parser.add_argument("--current_limit_ma", type=float, default=2000.0)
    parser.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    if args.current_limit_ma <= 0:
        parser.error("--current_limit_ma must be greater than zero")

    config = LeKiwiConfig(
        id=args.robot_id,
        robot_model=args.robot_model,
        left_port=args.left_port,
        right_port=args.right_port,
        cameras={},
    )
    robot = LeKiwi(config)
    if not robot.calibration_fpath.is_file():
        parser.error(
            f"Calibration file not found: {robot.calibration_fpath}. "
            "Run lekiwi_host calibration first and use the same --robot_id."
        )

    try:
        robot.connect(calibrate=False, home_lift=False)
        if not args.yes:
            answer = input(
                "Keep people and objects clear of the arms. Move to calibrated center? [y/N] "
            ).strip().lower()
            if answer not in {"y", "yes"}:
                print("Cancelled; no movement command was sent.")
                return
        move_to_center(robot, args.arm, args.duration, args.fps, args.current_limit_ma)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping and disconnecting.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
