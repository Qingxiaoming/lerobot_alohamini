#!/usr/bin/env python3

"""Save and restore a calibrated AlohaMini follower-arm natural pose."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from lerobot.robots.alohamini.config_lekiwi import LeKiwiConfig
from lerobot.robots.alohamini.lekiwi import LeKiwi


def _smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def _selected_buses(robot: LeKiwi, arm: str) -> list[tuple[object, list[str]]]:
    selected: list[tuple[object, list[str]]] = []
    if arm in ("left", "both"):
        selected.append((robot.left_bus, robot.left_arm_motors))
    if arm in ("right", "both"):
        if robot.right_bus is None:
            raise RuntimeError("The right arm is not configured.")
        selected.append((robot.right_bus, robot.right_arm_motors))
    return selected


def _read_positions(robot: LeKiwi, arm: str) -> dict[str, float]:
    positions: dict[str, float] = {}
    for bus, names in _selected_buses(robot, arm):
        positions.update(bus.sync_read("Present_Position", names))
    return positions


def _format_positions(positions: dict[str, float]) -> str:
    return "\n".join(f"  {name}: {value:.2f}" for name, value in positions.items())


def save_natural_pose(robot: LeKiwi, arm: str, pose_path: Path) -> None:
    positions = _read_positions(robot, arm)
    payload = {
        "robot_model": robot.config.robot_model,
        "robot_id": robot.id,
        "positions": positions,
    }
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Natural pose saved to {pose_path}:")
    print(_format_positions(positions))


def load_natural_pose(robot: LeKiwi, arm: str, pose_path: Path) -> dict[str, float]:
    if not pose_path.is_file():
        raise FileNotFoundError(f"Natural pose file not found: {pose_path}. Run the 'save' command first.")
    payload = json.loads(pose_path.read_text())
    if payload.get("robot_model") != robot.config.robot_model:
        raise ValueError(
            f"Natural pose is for {payload.get('robot_model')!r}, not {robot.config.robot_model!r}."
        )

    saved = payload.get("positions")
    if not isinstance(saved, dict):
        raise ValueError(f"Invalid natural pose file: {pose_path}")
    required = {name for _, names in _selected_buses(robot, arm) for name in names}
    missing = required - saved.keys()
    if missing:
        raise ValueError(f"Natural pose does not contain these motors: {sorted(missing)}")
    return {name: float(saved[name]) for name in required}


def move_to_natural_pose(
    robot: LeKiwi,
    arm: str,
    targets: dict[str, float],
    duration_s: float,
    fps: float,
    current_limit_ma: float,
) -> None:
    selected = _selected_buses(robot, arm)
    starts = _read_positions(robot, arm)

    for bus, names in selected:
        bus.enable_torque(names)

    steps = max(1, round(duration_s * fps))
    period_s = 1.0 / fps
    for step in range(1, steps + 1):
        loop_start = time.perf_counter()
        blend = _smoothstep(step / steps)
        for bus, names in selected:
            goal = {name: starts[name] + (targets[name] - starts[name]) * blend for name in names}
            bus.sync_write("Goal_Position", goal)
        robot.read_and_check_currents(limit_ma=current_limit_ma, print_currents=False)
        time.sleep(max(period_s - (time.perf_counter() - loop_start), 0.0))

    print("Natural-pose movement completed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save or restore an AlohaMini natural arm pose.")
    parser.add_argument("command", choices=("save", "move"))
    parser.add_argument(
        "--robot_model",
        required=True,
        choices=("alohamini1", "alohamini2", "alohamini2pro"),
    )
    parser.add_argument("--arm", choices=("left", "right", "both"), default="both")
    parser.add_argument("--left_port", default="/dev/am_arm_follower_left")
    parser.add_argument("--right_port", default="/dev/am_arm_follower_right")
    parser.add_argument("--robot_id", default="AlohaMiniRobot")
    parser.add_argument("--pose_file", type=Path)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--current_limit_ma", type=float, default=2000.0)
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation.")
    args = parser.parse_args()

    if args.duration <= 0 or args.fps <= 0 or args.current_limit_ma <= 0:
        parser.error("--duration, --fps, and --current_limit_ma must be greater than zero")

    robot = LeKiwi(
        LeKiwiConfig(
            id=args.robot_id,
            robot_model=args.robot_model,
            left_port=args.left_port,
            right_port=args.right_port,
            cameras={},
        )
    )
    if not robot.calibration_fpath.is_file():
        parser.error(
            f"Calibration file not found: {robot.calibration_fpath}. "
            "Run lekiwi_host calibration first and use the same --robot_id."
        )
    pose_path = args.pose_file or robot.calibration_fpath.with_suffix(".natural_pose.json")

    try:
        robot.connect(calibrate=False, home_lift=False)
        if args.command == "save":
            print("Arm torque is disabled. Manually place the selected arm(s) in the natural pose.")
            if not args.yes:
                answer = input("Save the current calibrated positions as the natural pose? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Cancelled; the natural pose was not changed.")
                    return
            save_natural_pose(robot, args.arm, pose_path)
        else:
            targets = load_natural_pose(robot, args.arm, pose_path)
            print("Current positions:")
            print(_format_positions(_read_positions(robot, args.arm)))
            print("Saved natural pose:")
            print(_format_positions(targets))
            if not args.yes:
                answer = input("Keep the arms clear. Move to the saved natural pose? [y/N] ").strip().lower()
                if answer not in {"y", "yes"}:
                    print("Cancelled; no movement command was sent.")
                    return
            move_to_natural_pose(robot, args.arm, targets, args.duration, args.fps, args.current_limit_ma)
    except KeyboardInterrupt:
        print("\nInterrupted; stopping and disconnecting.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    main()
