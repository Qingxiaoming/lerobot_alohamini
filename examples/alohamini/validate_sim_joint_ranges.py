#!/usr/bin/env python3

"""Validate AlohaMini joint mappings through the Genie Sim ZMQ bridge."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import zmq

REGULAR_JOINTS = (
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_yaw.pos",
    "arm_left_wrist_roll.pos",
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_yaw.pos",
    "arm_right_wrist_roll.pos",
)
GRIPPERS = ("arm_left_gripper.pos", "arm_right_gripper.pos")
ARM_JOINTS = REGULAR_JOINTS + GRIPPERS
BASE_FIELDS = ("x.vel", "y.vel", "theta.vel")
LIFT_FIELD = "lift_axis.height_mm"
DEFAULT_SAFE_RANGE_MIDPOINT = {joint: 50.0 if joint in GRIPPERS else 0.0 for joint in ARM_JOINTS}


@dataclass
class Checkpoint:
    joint: str
    target: float
    observed: float
    absolute_error: float
    passed: bool


def receive_latest(socket: zmq.Socket, timeout_ms: int) -> dict[str, Any]:
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    if not dict(poller.poll(timeout_ms)).get(socket):
        raise TimeoutError(f"No Genie Sim observation received for {timeout_ms / 1000:g}s")
    latest = json.loads(socket.recv_string())
    while True:
        try:
            latest = json.loads(socket.recv_string(flags=zmq.NOBLOCK))
        except zmq.Again:
            return latest


def action_from_observation(observation: dict[str, Any]) -> dict[str, float]:
    required = (*ARM_JOINTS, LIFT_FIELD)
    missing = [field for field in required if field not in observation]
    if missing:
        raise KeyError(f"Observation is missing fields: {missing}")
    action = {field: float(observation[field]) for field in required}
    action.update(dict.fromkeys(BASE_FIELDS, 0.0))
    return action


def load_baseline_pose(path: Path | None) -> dict[str, float]:
    if path is None:
        return dict(DEFAULT_SAFE_RANGE_MIDPOINT)
    payload = json.loads(path.read_text())
    positions = payload.get("positions", payload)
    if not isinstance(positions, dict):
        raise ValueError(f"Invalid baseline pose file: {path}")
    baseline = {
        f"{name}.pos" if not name.endswith(".pos") else name: float(value)
        for name, value in positions.items()
    }
    missing = sorted(set(ARM_JOINTS) - set(baseline))
    if missing:
        raise ValueError(f"Baseline pose is missing joints: {missing}")
    return {joint: baseline[joint] for joint in ARM_JOINTS}


def smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def move_joint(
    *,
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    joint: str,
    target: float,
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> dict[str, Any]:
    start = float(action[joint])
    steps = max(1, round(duration_s * rate_hz))
    period_s = 1.0 / rate_hz
    latest: dict[str, Any] = {}
    for step in range(1, steps + 1):
        loop_started = time.monotonic()
        alpha = smoothstep(step / steps)
        action[joint] = start + (target - start) * alpha
        command_socket.send_string(json.dumps(action))
        latest = receive_latest(observation_socket, timeout_ms)
        time.sleep(max(period_s - (time.monotonic() - loop_started), 0.0))
    action[joint] = target
    return latest


def move_pose(
    *,
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    targets: dict[str, float],
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> dict[str, Any]:
    starts = {joint: float(action[joint]) for joint in targets}
    steps = max(1, round(duration_s * rate_hz))
    period_s = 1.0 / rate_hz
    latest: dict[str, Any] = {}
    for step in range(1, steps + 1):
        loop_started = time.monotonic()
        alpha = smoothstep(step / steps)
        for joint, target in targets.items():
            action[joint] = starts[joint] + (target - starts[joint]) * alpha
        command_socket.send_string(json.dumps(action))
        latest = receive_latest(observation_socket, timeout_ms)
        time.sleep(max(period_s - (time.monotonic() - loop_started), 0.0))
    action.update(targets)
    return latest


def hold_and_measure(
    *,
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    joint: str,
    hold_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> float:
    deadline = time.monotonic() + hold_s
    samples: list[float] = []
    period_s = 1.0 / rate_hz
    while time.monotonic() < deadline:
        command_socket.send_string(json.dumps(action))
        observation = receive_latest(observation_socket, timeout_ms)
        samples.append(float(observation[joint]))
        time.sleep(period_s)
    if not samples:
        observation = receive_latest(observation_socket, timeout_ms)
        samples.append(float(observation[joint]))
    tail = samples[-min(5, len(samples)) :]
    return sum(tail) / len(tail)


def write_report(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    checkpoints: list[Checkpoint],
    interrupted: bool,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"alohamini_sim_joint_validation_{stamp}.json"
    md_path = output_dir / f"alohamini_sim_joint_validation_{stamp}.md"
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "sim_ip": args.sim_ip,
        "duration_s": args.duration,
        "hold_s": args.hold,
        "rate_hz": args.rate,
        "tolerance": args.tolerance,
        "amplitude": args.amplitude,
        "full_range": args.full_range,
        "baseline_pose": args.baseline_pose.as_posix()
        if args.baseline_pose
        else "calibrated safe-range midpoint (0 for arm joints, 50 for grippers)",
        "interrupted": interrupted,
        "checkpoints": [asdict(item) for item in checkpoints],
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    passed = sum(item.passed for item in checkpoints)
    lines = [
        "# AlohaMini Genie Sim joint-range validation",
        "",
        f"- Created: {payload['created_at']}",
        f"- Result: {passed}/{len(checkpoints)} checkpoints passed",
        f"- Interrupted: {interrupted}",
        f"- Numeric tolerance: {args.tolerance:g} normalized units",
        "",
        "| Joint | Target | Observed | Abs error | Pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    lines.extend(
        f"| `{item.joint}` | {item.target:.3f} | {item.observed:.3f} | "
        f"{item.absolute_error:.3f} | {'yes' if item.passed else 'no'} |"
        for item in checkpoints
    )
    lines.extend(
        [
            "",
            "## Manual visual checks",
            "",
            "- [ ] Joint direction matches the intended robot convention.",
            "- [ ] No visual mesh penetration across the tested range.",
            "- [ ] No unexpected self-collision or discontinuous jump.",
            "- [ ] Gripper 0/50/100 correspond to closed/middle/open.",
            "",
            "Numeric pass only means commanded and observed normalized values agree.",
            "It does not prove collision, dynamics, or real-robot safety.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5555)
    parser.add_argument("--observation-port", type=int, default=5556)
    parser.add_argument("--joint", action="append", choices=ARM_JOINTS)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=0.75)
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=3.0)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=20.0,
        help="Normalized displacement on each side of the calibrated safe-range midpoint.",
    )
    parser.add_argument(
        "--full-range",
        action="store_true",
        help="Test calibrated endpoints instead of the default small bidirectional displacement.",
    )
    parser.add_argument(
        "--baseline-pose",
        type=Path,
        help="Optional pose JSON override. Defaults to the calibrated safe-range midpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sim_joint_validation"))
    parser.add_argument("--yes", action="store_true", help="Run without the interactive confirmation.")
    args = parser.parse_args()

    if args.duration <= 0 or args.hold < 0 or args.rate <= 0 or args.timeout <= 0:
        parser.error("duration, rate, and timeout must be positive; hold must be non-negative")
    if args.tolerance < 0:
        parser.error("tolerance must be non-negative")
    if args.amplitude <= 0:
        parser.error("amplitude must be positive")

    joints = tuple(args.joint) if args.joint else ARM_JOINTS
    baseline = load_baseline_pose(args.baseline_pose)
    if not args.yes:
        print("This moves the simulated robot only.")
        print(f"Joints: {', '.join(joints)}")
        print("All untested joints remain at their calibrated safe-range midpoint.")
        answer = input("Type 'sim' to continue: ")
        if answer.strip().lower() != "sim":
            raise SystemExit("Cancelled.")

    context = zmq.Context()
    command_socket = context.socket(zmq.PUSH)
    command_socket.setsockopt(zmq.CONFLATE, 1)
    command_socket.connect(f"tcp://{args.sim_ip}:{args.command_port}")
    observation_socket = context.socket(zmq.PULL)
    observation_socket.setsockopt(zmq.CONFLATE, 1)
    observation_socket.connect(f"tcp://{args.sim_ip}:{args.observation_port}")

    checkpoints: list[Checkpoint] = []
    interrupted = False
    action: dict[str, float] | None = None
    timeout_ms = math.ceil(args.timeout * 1000)
    try:
        observation = receive_latest(observation_socket, timeout_ms)
        action = action_from_observation(observation)
        print("Moving all arm joints to their calibrated safe-range midpoint.", flush=True)
        move_pose(
            command_socket=command_socket,
            observation_socket=observation_socket,
            action=action,
            targets=baseline,
            duration_s=args.duration,
            rate_hz=args.rate,
            timeout_ms=timeout_ms,
        )
        for joint in joints:
            rest = baseline[joint]
            low_limit = 0.0 if joint in GRIPPERS else -100.0
            high_limit = 100.0
            low = low_limit if args.full_range else max(low_limit, rest - args.amplitude)
            high = high_limit if args.full_range else min(high_limit, rest + args.amplitude)
            targets = (rest, low, rest, high, rest)
            print(f"\n[{joint}]")
            for target in targets:
                print(f"  moving to {target:.1f}", flush=True)
                move_joint(
                    command_socket=command_socket,
                    observation_socket=observation_socket,
                    action=action,
                    joint=joint,
                    target=target,
                    duration_s=args.duration,
                    rate_hz=args.rate,
                    timeout_ms=timeout_ms,
                )
                observed = hold_and_measure(
                    command_socket=command_socket,
                    observation_socket=observation_socket,
                    action=action,
                    joint=joint,
                    hold_s=args.hold,
                    rate_hz=args.rate,
                    timeout_ms=timeout_ms,
                )
                error = abs(observed - target)
                checkpoint = Checkpoint(
                    joint=joint,
                    target=target,
                    observed=observed,
                    absolute_error=error,
                    passed=error <= args.tolerance,
                )
                checkpoints.append(checkpoint)
                print(
                    f"  observed={observed:.2f}, error={error:.2f}, "
                    f"{'PASS' if checkpoint.passed else 'FAIL'}",
                    flush=True,
                )
            action.update(baseline)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted. Holding the last commanded simulated pose.", flush=True)
        if action is not None:
            command_socket.send_string(json.dumps(action))
    finally:
        command_socket.close(linger=0)
        observation_socket.close(linger=0)
        context.term()
        json_path, md_path = write_report(
            args.output_dir,
            args=args,
            checkpoints=checkpoints,
            interrupted=interrupted,
        )
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
