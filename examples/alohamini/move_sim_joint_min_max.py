#!/usr/bin/env python3

"""Move one simulated AlohaMini joint to min, then max, through the ZMQ bridge."""

from __future__ import annotations

import argparse
import json
import math
import time
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


def smoothstep(progress: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return progress * progress * (3.0 - 2.0 * progress)


def move_joint(
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    joint: str,
    target: float,
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> None:
    start = float(action[joint])
    steps = max(1, round(duration_s * rate_hz))
    period_s = 1.0 / rate_hz
    for step in range(1, steps + 1):
        started = time.monotonic()
        alpha = smoothstep(step / steps)
        action[joint] = start + (target - start) * alpha
        command_socket.send_string(json.dumps(action))
        receive_latest(observation_socket, timeout_ms)
        time.sleep(max(0.0, period_s - (time.monotonic() - started)))
    action[joint] = target


def hold(
    command_socket: zmq.Socket,
    observation_socket: zmq.Socket,
    action: dict[str, float],
    duration_s: float,
    rate_hz: float,
    timeout_ms: int,
) -> None:
    deadline = time.monotonic() + duration_s
    period_s = 1.0 / rate_hz
    while time.monotonic() < deadline:
        command_socket.send_string(json.dumps(action))
        receive_latest(observation_socket, timeout_ms)
        time.sleep(period_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("joint", choices=ARM_JOINTS)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5555)
    parser.add_argument("--observation-port", type=int, default=5556)
    parser.add_argument("--move-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    if min(args.move_seconds, args.rate, args.timeout) <= 0 or args.hold_seconds < 0:
        parser.error("move-seconds, rate and timeout must be positive; hold-seconds cannot be negative")

    low = 0.0 if args.joint in GRIPPERS else -100.0
    high = 100.0
    if not args.yes:
        print("This moves the simulated robot only.")
        print(f"{args.joint}: current -> {low:g} -> {high:g}; other joints hold their current values.")
        answer = input("Type 'sim' to continue: ")
        raise SystemExit("Cancelled.")

    timeout_ms = math.ceil(args.timeout * 1000)
    context = zmq.Context()
    command_socket = context.socket(zmq.PUSH)
    command_socket.setsockopt(zmq.CONFLATE, 1)
    command_socket.connect(f"tcp://{args.sim_ip}:{args.command_port}")
    observation_socket = context.socket(zmq.PULL)
    observation_socket.setsockopt(zmq.CONFLATE, 1)
    observation_socket.connect(f"tcp://{args.sim_ip}:{args.observation_port}")

    try:
        observation = receive_latest(observation_socket, timeout_ms)
        required = (*ARM_JOINTS, LIFT_FIELD)
        missing = [field for field in required if field not in observation]
        if missing:
            raise KeyError(f"Observation is missing fields: {missing}")
        action = {field: float(observation[field]) for field in required}
        action.update(dict.fromkeys(BASE_FIELDS, 0.0))

        print(f"{args.joint}: moving to min {low:g}", flush=True)
        move_joint(
            command_socket,
            observation_socket,
            action,
            args.joint,
            low,
            args.move_seconds,
            args.rate,
            timeout_ms,
        )
        print(f"{args.joint}: holding min for {args.hold_seconds:g}s", flush=True)
        hold(
            command_socket,
            observation_socket,
            action,
            args.hold_seconds,
            args.rate,
            timeout_ms,
        )

        print(f"{args.joint}: moving to max {high:g}", flush=True)
        move_joint(
            command_socket,
            observation_socket,
            action,
            args.joint,
            high,
            args.move_seconds,
            args.rate,
            timeout_ms,
        )
        print(f"{args.joint}: holding max for {args.hold_seconds:g}s", flush=True)
        hold(
            command_socket,
            observation_socket,
            action,
            args.hold_seconds,
            args.rate,
            timeout_ms,
        )
        command_socket.send_string(json.dumps(action))
        print(f"DONE: {args.joint} stays at max {high:g}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted; leaving the simulation at its last commanded pose.", flush=True)
    finally:
        command_socket.close(linger=0)
        observation_socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
