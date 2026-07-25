#!/usr/bin/env python3

"""Mirror physical AlohaMini arm observations into the Genie Sim bridge."""

from __future__ import annotations

import argparse
import json
import time

import zmq

ARM_FIELDS = (
    "arm_left_shoulder_pan.pos",
    "arm_left_shoulder_lift.pos",
    "arm_left_elbow_flex.pos",
    "arm_left_wrist_flex.pos",
    "arm_left_wrist_yaw.pos",
    "arm_left_wrist_roll.pos",
    "arm_left_gripper.pos",
    "arm_right_shoulder_pan.pos",
    "arm_right_shoulder_lift.pos",
    "arm_right_elbow_flex.pos",
    "arm_right_wrist_flex.pos",
    "arm_right_wrist_yaw.pos",
    "arm_right_wrist_roll.pos",
    "arm_right_gripper.pos",
)
BASE_LIFT_FIELDS = ("x.vel", "y.vel", "theta.vel", "lift_axis.height_mm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-ip", required=True)
    parser.add_argument("--physical-observation-port", type=int, default=5556)
    parser.add_argument("--sim-ip", default="127.0.0.1")
    parser.add_argument("--sim-command-port", type=int, default=5555)
    parser.add_argument(
        "--include-base-lift",
        action="store_true",
        help="Also mirror physical base velocity and lift height. Disabled by default.",
    )
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    context = zmq.Context()
    physical = context.socket(zmq.PULL)
    physical.setsockopt(zmq.CONFLATE, 1)
    physical.connect(f"tcp://{args.physical_ip}:{args.physical_observation_port}")
    simulated = context.socket(zmq.PUSH)
    simulated.setsockopt(zmq.CONFLATE, 1)
    simulated.connect(f"tcp://{args.sim_ip}:{args.sim_command_port}")

    poller = zmq.Poller()
    poller.register(physical, zmq.POLLIN)
    fields = ARM_FIELDS + (BASE_LIFT_FIELDS if args.include_base_lift else ())
    frames = 0
    started = time.monotonic()
    last_report = started
    try:
        while True:
            if not dict(poller.poll(round(args.timeout_s * 1000))).get(physical):
                raise TimeoutError(
                    f"No observation received from {args.physical_ip}:"
                    f"{args.physical_observation_port} for {args.timeout_s:g}s"
                )
            observation = json.loads(physical.recv_string())
            missing = [name for name in fields if name not in observation]
            if missing:
                raise KeyError(f"Physical observation is missing fields: {missing}")
            simulated.send_string(json.dumps({name: observation[name] for name in fields}))
            frames += 1
            now = time.monotonic()
            if now - last_report >= 2.0:
                hz = frames / (now - started)
                print(f"Mirroring {len(fields)} fields at {hz:.1f} Hz", flush=True)
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        physical.close(linger=0)
        simulated.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
