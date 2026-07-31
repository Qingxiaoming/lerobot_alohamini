#!/usr/bin/env python3

import argparse
import json
import time

from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read simulator expert actions from the AlohaMini ZMQ bridge without commanding the robot."
    )
    parser.add_argument("--remote_ip", default="127.0.0.1")
    parser.add_argument("--robot_id", default="sim_action_inspector")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--poll_rate", type=float, default=30.0)
    args = parser.parse_args()
    if min(args.duration, args.poll_rate) <= 0:
        parser.error("--duration and --poll_rate must be positive")

    robot = LeKiwiClient(
        LeKiwiClientConfig(
            remote_ip=args.remote_ip,
            id=args.robot_id,
            robot_model="alohamini2pro",
        )
    )
    robot.connect()
    deadline = time.monotonic() + args.duration
    last_sequence = None
    try:
        while time.monotonic() < deadline:
            action, metadata = robot.get_record_action()
            sequence = metadata["action_sequence"]
            if sequence != last_sequence:
                print(
                    json.dumps(
                        {
                            **metadata,
                            "action": {name: float(action[name]) for name in robot.action_features},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_sequence = sequence
            time.sleep(1.0 / args.poll_rate)
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
