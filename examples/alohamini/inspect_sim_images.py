#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

from PIL import Image

from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save one RGB frame per simulator camera using LeRobot's dataset color semantics."
    )
    parser.add_argument("--remote_ip", default="127.0.0.1")
    parser.add_argument("--robot_id", default="sim_image_inspector")
    parser.add_argument("--output", type=Path, default=Path("/tmp/alohamini_sim_images"))
    args = parser.parse_args()

    robot = LeKiwiClient(
        LeKiwiClientConfig(
            remote_ip=args.remote_ip,
            id=args.robot_id,
            robot_model="alohamini2pro",
        )
    )
    robot.connect()
    try:
        observation = robot.get_observation()
    finally:
        robot.disconnect()

    args.output.mkdir(parents=True, exist_ok=True)
    result = {"image_metadata": robot.last_image_metadata, "frames": {}}
    for name in robot.config.cameras:
        frame = observation[name]
        output = args.output / f"{name}.png"
        Image.fromarray(frame).save(output)
        result["frames"][name] = {
            "path": str(output),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "min": int(frame.min()),
            "max": int(frame.max()),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
