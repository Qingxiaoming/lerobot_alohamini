#!/usr/bin/env python3

import argparse
import json
import time
from pathlib import Path

from lerobot.common.control_utils import init_keyboard_listener
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import load_action_macro, record_loop
from lerobot.teleoperators.bi_so_leader import BiSOLeader, BiSOLeaderConfig
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SOLeaderConfig
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def wait_for_segment_start(events: dict, label: str) -> bool:
    events["start_recording"] = False
    events["stop_current_episode"] = False
    events["exit_early"] = False
    print(f"Press 1 to start {label}. Press ESC to stop recording.")

    while not events["start_recording"]:
        if events["stop_recording"]:
            return False
        time.sleep(0.05)

    events["start_recording"] = False
    return True


def right_arm_action(action: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in action.items() if key.startswith("right_") and key.endswith(".pos")}


def leader_action_to_robot_action(leader_action: dict[str, float], obs: dict | None) -> dict[str, float]:
    action = {f"arm_{key}": value for key, value in leader_action.items() if key.endswith(".pos")}
    lift_height = 0.0 if obs is None else float(obs.get("lift_axis.height_mm", 0.0))
    action.update(
        {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "lift_axis.height_mm": lift_height,
        }
    )
    return action


def record_right_arm_macro(
    args: argparse.Namespace,
    leader_arm_config: BiSOLeaderConfig,
    robot_config: LeKiwiClientConfig,
) -> None:
    listener, events = init_keyboard_listener()
    if listener is None:
        raise RuntimeError("Macro recording requires keyboard listening.")

    leader_arm = BiSOLeader(leader_arm_config)
    robot = None if args.no_record_macro_robot_follow else LeKiwiClient(robot_config)
    frames: list[dict[str, dict[str, float]]] = []

    try:
        leader_arm.connect()
        leader_arm.disable_torque()
        if robot is not None:
            robot.connect()
            init_rerun(session_name="alohamini_macro_record")
            print("Follower tracking is enabled while recording the macro.")
        else:
            print("Follower tracking is disabled; recording leader motion only.")

        if not wait_for_segment_start(events, "right-arm macro recording"):
            print("Macro recording aborted before start.")
            return

        base = right_arm_action(leader_arm.get_action())
        if not base:
            raise RuntimeError("No right-arm leader action keys were read.")

        print("Recording macro. Move the right leader arm through the primitive, then press 2 to stop.")
        control_interval = 1.0 / args.fps
        while not events["stop_current_episode"]:
            if events["stop_recording"]:
                print("Macro recording aborted; not saving.")
                return

            start_t = time.perf_counter()
            leader_action = leader_arm.get_action()
            current = right_arm_action(leader_action)
            if args.record_macro_mode == "delta":
                values = {key: current[key] - base[key] for key in base}
                frames.append({"delta": values})
            else:
                frames.append({"action": current})

            if robot is not None:
                obs = robot.get_observation()
                robot_action = leader_action_to_robot_action(leader_action, obs)
                robot.send_action(robot_action)
                log_rerun_data(observation=obs, action=robot_action)

            time.sleep(max(control_interval - (time.perf_counter() - start_t), 0.0))

    finally:
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if leader_arm.is_connected:
            leader_arm.disconnect()
        if listener is not None:
            listener.stop()

    if not frames:
        raise RuntimeError("No macro frames were recorded.")

    payload = {
        "metadata": {
            "fps": args.fps,
            "mode": args.record_macro_mode,
            "side": "right",
            "arm_profile": args.arm_profile,
            "num_frames": len(frames),
        },
        "frames": frames,
    }
    output = Path(args.record_right_arm_macro)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(frames)} macro frames to {output}")


def main():
    parser = argparse.ArgumentParser(description="Record episodes with bi-arm teleoperation")
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset repo_id, e.g. liyitenga/record_20250914225057"
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--episode_time",
        type=int,
        default=None,
        help="Duration of each episode in seconds. If omitted, press 1 to start and 2 to end.",
    )
    parser.add_argument(
        "--reset_time",
        type=int,
        default=None,
        help="Deprecated and ignored. Reset the scene manually before pressing 1 for the next episode.",
    )
    parser.add_argument(
        "--task_description", type=str, default="My task description4", help="Task description"
    )
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1", help="Robot host IP")
    parser.add_argument("--robot_id", type=str, default="lekiwi_host", help="Robot ID")
    parser.add_argument(
        "--robot_model",
        type=str,
        default="alohamini1",
        choices=["alohamini1", "alohamini2", "alohamini2pro"],
        help="AlohaMini model. Must match the --robot_model used on the Pi host side.",
    )
    parser.add_argument("--leader_id", type=str, default="so101_leader_bi", help="Leader arm device ID")
    parser.add_argument(
        "--arm_profile",
        type=str,
        default="so-arm-5dof",
        choices=["so-arm-5dof", "am-leader-6dof"],
        help="Leader arm profile selector.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume recording on existing dataset")
    parser.add_argument(
        "--record_right_arm_macro",
        type=str,
        default=None,
        help="Record a right-arm macro JSON from the leader, then exit.",
    )
    parser.add_argument(
        "--record_macro_mode",
        choices=["delta", "action"],
        default="delta",
        help="Save macro positions relative to the pose when 1 is pressed, or absolute positions.",
    )
    parser.add_argument(
        "--no_record_macro_robot_follow",
        action="store_true",
        help="Record a macro from the leader only, without connecting the follower robot.",
    )
    parser.add_argument(
        "--right_arm_macro",
        type=str,
        default=None,
        help="Optional JSON action macro for the right arm. Press the trigger key to replay it.",
    )
    parser.add_argument(
        "--macro_trigger_key",
        type=str,
        default="g",
        help="Keyboard key that triggers --right_arm_macro replay.",
    )
    parser.add_argument(
        "--push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to upload the dataset to Hugging Face Hub after recording. Use '--push_to_hub false' to skip upload.",
    )

    args = parser.parse_args()
    timed_mode = args.episode_time is not None
    if args.reset_time is not None:
        print("Warning: --reset_time is deprecated and ignored; there is no separate reset stage.")
    if args.dataset is None and args.record_right_arm_macro is None:
        parser.error("--dataset is required unless --record_right_arm_macro is used.")

    robot_config = LeKiwiClientConfig(
        remote_ip=args.remote_ip,
        id=args.robot_id,
        robot_model=args.robot_model,
    )

    # === Robot and teleop config ===
    leader_arm_config = BiSOLeaderConfig(
        left_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_left",
            arm_profile=args.arm_profile,
        ),
        right_arm_config=SOLeaderConfig(
            port="/dev/am_arm_leader_right",
            arm_profile=args.arm_profile,
        ),
        id=args.leader_id,
    )

    if args.record_right_arm_macro is not None:
        record_right_arm_macro(args, leader_arm_config, robot_config)
        return

    action_macro = load_action_macro(args.right_arm_macro)

    keyboard_config = KeyboardTeleopConfig()

    robot = LeKiwiClient(robot_config)
    leader_arm = BiSOLeader(leader_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset setup ===
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    if args.resume:
        print("Resuming existing dataset:", args.dataset)
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset,
            root=HF_LEROBOT_HOME / args.dataset,
            image_writer_threads=4,
        )
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.dataset,
            fps=args.fps,
            features=dataset_features,
            robot_type=robot.name,
            use_videos=True,
            image_writer_threads=4,
        )
        print(f"Dataset created with id: {dataset.repo_id}")

    print(f"Local dataset path: {dataset.root.resolve()}")

    # === Connect devices ===
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_record")
    if not timed_mode and listener is None:
        raise RuntimeError(
            "Manual recording mode requires keyboard listening. "
            "Use --episode_time in headless mode."
        )

    if not robot.is_connected or not leader_arm.is_connected or not keyboard.is_connected:
        raise ValueError("Robot or teleop is not connected!")

    print("Starting record loop...")
    if timed_mode:
        print(f"Timed mode: episode_time={args.episode_time}s")
    else:
        print("Manual mode: press 1 to start an episode and press 2 to save it.")
    if action_macro is not None:
        print(f"Right-arm macro enabled: press {args.macro_trigger_key} to replay {args.right_arm_macro}")
    recorded_episodes = 0

    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        events["current_episode"] = recorded_episodes + 1
        events["total_episodes"] = args.num_episodes
        log_say(f"Recording episode {recorded_episodes + 1} of {args.num_episodes}")
        if not timed_mode and not wait_for_segment_start(events, "recording"):
            break

        # === Main record loop ===
        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[leader_arm, keyboard],
            control_time_s=args.episode_time if timed_mode else None,
            single_task=args.task_description,
            display_data=True,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            action_macro=action_macro,
            macro_trigger_key=args.macro_trigger_key,
            macro_sync_leader_side="right",
        )

        if events["rerecord_episode"]:
            log_say("Discard episode and wait to re-record")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        # The recording loop can exit before its first iteration when the user
        # presses 2 or ESC immediately after starting. Do not attempt to save an
        # empty episode; DatasetWriter deliberately rejects zero-frame episodes.
        if not dataset.has_pending_frames():
            log_say("No frames were recorded; skipping the empty episode")
            dataset.clear_episode_buffer()
            if events["stop_recording"]:
                break
            continue

        dataset.save_episode()
        recorded_episodes += 1

    # === Clean up ===
    log_say("Stop recording")
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()
    if listener is not None:
        listener.stop()
    dataset.finalize()
    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    if args.push_to_hub:
        print(f"Uploading dataset to Hugging Face Hub: {dataset.repo_id}")
        dataset.push_to_hub()
        print(f"Dataset uploaded to: https://huggingface.co/datasets/{dataset.repo_id}")
    else:
        print("Skipping Hugging Face upload because --push_to_hub is false.")


if __name__ == "__main__":
    main()
