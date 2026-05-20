#!/usr/bin/env python3

import argparse
import time

import lerobot.robots.alohamini  # noqa: F401 — registers alohamini_client robot type

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.rollout.inference.factory import SyncInferenceConfig, create_inference_engine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot
from lerobot.robots.alohamini import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import auto_select_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts, hw_to_dataset_features
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun


def main():
    parser = argparse.ArgumentParser(description="Evaluate AlohaMini robot with a pretrained policy")
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--episode_time", type=int, default=60)
    parser.add_argument("--task_description", type=str, default="robot task")
    parser.add_argument("--hf_model_id", type=str, required=True)
    parser.add_argument("--hf_dataset_id", type=str, required=True)
    parser.add_argument("--remote_ip", type=str, default="127.0.0.1")
    parser.add_argument("--robot_id", type=str, default="lekiwi")
    parser.add_argument("--robot_model", type=str, default="alohamini1",
                        choices=["alohamini1", "alohamini2", "alohamini2pro"],
                        help="Must match the robot_model on the Pi host side")
    args = parser.parse_args()

    device = str(auto_select_torch_device())

    # === Policy ===
    policy_cfg = PreTrainedConfig.from_pretrained(args.hf_model_id)
    policy_cfg.pretrained_path = args.hf_model_id
    policy = get_policy_class(policy_cfg.type).from_pretrained(args.hf_model_id, config=policy_cfg)
    policy = policy.to(device)
    policy.eval()

    # === Robot ===
    robot_config = LeKiwiClientConfig(remote_ip=args.remote_ip, id=args.robot_id,
                                      robot_model=args.robot_model)
    robot = LeKiwiClient(robot_config)
    robot.connect()
    robot_wrapper = ThreadSafeRobot(robot)

    # === Processors ===
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # === Dataset features ===
    # Use all observation features (pos + base vel/height) to match what record_bi.py records.
    observation_features_hw = robot.observation_features
    action_features_hw = robot.action_features

    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=create_initial_features(action=action_features_hw),
        use_videos=True,
    )
    observation_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=create_initial_features(observation=observation_features_hw),
        use_videos=True,
    )
    dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)
    hw_features = hw_to_dataset_features(observation_features_hw, "observation")
    ordered_action_keys = list(action_features_hw.keys())

    # === Dataset ===
    dataset = LeRobotDataset.create(
        repo_id=args.hf_dataset_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # === Policy processors (needs dataset stats) ===
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.hf_model_id,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # === Inference engine ===
    engine = create_inference_engine(
        SyncInferenceConfig(),
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        robot_wrapper=robot_wrapper,
        hw_features=hw_features,
        dataset_features=dataset_features,
        ordered_action_keys=ordered_action_keys,
        task=args.task_description,
        fps=float(args.fps),
        device=device,
    )
    engine.start()

    init_rerun(session_name="alohamini_evaluate")
    log_say("Starting evaluation")

    control_interval = 1.0 / args.fps
    recorded = 0

    while recorded < args.num_episodes:
        log_say(f"Eval episode {recorded + 1} of {args.num_episodes}")
        engine.reset()
        start = time.perf_counter()

        while (time.perf_counter() - start) < args.episode_time:
            loop_start = time.perf_counter()

            obs_raw = robot.get_observation()
            obs_processed = robot_observation_processor(obs_raw)
            obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            action_tensor = engine.get_action(obs_frame)
            if action_tensor is not None:
                action_dict = {k: action_tensor[i].item() for i, k in enumerate(ordered_action_keys)}
                robot.send_action(robot_action_processor((action_dict, obs_raw)))
                action_frame = build_dataset_frame(dataset_features, action_dict, prefix=ACTION)
                dataset.add_frame({**obs_frame, **action_frame, "task": args.task_description})

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)

        dataset.save_episode()
        recorded += 1

    log_say("Evaluation complete")
    engine.stop()
    robot.disconnect()
    dataset.finalize()
    dataset.push_to_hub()


if __name__ == "__main__":
    main()
