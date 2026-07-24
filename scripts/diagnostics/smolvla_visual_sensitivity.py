#!/usr/bin/env python3

"""Measure how much a SmolVLA checkpoint reacts to the visible blue block.

This is an offline diagnostic: it reads dataset frames and a local checkpoint,
but never connects to or commands a robot.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

CAMERAS = (
    "observation.images.forward",
    "observation.images.chest",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)


@dataclass(frozen=True)
class Probe:
    episode: int
    timestamp_s: float
    target_camera: str


# Frames selected after visual inspection: the block is clearly visible in the
# named wrist camera and the set covers both left- and right-wrist approaches.
DEFAULT_PROBES = (
    Probe(0, 75.0, "observation.images.wrist_left"),
    Probe(50, 51.0, "observation.images.wrist_left"),
    Probe(75, 39.0, "observation.images.wrist_right"),
    Probe(93, 39.0, "observation.images.wrist_right"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/yan/.cache/huggingface/lerobot/c3smolvla_l1"),
    )
    parser.add_argument("--dataset-repo-id", default="c3smolvla_l1")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/train/smolvla_c3_l1/checkpoints/060000/pretrained_model"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override the checkpoint's flow-matching denoising steps.",
    )
    return parser.parse_args()


def load_episode_rows(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata found below {dataset_root}")
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def dataset_index_for_probe(rows: pd.DataFrame, probe: Probe, fps: float) -> int:
    row = rows.loc[rows["episode_index"] == probe.episode]
    if len(row) != 1:
        raise ValueError(f"Expected one metadata row for episode {probe.episode}, got {len(row)}")
    row = row.iloc[0]
    frame_index = min(round(probe.timestamp_s * fps), int(row["length"]) - 1)
    return int(row["dataset_from_index"]) + frame_index


def largest_blue_bbox(image: torch.Tensor) -> tuple[int, int, int, int]:
    """Return an expanded bbox for the largest saturated-blue component."""
    rgb = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([100, 100, 40]), np.array([135, 255, 255]))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        raise RuntimeError("No saturated-blue connected component found")

    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (int(value) for value in stats[component])
    if area < 100:
        raise RuntimeError(f"Largest blue component is too small for a reliable mask: {area} pixels")

    margin = max(12, round(0.2 * max(width, height)))
    image_height, image_width = rgb.shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(image_width, x + width + margin)
    y1 = min(image_height, y + height + margin)
    return x0, y0, x1, y1


def make_conditions(sample: dict, target_camera: str) -> tuple[dict[str, dict], tuple[int, int, int, int]]:
    bbox = largest_blue_bbox(sample[target_camera])
    x0, y0, x1, y1 = bbox
    other_wrist = (
        "observation.images.wrist_right"
        if target_camera.endswith("wrist_left")
        else "observation.images.wrist_left"
    )

    conditions: dict[str, dict] = {"original": copy.deepcopy(sample)}

    block_masked = copy.deepcopy(sample)
    block_masked[target_camera][:, y0:y1, x0:x1] = 0.5
    conditions["block_bbox_gray"] = block_masked

    target_camera_masked = copy.deepcopy(sample)
    target_camera_masked[target_camera].fill_(0.5)
    conditions["target_wrist_gray"] = target_camera_masked

    other_camera_masked = copy.deepcopy(sample)
    other_camera_masked[other_wrist].fill_(0.5)
    conditions["other_wrist_gray"] = other_camera_masked
    return conditions, bbox


def predict_chunk(policy, preprocessor, sample: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    processed = preprocessor(copy.deepcopy(sample))
    policy.reset()
    with torch.inference_mode():
        actions = policy.predict_action_chunk(processed)
    return actions.squeeze(0).detach().float().cpu()


def difference(reference: torch.Tensor, candidate: torch.Tensor, horizon: int) -> dict[str, float]:
    delta = candidate[:horizon] - reference[:horizon]
    per_dim = delta.abs().mean(dim=0)
    return {
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "max": float(delta.abs().max()),
        "left_arm_mae": float(per_dim[:7].mean()),
        "right_arm_mae": float(per_dim[7:14].mean()),
        "base_lift_mae": float(per_dim[14:].mean()),
    }


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{key}={value:.5f}" for key, value in metrics.items())


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config)
    policy = policy.to(args.device)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=dataset_root,
        video_backend="torchcodec",
    )
    rows = load_episode_rows(dataset_root)
    fps = float(dataset.meta.fps)

    aggregate: dict[tuple[str, int], list[dict[str, float]]] = {}
    stochastic: dict[int, list[dict[str, float]]] = {10: [], 50: []}

    for probe in DEFAULT_PROBES:
        index = dataset_index_for_probe(rows, probe, fps)
        sample = dataset[index]
        conditions, bbox = make_conditions(sample, probe.target_camera)
        print(
            f"\nepisode={probe.episode} t={probe.timestamp_s:.1f}s index={index} "
            f"target={probe.target_camera} blue_bbox={bbox}"
        )

        predictions: dict[tuple[str, int], torch.Tensor] = {}
        for condition_name, condition_sample in conditions.items():
            for seed in args.seeds:
                predictions[(condition_name, seed)] = predict_chunk(
                    policy, preprocessor, condition_sample, seed
                )

        for horizon in (10, 50):
            print(f"  horizon={horizon}")
            for condition_name in conditions:
                if condition_name == "original":
                    continue
                paired = [
                    difference(
                        predictions[("original", seed)],
                        predictions[(condition_name, seed)],
                        horizon,
                    )
                    for seed in args.seeds
                ]
                averaged = {
                    key: float(np.mean([metrics[key] for metrics in paired])) for key in paired[0]
                }
                aggregate.setdefault((condition_name, horizon), []).append(averaged)
                print(f"    {condition_name:18s} {format_metrics(averaged)}")

            if len(args.seeds) >= 2:
                noise_metrics = difference(
                    predictions[("original", args.seeds[0])],
                    predictions[("original", args.seeds[1])],
                    horizon,
                )
                stochastic[horizon].append(noise_metrics)
                print(f"    {'different_noise':18s} {format_metrics(noise_metrics)}")

    print("\n=== Mean across probes ===")
    for horizon in (10, 50):
        print(f"horizon={horizon}")
        for condition_name in ("block_bbox_gray", "target_wrist_gray", "other_wrist_gray"):
            rows_for_condition = aggregate[(condition_name, horizon)]
            averaged = {
                key: float(np.mean([metrics[key] for metrics in rows_for_condition]))
                for key in rows_for_condition[0]
            }
            print(f"  {condition_name:18s} {format_metrics(averaged)}")
        if stochastic[horizon]:
            noise_average = {
                key: float(np.mean([metrics[key] for metrics in stochastic[horizon]]))
                for key in stochastic[horizon][0]
            }
            print(f"  {'different_noise':18s} {format_metrics(noise_average)}")


if __name__ == "__main__":
    main()
