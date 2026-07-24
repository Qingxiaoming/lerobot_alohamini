#!/usr/bin/env python3

"""Test whether SmolVLA reacts consistently when the visible blue block moves."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors

PROBES = (
    (0, 75.0, "observation.images.wrist_left"),
    (50, 51.0, "observation.images.wrist_left"),
    (75, 39.0, "observation.images.wrist_right"),
    (93, 39.0, "observation.images.wrist_right"),
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
    parser.add_argument("--shift-px", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 2000, 3000])
    return parser.parse_args()


def largest_blue_component(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    blue = cv2.inRange(hsv, np.array([100, 100, 40]), np.array([135, 255, 255]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(blue)
    if count <= 1:
        raise RuntimeError("No saturated-blue component found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = (int(value) for value in stats[component])
    if area < 100:
        raise RuntimeError(f"Largest blue component is too small: {area} pixels")
    mask = np.where(labels == component, 255, 0).astype(np.uint8)
    # Include compressed/shaded boundary pixels around the saturated-blue core.
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
    return mask, (x, y, width, height)


def move_blue_component(image: torch.Tensor, shift_x: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    rgb = (
        image.detach()
        .cpu()
        .permute(1, 2, 0)
        .clamp(0, 1)
        .mul(255)
        .to(torch.uint8)
        .numpy()
    )
    mask, bbox = largest_blue_component(rgb)
    height, width = mask.shape
    transform = np.float32([[1, 0, shift_x], [0, 1, 0]])

    # Fill the source location with real texture sampled from the opposite side
    # of the same frame. Inpainting leaves a conspicuous smooth blob on the
    # carpet and would confound "object moved" with "new blur appeared".
    x, _, width_box, _ = bbox
    donor_offset = width // 2 if x + width_box / 2 < width / 2 else -(width // 2)
    donor = cv2.warpAffine(
        rgb,
        np.float32([[1, 0, -donor_offset], [0, 1, 0]]),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    background = rgb.copy()
    background[mask > 0] = donor[mask > 0]
    shifted_rgb = cv2.warpAffine(
        rgb,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    shifted_mask = cv2.warpAffine(
        mask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    alpha = shifted_mask.astype(np.float32)[..., None] / 255.0
    result = shifted_rgb.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    tensor = torch.from_numpy(result.round().clip(0, 255).astype(np.uint8)).permute(2, 0, 1).float() / 255
    return tensor, bbox


def load_rows(dataset_root: Path) -> pd.DataFrame:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def dataset_index(rows: pd.DataFrame, episode: int, timestamp_s: float, fps: float) -> int:
    row = rows.loc[rows["episode_index"] == episode].iloc[0]
    frame = min(round(timestamp_s * fps), int(row["length"]) - 1)
    return int(row["dataset_from_index"]) + frame


def predict(policy, preprocessor, sample: dict, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processed = preprocessor(copy.deepcopy(sample))
    policy.reset()
    with torch.inference_mode():
        return policy.predict_action_chunk(processed).squeeze(0).detach().float().cpu()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(cosine_similarity(left.flatten(), right.flatten(), dim=0))


def mae(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().mean())


def arm_slice(name: str) -> slice:
    if name.endswith("wrist_left"):
        return slice(0, 7)
    return slice(7, 14)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = str(checkpoint)
    policy = get_policy_class(config.type).from_pretrained(checkpoint, config=config).to(args.device)
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
    rows = load_rows(dataset_root)
    summaries: dict[int, list[dict[str, float]]] = {10: [], 50: []}

    for episode, timestamp_s, camera in PROBES:
        index = dataset_index(rows, episode, timestamp_s, float(dataset.meta.fps))
        original = dataset[index]
        shifted_left = copy.deepcopy(original)
        shifted_right = copy.deepcopy(original)
        shifted_left[camera], bbox = move_blue_component(original[camera], -args.shift_px)
        shifted_right[camera], _ = move_blue_component(original[camera], args.shift_px)

        predictions: dict[tuple[str, int], torch.Tensor] = {}
        for name, sample in (
            ("original", original),
            ("shift_left", shifted_left),
            ("shift_right", shifted_right),
        ):
            for seed in args.seeds:
                predictions[(name, seed)] = predict(policy, preprocessor, sample, seed)

        print(
            f"\nepisode={episode} t={timestamp_s:.1f}s camera={camera} "
            f"blue_xywh={bbox} shift=±{args.shift_px}px"
        )
        target_arm = arm_slice(camera)
        for horizon in (10, 50):
            paired = []
            direction_vectors = []
            for seed in args.seeds:
                original_actions = predictions[("original", seed)][:horizon, target_arm]
                left_actions = predictions[("shift_left", seed)][:horizon, target_arm]
                right_actions = predictions[("shift_right", seed)][:horizon, target_arm]
                left_delta = left_actions - original_actions
                right_delta = right_actions - original_actions
                direction_vectors.append(right_actions - left_actions)
                paired.append(
                    {
                        "left_vs_right_mae": mae(left_actions, right_actions),
                        "left_delta_mae": float(left_delta.abs().mean()),
                        "right_delta_mae": float(right_delta.abs().mean()),
                        "opposite_cos": cosine(left_delta, right_delta),
                    }
                )

            noise = [
                mae(
                    predictions[("original", args.seeds[i])][:horizon, target_arm],
                    predictions[("original", args.seeds[i + 1])][:horizon, target_arm],
                )
                for i in range(len(args.seeds) - 1)
            ]
            direction_consistency = [
                cosine(direction_vectors[i], direction_vectors[i + 1])
                for i in range(len(direction_vectors) - 1)
            ]
            result = {
                key: float(np.mean([item[key] for item in paired])) for key in paired[0]
            }
            result["noise_mae"] = float(np.mean(noise))
            result["direction_seed_cos"] = float(np.mean(direction_consistency))
            result["signal_noise_ratio"] = result["left_vs_right_mae"] / max(result["noise_mae"], 1e-8)
            summaries[horizon].append(result)
            print(
                f"  horizon={horizon:2d} "
                + " ".join(f"{key}={value:.4f}" for key, value in result.items())
            )

    print("\n=== Mean across probes ===")
    for horizon in (10, 50):
        mean_result = {
            key: float(np.mean([item[key] for item in summaries[horizon]]))
            for key in summaries[horizon][0]
        }
        print(
            f"horizon={horizon:2d} "
            + " ".join(f"{key}={value:.4f}" for key, value in mean_result.items())
        )


if __name__ == "__main__":
    main()
