#!/usr/bin/env python3

"""Automatically collect scripted Genie Sim episodes into LeRobot.

The collector is task-parameterized: ``--task`` selects the simulator task
plugin whose episode script runs and whose structured result is validated
(``c3_l1`` by default, ``alohaminipro_fruits`` for the fruits scene).
"""

from __future__ import annotations

import argparse
import bisect
import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

CAMERAS = ("forward", "chest", "wrist_left", "wrist_right")
EPISODE_RESULT_FILENAME = "episode_result.json"
EPISODE_RESULT_STATUSES = frozenset(
    ("success", "failure", "timeout", "aborted", "error")
)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def nearest_index(stamps: list[int], target: int) -> int:
    right = bisect.bisect_left(stamps, target)
    if right == 0:
        return 0
    if right == len(stamps):
        return len(stamps) - 1
    left = right - 1
    return left if target - stamps[left] <= stamps[right] - target else right


def latest_not_after_index(stamps: list[int], target: int) -> int:
    index = bisect.bisect_right(stamps, target) - 1
    if index < 0:
        raise ValueError("target precedes the first action")
    return index


@dataclass(frozen=True)
class Sample:
    stamp_ns: int
    frame_index: int
    state_index: int
    action_index: int


def build_sample_plan(
    frame_stamps: list[int],
    state_stamps: list[int],
    action_stamps: list[int],
    fps: int,
) -> list[Sample]:
    if not frame_stamps or not state_stamps or not action_stamps:
        raise ValueError("raw episode must contain frames, states, and actions")
    if fps <= 0:
        raise ValueError("fps must be positive")
    start = max(frame_stamps[0], state_stamps[0], action_stamps[0])
    end = min(frame_stamps[-1], state_stamps[-1])
    if end < start:
        raise ValueError("raw episode streams do not overlap")
    count = ((end - start) * fps) // 1_000_000_000 + 1
    return [
        Sample(
            stamp_ns=start + round(index * 1_000_000_000 / fps),
            frame_index=nearest_index(
                frame_stamps, start + round(index * 1_000_000_000 / fps)
            ),
            state_index=nearest_index(
                state_stamps, start + round(index * 1_000_000_000 / fps)
            ),
            action_index=latest_not_after_index(
                action_stamps, start + round(index * 1_000_000_000 / fps)
            ),
        )
        for index in range(count)
    ]


def rgba_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[-1] != 4 or frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 HWC RGBA, got shape={frame.shape} dtype={frame.dtype}")
    return np.ascontiguousarray(frame[..., :3])


class RawEpisode:
    def __init__(self, attempt_dir: Path) -> None:
        self.attempt_dir = attempt_dir
        self.raw_dir = attempt_dir / "raw"
        self.metadata = json.loads((self.raw_dir / "metadata.json").read_text(encoding="utf-8"))
        self.frames = read_jsonl(self.raw_dir / "frames.jsonl")
        self.actions = read_jsonl(self.raw_dir / "actions.jsonl")
        self.states = read_jsonl(self.raw_dir / "states.jsonl")
        self.cameras = tuple(self.metadata["cameras"])
        self.shape = tuple(self.metadata["frame_shape"])
        self.state_order = tuple(self.metadata["state_order"])
        if int(self.metadata.get("version", -1)) != 1:
            raise ValueError(f"unsupported raw bundle version: {self.metadata.get('version')}")
        if self.cameras != CAMERAS:
            raise ValueError(f"unexpected camera order: {self.cameras}")
        if len(self.shape) != 3 or self.shape[-1] != 4:
            raise ValueError(f"unexpected raw frame shape: {self.shape}")
        if len(self.state_order) != 18:
            raise ValueError(f"unexpected state/action width: {len(self.state_order)}")
        if len(self.frames) != int(self.metadata["frame_count"]):
            raise ValueError("frame index and metadata count disagree")
        if len(self.actions) != int(self.metadata["action_count"]):
            raise ValueError("action index and metadata count disagree")
        if len(self.states) != int(self.metadata["state_count"]):
            raise ValueError("state index and metadata count disagree")
        if int(self.metadata.get("dropped_incomplete_batches_after_ready", 0)) != 0:
            raise ValueError("raw recorder dropped synchronized camera data during the episode")

        self._validate_rows()

        self._camera_arrays = {}
        expected_bytes = len(self.frames) * int(np.prod(self.shape))
        for camera in self.cameras:
            path = self.raw_dir / f"{camera}.rgba"
            if path.stat().st_size != expected_bytes:
                raise ValueError(f"raw camera size mismatch for {camera}: {path.stat().st_size}")
            self._camera_arrays[camera] = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(len(self.frames), *self.shape),
            )

    def _validate_rows(self) -> None:
        if [int(row["index"]) for row in self.frames] != list(range(len(self.frames))):
            raise ValueError("raw frame indices are not contiguous")
        for kind, rows in (("frame", self.frames), ("action", self.actions), ("state", self.states)):
            stamps = [int(row["stamp_ns"]) for row in rows]
            if not stamps:
                raise ValueError(f"raw episode contains no {kind} samples")
            if any(stamp <= 0 for stamp in stamps):
                raise ValueError(f"{kind} stream contains an invalid timestamp")
            if stamps != sorted(stamps):
                raise ValueError(f"{kind} timestamps are not monotonic")
        for kind, rows in (("action", self.actions), ("state", self.states)):
            for row in rows:
                values = np.asarray(row["values"], dtype=np.float64)
                if values.shape != (len(self.state_order),) or not np.isfinite(values).all():
                    raise ValueError(f"{kind} row is not a finite 18-D vector")

    def sample_plan(self, fps: int) -> list[Sample]:
        return build_sample_plan(
            [int(row["stamp_ns"]) for row in self.frames],
            [int(row["stamp_ns"]) for row in self.states],
            [int(row["stamp_ns"]) for row in self.actions],
            fps,
        )

    def image(self, camera: str, frame_index: int) -> np.ndarray:
        return rgba_to_rgb(np.asarray(self._camera_arrays[camera][frame_index]))


def docker_inspect(container: str) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", container],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    if len(payload) != 1 or not payload[0].get("State", {}).get("Running", False):
        raise RuntimeError(f"container is not running: {container}")
    return payload[0]


def workspace_mount(inspect: dict[str, Any]) -> Path:
    matches = [
        Path(mount["Source"])
        for mount in inspect.get("Mounts", [])
        if mount.get("Destination") == "/workspace" and mount.get("Type") == "bind"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one /workspace bind mount, found {matches}")
    return matches[0]


def capture_attempt(
    container: str,
    container_output: str,
    episode_action: str,
    task_id: str = "c3_l1",
) -> int:
    command = [
        "docker",
        "exec",
        "-u",
        "1000:1000",
        "-e",
        f"GENIESIM_TASK={task_id}",
        "-e",
        f"GENIESIM_TASK_ID={task_id}",
        container,
        "bash",
        "-lc",
        (
            "source /workspace/devel/setup.bash && "
            "exec ros2 run genie_sim_engine alohamini_c3_l1_capture.sh \"$@\""
        ),
        "record-sim",
        "--output",
        container_output,
        "--action",
        episode_action,
        "--attempt-id",
        Path(container_output).name,
    ]
    return subprocess.run(command, check=False).returncode


def load_episode_result(attempt_dir: Path, task_id: str = "c3_l1") -> dict[str, Any] | None:
    """Load a valid shared EpisodeResult correlated to this raw attempt."""
    path = attempt_dir / EPISODE_RESULT_FILENAME
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    required = {
        "schema_version",
        "task_id",
        "attempt_id",
        "reset_generation",
        "status",
        "reason",
        "terminal_stage",
        "started_at_s",
        "finished_at_s",
        "metrics",
    }
    if not required.issubset(payload):
        return None
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
        or payload["task_id"] != task_id
    ):
        return None
    if payload["attempt_id"] != attempt_dir.name:
        return None
    generation = payload["reset_generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        return None
    if payload["status"] not in EPISODE_RESULT_STATUSES:
        return None
    if not isinstance(payload["reason"], str):
        return None
    terminal_stage = payload["terminal_stage"]
    if terminal_stage is not None and not isinstance(terminal_stage, str):
        return None
    started = payload["started_at_s"]
    finished = payload["finished_at_s"]
    if (
        isinstance(started, bool)
        or isinstance(finished, bool)
        or not isinstance(started, (int, float))
        or not isinstance(finished, (int, float))
        or started < 0
        or finished < started
    ):
        return None
    if not isinstance(payload["metrics"], dict):
        return None
    return payload


def load_raw_metadata(attempt_dir: Path) -> dict[str, Any] | None:
    """Load the recorder's raw stream counts for a rejected attempt."""
    path = attempt_dir / "raw" / "metadata.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    keys = (
        "frame_count",
        "action_count",
        "state_count",
        "dropped_incomplete_batches",
        "dropped_incomplete_batches_after_ready",
    )
    return {key: payload[key] for key in keys if key in payload}


def episode_succeeded(
    attempt_dir: Path,
    returncode: int,
    episode_action: str,
    task_id: str = "c3_l1",
) -> bool:
    if returncode != 0:
        return False
    if episode_action != "run":
        return True
    result = load_episode_result(attempt_dir, task_id)
    return result is not None and result["status"] == "success"


def create_dataset(args: argparse.Namespace, robot: LeKiwiClient) -> LeRobotDataset:
    root = args.root if args.root is not None else HF_LEROBOT_HOME / args.dataset
    if args.resume:
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset,
            root=root,
            image_writer_threads=args.image_writer_threads,
        )
        if dataset.meta.fps != args.fps:
            raise ValueError(
                f"resume dataset fps is {dataset.meta.fps}, requested {args.fps}"
            )
        return dataset
    features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR),
    }
    return LeRobotDataset.create(
        repo_id=args.dataset,
        root=root,
        fps=args.fps,
        features=features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=args.image_writer_threads,
    )


def append_episode(
    dataset: LeRobotDataset,
    raw: RawEpisode,
    fps: int,
    task: str,
    expected_order: tuple[str, ...],
) -> int:
    if raw.state_order != expected_order:
        raise ValueError("raw state/action order does not match LeRobot alohamini2pro")
    plan = raw.sample_plan(fps)
    unique_source_frames = len({sample.frame_index for sample in plan})
    if unique_source_frames < len(plan):
        print(
            f"Source images: {unique_source_frames} unique ROS frames -> "
            f"{len(plan)} dataset frames at {fps} FPS "
            f"({len(plan) - unique_source_frames} repeated)"
        )
    for index, sample in enumerate(plan):
        state = np.asarray(raw.states[sample.state_index]["values"], dtype=np.float32)
        action = np.asarray(raw.actions[sample.action_index]["values"], dtype=np.float32)
        frame = {
            "observation.state": state,
            "action": action,
            "task": task,
        }
        for camera in raw.cameras:
            frame[f"observation.images.{camera}"] = raw.image(camera, sample.frame_index)
        dataset.add_frame(frame)
        if index and index % 250 == 0:
            print(f"Converted {index}/{len(plan)} frames", flush=True)
    dataset.save_episode()
    return len(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--task",
        default="c3_l1",
        choices=("c3_l1", "alohaminipro_fruits"),
        help=(
            "simulator task plugin whose episode script runs and whose "
            "structured result is validated (default: c3_l1)"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset directory; defaults to HF_LEROBOT_HOME/<dataset>.",
    )
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--task_description", default="pickup1")
    parser.add_argument("--robot_id", default="sim_c3_l1")
    parser.add_argument("--robot_model", default="alohamini2pro", choices=["alohamini2pro"])
    parser.add_argument("--container", default="geniesim3")
    parser.add_argument("--episode_action", default="run", choices=["run", "pick", "place"])
    parser.add_argument(
        "--allow_partial_episode",
        action="store_true",
        help="Allow pick/place diagnostics to be saved despite lacking task success.",
    )
    parser.add_argument("--max_attempts", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep_raw", action="store_true")
    parser.add_argument("--keep_failed", action="store_true")
    parser.add_argument("--image_writer_threads", type=int, default=4)
    parser.add_argument(
        "--push_to_hub",
        type=parse_bool,
        nargs="?",
        const=True,
        default=False,
    )
    args = parser.parse_args()
    if min(args.num_episodes, args.fps, args.image_writer_threads) <= 0:
        parser.error("--num_episodes, --fps, and --image_writer_threads must be positive")
    if args.episode_action != "run" and not args.allow_partial_episode:
        parser.error("pick/place are diagnostics; pass --allow_partial_episode to save them")
    max_attempts = args.max_attempts or args.num_episodes * 3
    if max_attempts < args.num_episodes:
        parser.error("--max_attempts must be at least --num_episodes")

    inspect = docker_inspect(args.container)
    host_workspace = workspace_mount(inspect)
    raw_root = host_workspace / ".record_sim_raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    robot = LeKiwiClient(
        LeKiwiClientConfig(
            remote_ip="127.0.0.1",
            id=args.robot_id,
            robot_model=args.robot_model,
        )
    )
    expected_order = tuple(robot.action_features)
    dataset: LeRobotDataset | None = None
    successes = 0
    session_log = raw_root / f"session-{uuid.uuid4().hex[:12]}.jsonl"

    try:
        for attempt in range(1, max_attempts + 1):
            if successes >= args.num_episodes:
                break
            attempt_id = f"attempt-{attempt:04d}-{uuid.uuid4().hex[:8]}"
            attempt_dir = raw_root / attempt_id
            container_output = f"/workspace/.record_sim_raw/{attempt_id}"
            attempt_started_s = time.monotonic()
            print(f"Attempt {attempt}/{max_attempts}: {attempt_id}", flush=True)
            returncode = capture_attempt(
                args.container,
                container_output,
                args.episode_action,
                args.task,
            )
            capture_elapsed_s = time.monotonic() - attempt_started_s
            episode_result = (
                load_episode_result(attempt_dir, args.task)
                if args.episode_action == "run"
                else None
            )
            success = episode_succeeded(
                attempt_dir,
                returncode,
                args.episode_action,
                args.task,
            )
            if not success:
                if episode_result is not None:
                    failure_reason = f"task_result_{episode_result['status']}"
                elif returncode != 0:
                    failure_reason = f"capture_returncode_{returncode}"
                else:
                    failure_reason = "task_result_missing_or_invalid"
                failure_row = {
                    "attempt": attempt,
                    "attempt_id": attempt_id,
                    "saved": False,
                    "reason": failure_reason,
                }
                if episode_result is not None:
                    failure_row["reset_generation"] = episode_result["reset_generation"]
                    failure_row["episode_result"] = episode_result
                raw_metadata = load_raw_metadata(attempt_dir)
                if raw_metadata is not None:
                    failure_row["raw_metadata"] = raw_metadata
                append_jsonl(session_log, failure_row)
                print(
                    f"Attempt failed (returncode={returncode}); episode was not added "
                    f"[capture {capture_elapsed_s:.1f}s]",
                    flush=True,
                )
                if not args.keep_failed:
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                continue

            try:
                raw = RawEpisode(attempt_dir)
            except ValueError as exc:
                append_jsonl(
                    session_log,
                    {
                        "attempt": attempt,
                        "attempt_id": attempt_id,
                        "saved": False,
                        "reason": "conversion_failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(f"Conversion failed ({type(exc).__name__}: {exc})")
                if args.keep_failed:
                    print(f"Failed raw bundle retained at {attempt_dir}")
                else:
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                    print(f"Failed raw bundle deleted: {attempt_dir}")
                continue
            try:
                if dataset is None:
                    dataset = create_dataset(args, robot)
                frame_count = append_episode(
                    dataset,
                    raw,
                    args.fps,
                    args.task_description,
                    expected_order,
                )
            except Exception:
                if dataset is not None and dataset.has_pending_frames():
                    dataset.clear_episode_buffer()
                print(f"Dataset conversion failed; raw bundle retained at {attempt_dir}")
                raise
            successes += 1
            success_row = {
                "attempt": attempt,
                "attempt_id": attempt_id,
                "saved": True,
                "dataset_episode": dataset.meta.total_episodes - 1,
                "frames": frame_count,
            }
            if episode_result is not None:
                success_row["reset_generation"] = episode_result["reset_generation"]
                success_row["episode_result"] = episode_result
            append_jsonl(session_log, success_row)
            print(
                f"Saved successful episode {successes}/{args.num_episodes} "
                f"({frame_count} frames) [capture {capture_elapsed_s:.1f}s, "
                f"convert {time.monotonic() - attempt_started_s - capture_elapsed_s:.1f}s]",
                flush=True,
            )
            if not args.keep_raw:
                shutil.rmtree(attempt_dir)
    finally:
        if dataset is not None:
            dataset.finalize()

    if dataset is None or successes != args.num_episodes:
        raise RuntimeError(
            f"collected {successes}/{args.num_episodes} successful episodes in {max_attempts} attempts"
        )
    print(f"Dataset saved locally at: {dataset.root.resolve()}")
    print(f"Attempt log: {session_log.resolve()}")
    if args.push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    main()
