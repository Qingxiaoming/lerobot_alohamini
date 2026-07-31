import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[2] / "examples" / "alohamini" / "record_sim.py"
SPEC = importlib.util.spec_from_file_location("record_sim", SCRIPT)
record_sim = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = record_sim
SPEC.loader.exec_module(record_sim)


def test_build_sample_plan_resamples_frames_and_holds_latest_action():
    plan = record_sim.build_sample_plan(
        frame_stamps=[0, 100_000_000, 200_000_000, 300_000_000],
        state_stamps=[0, 100_000_000, 200_000_000, 300_000_000],
        action_stamps=[50_000_000, 150_000_000],
        fps=10,
    )

    assert [sample.stamp_ns for sample in plan] == [50_000_000, 150_000_000, 250_000_000]
    assert [sample.frame_index for sample in plan] == [0, 1, 2]
    assert [sample.action_index for sample in plan] == [0, 1, 1]


def test_rgba_to_rgb_drops_alpha_without_swapping_red_and_blue():
    rgba = np.array([[[240, 30, 10, 7]]], dtype=np.uint8)

    rgb = record_sim.rgba_to_rgb(rgba)

    np.testing.assert_array_equal(rgb, np.array([[[240, 30, 10]]], dtype=np.uint8))


def test_workspace_mount_requires_exact_workspace_bind():
    inspect = {
        "Mounts": [
            {"Type": "bind", "Source": "/host/genie", "Destination": "/workspace"},
            {"Type": "volume", "Source": "cache", "Destination": "/cache"},
        ]
    }

    assert record_sim.workspace_mount(inspect) == Path("/host/genie")


def test_episode_success_requires_full_task_result_log(tmp_path):
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")
    (tmp_path / "episode.log").write_text("data: pending\n", encoding="utf-8")
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")
    (tmp_path / "episode.log").write_text("data: success\n", encoding="utf-8")
    assert record_sim.episode_succeeded(tmp_path, 0, "run")
    assert not record_sim.episode_succeeded(tmp_path, 3, "run")


def test_raw_bundle_converts_into_real_lerobot_episode(tmp_path):
    attempt_dir = tmp_path / "attempt"
    raw_dir = attempt_dir / "raw"
    raw_dir.mkdir(parents=True)
    shape = (480, 640, 4)
    frame_stamps = [1_000_000_000, 1_100_000_000, 1_200_000_000]
    frames = []
    camera_payloads = {camera: bytearray() for camera in record_sim.CAMERAS}
    for index, stamp in enumerate(frame_stamps):
        frames.append({"index": index, "stamp_ns": stamp})
        rgba = np.zeros(shape, dtype=np.uint8)
        rgba[..., 0] = 10 + index
        rgba[..., 1] = 20
        rgba[..., 2] = 30
        rgba[..., 3] = 255
        for camera in record_sim.CAMERAS:
            camera_payloads[camera].extend(rgba.tobytes())
    for camera, payload in camera_payloads.items():
        (raw_dir / f"{camera}.rgba").write_bytes(payload)

    order = tuple(
        record_sim.LeKiwiClient(
            record_sim.LeKiwiClientConfig(
                remote_ip="127.0.0.1",
                id="test_sim",
                robot_model="alohamini2pro",
            )
        ).action_features
    )
    actions = [
        {"stamp_ns": stamp, "sequence": index + 1, "values": [float(index)] * 18}
        for index, stamp in enumerate(frame_stamps[:2])
    ]
    states = [
        {"stamp_ns": stamp, "values": [float(index + 10)] * 18}
        for index, stamp in enumerate(frame_stamps)
    ]
    metadata = {
        "version": 1,
        "cameras": list(record_sim.CAMERAS),
        "source_encoding": "rgba8",
        "frame_shape": list(shape),
        "state_order": list(order),
        "frame_count": len(frames),
        "action_count": len(actions),
        "state_count": len(states),
        "dropped_incomplete_batches": 0,
        "dropped_incomplete_batches_after_ready": 0,
    }
    (raw_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name, rows in (("frames", frames), ("actions", actions), ("states", states)):
        (raw_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    robot = record_sim.LeKiwiClient(
        record_sim.LeKiwiClientConfig(
            remote_ip="127.0.0.1",
            id="test_sim",
            robot_model="alohamini2pro",
        )
    )
    features = {
        **record_sim.hw_to_dataset_features(robot.action_features, record_sim.ACTION),
        **record_sim.hw_to_dataset_features(robot.observation_features, record_sim.OBS_STR),
    }
    dataset = record_sim.LeRobotDataset.create(
        repo_id="tests/alohamini_record_sim",
        root=tmp_path / "dataset",
        fps=10,
        features=features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=2,
    )
    try:
        count = record_sim.append_episode(
            dataset,
            record_sim.RawEpisode(attempt_dir),
            fps=10,
            task="pickup1",
            expected_order=order,
        )
    finally:
        dataset.finalize()

    assert count == 3
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 3
