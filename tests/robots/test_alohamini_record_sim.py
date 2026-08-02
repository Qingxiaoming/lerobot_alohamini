import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_capture_passes_attempt_identity_to_simulator(monkeypatch):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(record_sim.subprocess, "run", fake_run)

    returncode = record_sim.capture_attempt(
        "geniesim3",
        "/workspace/.record_sim_raw/attempt-0007-deadbeef",
        "run",
        "alohaminipro_fruits",
    )

    assert returncode == 0
    assert captured["check"] is False
    index = captured["command"].index("--attempt-id")
    assert captured["command"][index + 1] == "attempt-0007-deadbeef"
    task_env_index = captured["command"].index("-e")
    assert captured["command"][task_env_index + 1] == "GENIESIM_TASK=alohaminipro_fruits"
    assert (
        captured["command"][task_env_index + 3]
        == "GENIESIM_TASK_ID=alohaminipro_fruits"
    )


def test_episode_success_requires_correlated_structured_result(tmp_path):
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")
    (tmp_path / "episode.log").write_text("data: success\n", encoding="utf-8")
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")

    result = {
        "schema_version": 1,
        "task_id": "c3_l1",
        "attempt_id": tmp_path.name,
        "reset_generation": 7,
        "status": "failure",
        "reason": "simulator task-state reported failure",
        "terminal_stage": "retreat",
        "started_at_s": 100.0,
        "finished_at_s": 101.0,
        "metrics": {"source": "/genie_sim/task_state/result"},
    }
    result_path = tmp_path / record_sim.EPISODE_RESULT_FILENAME
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")

    result["status"] = "pending"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")

    result["attempt_id"] = "another-attempt"
    result["status"] = "success"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")

    result["attempt_id"] = tmp_path.name
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert record_sim.episode_succeeded(tmp_path, 0, "run")
    assert not record_sim.episode_succeeded(tmp_path, 3, "run")


def test_episode_result_task_id_is_parameterized(tmp_path):
    result = {
        "schema_version": 1,
        "task_id": "alohaminipro_fruits",
        "attempt_id": tmp_path.name,
        "reset_generation": 3,
        "status": "success",
        "reason": "simulator task-state reported success",
        "terminal_stage": "retreat",
        "started_at_s": 100.0,
        "finished_at_s": 101.0,
        "metrics": {"source": "/genie_sim/task_state/result"},
    }
    (tmp_path / record_sim.EPISODE_RESULT_FILENAME).write_text(
        json.dumps(result), encoding="utf-8"
    )
    # The default c3_l1 collector must reject the fruits result...
    assert not record_sim.episode_succeeded(tmp_path, 0, "run")
    # ...and the matching task id must accept it.
    assert record_sim.episode_succeeded(
        tmp_path, 0, "run", task_id="alohaminipro_fruits"
    )
    assert not record_sim.episode_succeeded(
        tmp_path, 0, "run", task_id="c3_l1"
    )


def test_load_raw_metadata_returns_count_summary_or_none(tmp_path):
    attempt_dir = tmp_path / "attempt"
    assert record_sim.load_raw_metadata(attempt_dir) is None

    raw_dir = attempt_dir / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "metadata.json").write_text("not json", encoding="utf-8")
    assert record_sim.load_raw_metadata(attempt_dir) is None

    (raw_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": 1,
                "cameras": ["forward"],
                "frame_count": 0,
                "action_count": 5,
                "state_count": 42,
                "dropped_incomplete_batches": 1,
                "dropped_incomplete_batches_after_ready": 0,
            }
        ),
        encoding="utf-8",
    )
    assert record_sim.load_raw_metadata(attempt_dir) == {
        "frame_count": 0,
        "action_count": 5,
        "state_count": 42,
        "dropped_incomplete_batches": 1,
        "dropped_incomplete_batches_after_ready": 0,
    }


def test_rejected_attempt_records_reset_generation_and_raw_metadata(monkeypatch, tmp_path):
    attempts = []

    class FakeRobot:
        action_features = tuple(f"field_{index}" for index in range(18))
        name = "alohamini"

    class FakeDataset:
        def __init__(self):
            self.root = tmp_path / "dataset"
            self.meta = SimpleNamespace(total_episodes=1)
            self.finalized = False

        def has_pending_frames(self):
            return False

        def finalize(self):
            self.finalized = True

    dataset = FakeDataset()

    def fake_capture_attempt(_container, container_output, _episode_action, _task_id=None):
        attempt_dir = tmp_path / ".record_sim_raw" / Path(container_output).name
        attempt_dir.mkdir(parents=True)
        raw_dir = attempt_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "frame_count": 0,
                    "action_count": 0,
                    "state_count": 1593,
                    "dropped_incomplete_batches": 0,
                    "dropped_incomplete_batches_after_ready": 0,
                }
            ),
            encoding="utf-8",
        )
        if not attempts:
            result = {
                "schema_version": 1,
                "task_id": "c3_l1",
                "attempt_id": attempt_dir.name,
                "reset_generation": 2,
                "status": "failure",
                "reason": "simulator task-state reported failure",
                "terminal_stage": "retreat",
                "started_at_s": 100.0,
                "finished_at_s": 101.0,
                "metrics": {"source": "/genie_sim/task_state/result"},
            }
            (attempt_dir / record_sim.EPISODE_RESULT_FILENAME).write_text(
                json.dumps(result), encoding="utf-8"
            )
        attempts.append(attempt_dir)
        return 0

    monkeypatch.setattr(record_sim, "docker_inspect", lambda _container: {})
    monkeypatch.setattr(record_sim, "workspace_mount", lambda _inspect: tmp_path)
    monkeypatch.setattr(record_sim, "LeKiwiClient", lambda _config: FakeRobot())
    monkeypatch.setattr(record_sim, "capture_attempt", fake_capture_attempt)
    monkeypatch.setattr(record_sim, "episode_succeeded", lambda *_args: False)
    monkeypatch.setattr(record_sim, "RawEpisode", lambda attempt_dir: SimpleNamespace())
    monkeypatch.setattr(record_sim, "create_dataset", lambda *_args: dataset)
    monkeypatch.setattr(record_sim, "append_episode", lambda *_args: 12)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_sim.py",
            "--dataset",
            "test",
            "--root",
            str(tmp_path),
            "--num_episodes",
            "1",
            "--max_attempts",
            "3",
            "--keep_failed",
        ],
    )

    with pytest.raises(RuntimeError):
        record_sim.main()

    rows = [
        json.loads(line)
        for line in next((tmp_path / ".record_sim_raw").glob("session-*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 3
    for row in rows:
        assert row["saved"] is False

    first = rows[0]
    assert first["reason"] == "task_result_failure"
    assert first["reset_generation"] == 2
    assert first["episode_result"]["status"] == "failure"
    assert first["raw_metadata"] == {
        "frame_count": 0,
        "action_count": 0,
        "state_count": 1593,
        "dropped_incomplete_batches": 0,
        "dropped_incomplete_batches_after_ready": 0,
    }

    for row in rows[1:]:
        assert row["reason"] == "task_result_missing_or_invalid"
        assert "reset_generation" not in row
        assert row["raw_metadata"]["state_count"] == 1593


def test_conversion_failure_deletes_attempt_and_continues(monkeypatch, tmp_path):
    attempts = []

    class FakeRobot:
        action_features = tuple(f"field_{index}" for index in range(18))
        name = "alohamini"

    class FakeDataset:
        def __init__(self):
            self.root = tmp_path / "dataset"
            self.meta = SimpleNamespace(total_episodes=1)
            self.finalized = False

        def has_pending_frames(self):
            return False

        def finalize(self):
            self.finalized = True

    dataset = FakeDataset()

    def fake_capture_attempt(_container, container_output, _episode_action, _task_id=None):
        attempt_dir = tmp_path / ".record_sim_raw" / Path(container_output).name
        attempt_dir.mkdir(parents=True)
        attempts.append(attempt_dir)
        return 0

    def fake_raw_episode(attempt_dir):
        if attempt_dir == attempts[0]:
            raise ValueError("raw recorder dropped synchronized camera data during the episode")
        return SimpleNamespace()

    monkeypatch.setattr(record_sim, "docker_inspect", lambda _container: {})
    monkeypatch.setattr(record_sim, "workspace_mount", lambda _inspect: tmp_path)
    monkeypatch.setattr(record_sim, "LeKiwiClient", lambda _config: FakeRobot())
    monkeypatch.setattr(record_sim, "capture_attempt", fake_capture_attempt)
    monkeypatch.setattr(record_sim, "episode_succeeded", lambda *_args: True)
    monkeypatch.setattr(record_sim, "RawEpisode", fake_raw_episode)
    monkeypatch.setattr(record_sim, "create_dataset", lambda *_args: dataset)
    monkeypatch.setattr(record_sim, "append_episode", lambda *_args: 12)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_sim.py",
            "--dataset",
            "test",
            "--root",
            str(dataset.root),
            "--num_episodes",
            "1",
            "--max_attempts",
            "2",
        ],
    )

    record_sim.main()

    assert len(attempts) == 2
    assert not attempts[0].exists()
    assert not attempts[1].exists()
    rows = [
        json.loads(line)
        for line in next((tmp_path / ".record_sim_raw").glob("session-*.jsonl"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["reason"] == "conversion_failed"
    assert rows[0]["error_type"] == "ValueError"
    assert "dropped synchronized camera data" in rows[0]["error"]
    assert rows[1]["saved"] is True
    assert dataset.finalized


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
