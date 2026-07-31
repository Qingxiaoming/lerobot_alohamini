import base64

import cv2
import numpy as np

from lerobot.robots.alohamini.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.alohamini.lekiwi_client import LeKiwiClient
from lerobot.utils.constants import ACTION


def test_sim_record_action_requires_and_preserves_alohamini2pro_order():
    client = LeKiwiClient(
        LeKiwiClientConfig(remote_ip="127.0.0.1", id="test", robot_model="alohamini2pro")
    )
    expected = {name: float(index) for index, name in enumerate(client.action_features)}
    observation = {
        "_recording": {
            "action": expected,
            "action_sequence": 17,
            "command_stamp_ns": 123456,
        }
    }

    assert client._update_record_action_from_observation(observation)
    np.testing.assert_array_equal(client.last_remote_action[ACTION], np.arange(18, dtype=np.float32))
    assert list(client.last_remote_action)[:-1] == list(client.action_features)
    assert client.last_recording_metadata == {
        "action_sequence": 17,
        "command_stamp_ns": 123456,
    }


def test_sim_record_action_rejects_incomplete_payload():
    client = LeKiwiClient(
        LeKiwiClientConfig(remote_ip="127.0.0.1", id="test", robot_model="alohamini2pro")
    )

    assert not client._update_record_action_from_observation(
        {"_recording": {"action": {"x.vel": 0.0}, "action_sequence": 1}}
    )
    assert client.last_remote_action == {}


def test_sim_jpeg_metadata_converts_opencv_bgr_decode_to_lerobot_rgb():
    client = LeKiwiClient(
        LeKiwiClientConfig(remote_ip="127.0.0.1", id="test", robot_model="alohamini2pro")
    )
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    rgb[..., 1] = 30
    rgb[..., 2] = 10
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    observation = {
        "_images": {
            "encoding": "jpeg",
            "color_space": "rgb",
            "frames": {
                "forward": {
                    "source_stamp_ns": 123456,
                    "shape": [480, 640, 3],
                }
            },
        },
        "forward": base64.b64encode(encoded).decode("ascii"),
    }

    frames, _ = client._remote_state_from_obs(observation)

    assert frames["forward"].shape == (480, 640, 3)
    assert frames["forward"].dtype == np.uint8
    assert float(frames["forward"][..., 0].mean()) > 230
    assert float(frames["forward"][..., 2].mean()) < 20
    assert client.last_image_metadata["frames"]["forward"]["source_stamp_ns"] == 123456


def test_sim_image_with_wrong_shape_is_rejected():
    client = LeKiwiClient(
        LeKiwiClientConfig(remote_ip="127.0.0.1", id="test", robot_model="alohamini2pro")
    )
    ok, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
    assert ok

    frames, _ = client._remote_state_from_obs(
        {
            "_images": {"encoding": "jpeg", "color_space": "rgb"},
            "forward": base64.b64encode(encoded).decode("ascii"),
        }
    )

    assert frames == {}
