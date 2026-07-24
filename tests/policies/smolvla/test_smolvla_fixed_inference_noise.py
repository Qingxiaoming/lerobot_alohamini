#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def make_policy(config: SmolVLAConfig) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.__new__(SmolVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = config
    return policy


def test_fixed_inference_noise_is_reused_without_advancing_global_rng():
    policy = make_policy(
        SmolVLAConfig(
            chunk_size=5,
            n_action_steps=5,
            max_action_dim=7,
            inference_fixed_noise_seed=123,
        )
    )
    state = torch.zeros(1, 4)

    torch.manual_seed(99)
    expected_next_random_value = torch.randn(1)
    torch.manual_seed(99)
    first = policy._make_fixed_inference_noise(state)
    second = policy._make_fixed_inference_noise(state)
    actual_next_random_value = torch.randn(1)

    assert first is not None
    assert first.shape == (1, 5, 7)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(actual_next_random_value, expected_next_random_value)


def test_fixed_inference_noise_is_disabled_by_default():
    policy = make_policy(SmolVLAConfig())

    assert policy._make_fixed_inference_noise(torch.zeros(1, 4)) is None
