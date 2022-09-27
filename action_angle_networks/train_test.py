# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

"""Tests for train."""

import importlib
import tempfile
from distutils.command.config import config

import ml_collections

from absl.testing import absltest, parameterized

from action_angle_networks import train


def get_config(config_name: str, simulation: str) -> ml_collections.ConfigDict:
    """Returns the appropriate config."""
    config_module = importlib.import_module(
        f"action_angle_networks.configs.{simulation}.{config_name}"
    )
    return config_module.get_config()


class TrainTest(parameterized.TestCase):
    @parameterized.product(
        config_name=[
            "action_angle_flow",
            "action_angle_mlp",
            "euler_update_flow",
            "euler_update_mlp",
            "neural_ode",
            "hamiltonian_neural_network",
        ],
        simulation=["orbit"],
    )
    def test_train_and_evaluate(self, config_name: str, simulation: str):
        # Load config.
        config = get_config(config_name, simulation)
        config.num_train_steps = 5

        # Create a temporary directory where metrics are written.
        workdir = tempfile.mkdtemp()

        # Training should proceed without any errors.
        train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    absltest.main()
