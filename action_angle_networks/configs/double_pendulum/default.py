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

"""Default hyperparameter configuration."""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Returns a training configuration."""
    config = ml_collections.ConfigDict()
    config.rng_seed = 0
    config.num_trajectories = 1
    config.dimensions_per_trajectory = 2
    config.single_step_predictions = True
    config.num_samples = 1000
    config.split_on = "times"
    config.train_split_proportion = 100 / 1000
    config.test_split_proportion = 500 / 1000
    config.time_delta = 1.0
    config.train_time_jump_schedule = "linear"
    config.train_time_jump_range = (1, 10)
    config.test_time_jumps = (1, 2, 5, 10, 20, 50)
    config.num_train_steps = 5000
    config.eval_cadence = 50
    config.scaler = "standard"
    config.simulation = "double_pendulum"
    config.regularizations = ml_collections.ConfigDict()
    config.simulation_parameter_ranges = ml_collections.ConfigDict(
        {
            "l_1": (0.5,),
            "l_2": (0.5,),
            "m_1": (0.5,),
            "m_2": (0.5,),
            "theta_1_init": (0.5,),
            "theta_2_init": (0.5,),
        }
    )
    return config
