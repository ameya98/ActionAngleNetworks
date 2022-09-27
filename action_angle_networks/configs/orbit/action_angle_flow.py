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

"""Action-Angle Networks with flow-based encoder and decoders."""

import ml_collections

from action_angle_networks.configs.orbit import default


def get_config() -> ml_collections.ConfigDict:
    """Returns a training configuration."""
    config = default.get_config()
    config.model = "action-angle-network"
    config.encoder_decoder_type = "flow"
    config.latent_size = 100
    config.activation = "sigmoid"
    config.flow_type = "shear"
    config.num_flow_layers = 20
    config.num_angular_velocity_net_layers = 2
    config.num_train_steps = 50000
    if config.flow_type == "masked-coupling":
        config.flow_spline_range_min = -3
        config.flow_spline_range_max = 3
        config.flow_spline_bins = 100
    config.polar_action_angles = True
    config.learning_rate = 1e-3
    config.batch_size = 100
    config.regularizations = ml_collections.ConfigDict(
        {
            "actions": 1.0,
            "angular_velocities": 0.0,
            "encoded_decoded_differences": 0.0,
        }
    )
    return config
