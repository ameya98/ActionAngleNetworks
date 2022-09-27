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

"""Neural Ordinary Differential Equations with MLP-based encoder and decoders."""

import ml_collections

from action_angle_networks.configs.harmonic_motion import default


def get_config() -> ml_collections.ConfigDict:
    """Returns a training configuration."""
    config = default.get_config()
    config.model = "neural-ode"
    config.encoder_decoder_type = "mlp"
    config.latent_size = 100
    config.num_derivative_net_layers = 4
    config.num_encoder_layers = 1
    config.num_decoder_layers = 1
    config.activation = "relu"
    config.learning_rate = 1e-3
    config.batch_size = 100
    config.num_train_steps = 50000
    return config
