"""Neural Ordinary Differential Equations with MLP-based encoder and decoders."""

import ml_collections

from action_angle_networks.configs.md17 import default


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
