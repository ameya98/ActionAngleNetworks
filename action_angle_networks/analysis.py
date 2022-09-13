"""Scripts for loading and analyzing trained models."""
import os
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_collections
import yaml
from clu import checkpoint
from flax.training import train_state

from action_angle_networks import scalers, train
from action_angle_networks.configs.harmonic_motion import (
    action_angle_flow,
    action_angle_mlp,
    euler_update_flow,
    euler_update_mlp,
)

_ALL_CONFIGS = {
    "action_angle_flow": action_angle_flow.get_config(),
    "action_angle_mlp": action_angle_mlp.get_config(),
    "euler_update_flow": euler_update_flow.get_config(),
    "euler_update_mlp": euler_update_mlp.get_config(),
}


def cast_keys_as_int(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """Returns a dictionary with string keys converted to integers, wherever possible."""
    casted_dictionary = {}
    for key, val in dictionary.items():
        try:
            val = cast_keys_as_int(val)
        except AttributeError:
            pass

        try:
            key = int(key)
        except ValueError:
            pass
        finally:
            casted_dictionary[key] = val
    return casted_dictionary


def load_from_workdir(
    workdir: str, default_config: Optional[str] = None
) -> Tuple[
    ml_collections.ConfigDict, scalers.Scaler, train_state.TrainState, Dict[Any, Any]
]:
    """Loads the scaler, model and auxiliary data from the supplied workdir."""

    if not os.path.exists(workdir):
        raise FileNotFoundError(f"{workdir} does not exist.")

    # Load config.
    saved_config_path = os.path.join(workdir, "config.yml")
    if os.path.exists(saved_config_path):
        print("Saved config found. Loading...")
        with open(saved_config_path, "r") as config_file:
            config = yaml.unsafe_load(config_file)
        assert config is not None
    else:
        print(f"No saved config found. Using default config: {default_config}.")
        if default_config is None:
            raise ValueError("Please supply a value for default_config.")
        config = _ALL_CONFIGS[default_config]
    print(config)

    # Mimic what we do in train.py.
    # To be honest, this doesn't really matter right now,
    # because we only use the structure of dummy_state.
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, _ = jax.random.split(rng)
    rng, state_rng = jax.random.split(rng)

    # Set up dummy variables to obtain the structure.
    dummy_scaler = train.create_scaler(config)
    dummy_state = train.create_train_state(
        config,
        state_rng,
        (
            jnp.zeros((1, config.num_trajectories)),
            jnp.zeros((1, config.num_trajectories)),
            0.0,
        ),
    )

    # Load the actual values.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)
    data = ckpt.restore(
        {"scaler": dummy_scaler, "best_state": dummy_state, "auxiliary_data": None}
    )
    return (
        config,
        data["scaler"],
        data["best_state"],
        cast_keys_as_int(data["auxiliary_data"]),
    )
