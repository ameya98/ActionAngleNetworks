"""Scripts for loading and analyzing trained models."""
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import yaml

from absl import logging
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
    workdir: str,
    default_config: Optional[str] = None,
) -> Tuple[
    ml_collections.ConfigDict, scalers.Scaler, train_state.TrainState, Dict[Any, Any]
]:
    """Loads the scaler, model and auxiliary data from the supplied workdir."""

    if not os.path.exists(workdir):
        raise FileNotFoundError(f"{workdir} does not exist.")

    # Load config.
    saved_config_path = os.path.join(workdir, "config.yml")
    if os.path.exists(saved_config_path):
        logging.info("Saved config found. Loading...")
        with open(saved_config_path, "r") as config_file:
            config = yaml.unsafe_load(config_file)
        assert config is not None
    else:
        logging.info(
            f"No saved config found. Using default config: %s.", default_config
        )
        if default_config is None:
            raise ValueError("Please supply a value for default_config.")
        config = _ALL_CONFIGS[default_config]

    logging.info("Using config: %s", config)

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


def get_performance_against_steps(
    workdir: Sequence[str],
) -> Tuple[
    Dict[chex.Numeric, chex.Numeric], Dict[chex.Numeric, chex.Numeric], chex.Array
]:
    """Returns test performance against number of training steps for a single experiment."""

    config, scaler, _, aux = load_from_workdir(workdir)

    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    test_simulation_parameters = aux["test"]["simulation_parameters"]
    all_test_metrics = aux["test"]["metrics"]
    steps = all_test_metrics.keys()

    true_position, true_momentum = train.inverse_transform_with_scaler(
        test_positions[:1], test_momentums[:1], scaler
    )
    actual_hamiltonian = train.get_compute_hamiltonian_fn(config)(
        true_position, true_momentum, test_simulation_parameters
    )
    actual_hamiltonian = np.asarray(actual_hamiltonian).squeeze()

    prediction_losses = {
        jump: np.asarray(
            [all_test_metrics[step][jump]["prediction_loss"] for step in steps]
        )
        for jump in config.test_time_jumps
    }
    delta_hamiltonians = {
        jump: np.asarray(
            [
                all_test_metrics[step][jump]["mean_change_in_hamiltonians"]
                for step in steps
            ]
        )
        / actual_hamiltonian
        for jump in config.test_time_jumps
    }

    return prediction_losses, delta_hamiltonians, steps


def get_performance_against_samples(
    workdirs: Sequence[str],
) -> Tuple[
    Dict[chex.Numeric, Dict[chex.Numeric, chex.Numeric]],
    Dict[chex.Numeric, Dict[chex.Numeric, chex.Numeric]],
]:
    """Returns test performance against number of training samples for a list of experiments."""

    all_prediction_losses = {}
    all_delta_hamiltonians = {}
    all_actual_hamiltonians = {}

    for workdir in workdirs:
        config, scaler, state, aux = load_from_workdir(workdir)

        test_positions = aux["test"]["positions"]
        test_momentums = aux["test"]["momentums"]
        test_simulation_parameters = aux["test"]["simulation_parameters"]
        all_test_metrics = aux["test"]["metrics"]

        true_position, true_momentum = train.inverse_transform_with_scaler(
            test_positions[:1], test_momentums[:1], scaler
        )
        actual_hamiltonian = train.get_compute_hamiltonian_fn(config)(
            true_position, true_momentum, test_simulation_parameters
        )
        actual_hamiltonian = np.asarray(actual_hamiltonian).squeeze()

        best_step = state.step - 1
        prediction_losses = {
            jump: np.asarray(all_test_metrics[best_step][jump]["prediction_loss"])
            for jump in config.test_time_jumps
        }
        delta_hamiltonians = {
            jump: np.asarray(
                all_test_metrics[best_step][jump]["mean_change_in_hamiltonians"]
            )
            / actual_hamiltonian
            for jump in config.test_time_jumps
        }

        num_training_samples = config.train_split_proportion * config.num_samples
        all_prediction_losses[num_training_samples] = prediction_losses
        all_delta_hamiltonians[num_training_samples] = delta_hamiltonians
        all_actual_hamiltonians[num_training_samples] = actual_hamiltonian

    all_actual_hamiltonians_values = list(all_actual_hamiltonians.values())
    if not np.allclose(
        all_actual_hamiltonians_values, all_actual_hamiltonians_values[0]
    ):
        raise ValueError("Actual Hamiltonians not all equal.")

    return all_prediction_losses, all_delta_hamiltonians
