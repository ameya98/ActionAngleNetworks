"""Scripts for loading and analyzing trained models."""
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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
    default_config: Optional[ml_collections.ConfigDict] = None,
) -> Tuple[
    ml_collections.ConfigDict, scalers.Scaler, train_state.TrainState, Dict[Any, Any]
]:
    """Loads the scaler, model and auxiliary data from the supplied workdir."""

    if not os.path.exists(workdir):
        raise FileNotFoundError(f"{workdir} does not exist.")

    # Load config.
    config = default_config
    saved_config_path = os.path.join(workdir, "config.yml")
    if os.path.exists(saved_config_path):
        logging.info("Saved config found. Loading...")

        if default_config is None:
            config = ml_collections.ConfigDict()
        else:
            config = default_config

        with open(saved_config_path, "r") as config_file:
            loaded_config = yaml.unsafe_load(config_file)
        config.update(loaded_config)

        assert config is not None
    else:
        logging.info(
            f"No saved config found. Using default config: %s.", default_config
        )
        if default_config is None:
            raise ValueError("Please supply a value for default_config.")
        config = default_config

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
    workdirs: Sequence[str], log_num_parameters: bool = True
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

        if log_num_parameters:
            num_parameters = sum(
                jax.tree_leaves(jax.tree_map(lambda arr: arr.size, state.params))
            )
            logging.info("workdir: %s, num_parameters: %d", workdir, num_parameters)

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


def get_performance_against_parameters(
    workdirs: Sequence[str],
) -> Tuple[
    Dict[chex.Numeric, Dict[chex.Numeric, chex.Numeric]],
    Dict[chex.Numeric, Dict[chex.Numeric, chex.Numeric]],
]:
    """Returns test performance against number of parameters for a list of experiments."""

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

        num_parameters = sum(
            jax.tree_leaves(jax.tree_map(lambda arr: arr.size, state.params))
        )
        all_prediction_losses[num_parameters] = prediction_losses
        all_delta_hamiltonians[num_parameters] = delta_hamiltonians
        all_actual_hamiltonians[num_parameters] = actual_hamiltonian

    all_actual_hamiltonians_values = list(all_actual_hamiltonians.values())
    if not np.allclose(
        all_actual_hamiltonians_values, all_actual_hamiltonians_values[0]
    ):
        raise ValueError("Actual Hamiltonians not all equal.")

    return all_prediction_losses, all_delta_hamiltonians


def _measure_execution_time(func: Callable[[Any], Any], *args, **kwargs):
    """Measures the time taken to execute a jittable function."""
    # Wrap in JIT.
    jitted_func = jax.jit(func)

    # Call the function once so that we don't measure JAX tracing time.
    jitted_func(*args, **kwargs)

    # Now measure the actual time taken.
    start_time = time.time()
    jax.block_until_ready(jitted_func(*args, **kwargs))
    return time.time() - start_time


def get_inference_times(workdir: str) -> Dict[chex.Numeric, chex.Numeric]:
    """Returns the inference time for different jump sizes."""

    config, _, state, aux = load_from_workdir(workdir)
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]

    inference_times = {}
    for jump in config.test_time_jumps:
        inference_times[jump] = _measure_execution_time(
            state.apply_fn,
            state.params,
            test_positions,
            test_momentums,
            jump * config.time_delta,
        )
    return inference_times


def get_performance_against_time(workdir: str) -> Dict[int, chex.Numeric]:
    """Returns errors as a function of time from an initial point."""

    config, _, state, aux = load_from_workdir(workdir)
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]

    errors = {}
    jumps = [1, 2, 5, 10, 20, 50, 100, 200, 400]
    for jump in jumps:
        (
            curr_positions,
            curr_momentums,
            target_positions,
            target_momentums,
        ) = train.get_coordinates_for_time_jump(test_positions, test_momentums, jump)
        curr_positions = curr_positions[:1]
        curr_momentums = curr_momentums[:1]
        target_positions = target_positions[:1]
        target_momentums = target_momentums[:1]
        (predicted_positions, predicted_momentums, _,) = train.compute_predictions(
            state, curr_positions, curr_momentums, jump * config.time_delta
        )
        errors[jump] = train.compute_loss(
            predicted_positions,
            predicted_momentums,
            target_positions,
            target_momentums,
            time_deltas=config.time_delta,
        )
    return errors


def get_true_trajectories(workdir: str, jump: int) -> Dict[int, chex.Numeric]:
    """Returns true trajectories."""

    _, scaler, _, aux = load_from_workdir(workdir)
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    (
        _,
        _,
        target_positions,
        target_momentums,
    ) = train.get_coordinates_for_time_jump(test_positions, test_momentums, jump)
    return train.inverse_transform_with_scaler(
        target_positions, target_momentums, scaler
    )


def get_predicted_trajectories(workdir: str, jump: int) -> Dict[int, chex.Numeric]:
    """Returns predicted trajectories."""

    config, scaler, state, aux = load_from_workdir(workdir)
    test_positions = aux["test"]["positions"]
    test_momentums = aux["test"]["momentums"]
    (
        curr_positions,
        curr_momentums,
        *_,
    ) = train.get_coordinates_for_time_jump(test_positions, test_momentums, jump)
    (predicted_positions, predicted_momentums, _,) = train.compute_predictions(
        state, curr_positions, curr_momentums, jump * config.time_delta
    )
    return train.inverse_transform_with_scaler(
        predicted_positions, predicted_momentums, scaler
    )
