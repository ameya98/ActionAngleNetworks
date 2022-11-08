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

"""Training code for the MD17 dataset."""

import functools
import os
from typing import Callable, Dict, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
import yaml

from absl import logging
from clu import checkpoint, metric_writers, parameter_overview
from flax.training import train_state

from action_angle_networks import models, scalers, train
from action_angle_networks.simulation import md17_simulation


def update_config(config: ml_collections.ConfigDict, **kwargs) -> None:
    """Updates the config with the computed parameters."""
    config = config.unlock()
    config.update(**kwargs)
    config.lock()


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> Tuple[chex.Array, chex.Array, Dict[str, Dict[str, chex.Array]]]:
    """Performs training and evaluation with the given configuration."""
    # Update config.
    update_config(config)

    # Set up logging.
    summary_writer = metric_writers.create_default_writer(workdir)
    summary_writer.write_hparams(config.to_dict())

    time_delta = config.time_delta
    test_time_jumps = config.test_time_jumps
    num_trajectories = config.num_trajectories
    num_samples = config.num_samples
    train_split_proportion = config.train_split_proportion
    test_split_proportion = config.test_split_proportion
    num_train_steps = config.num_train_steps
    regularizations = config.regularizations.to_dict()
    eval_cadence = config.eval_cadence

    # Generate data.
    logging.info("Loading data...")
    rng = jax.random.PRNGKey(config.rng_seed)
    simulation_parameters = {}
    compute_hamiltonian_fn = lambda positions, momentums, simulation_parameters: 0.0
    all_positions, all_momentums, _ = md17_simulation.load_trajectory(
        config.molecule, num_samples, resample=True
    )

    # Update config with the number of trajectories.
    # This is molecule dependent.
    num_trajectories = all_positions.shape[1]
    update_config(config, num_trajectories=num_trajectories)

    # Reshape coordinates.
    print(all_positions.shape, all_momentums.shape)
    all_positions = all_positions.reshape((num_samples, -1))
    all_momentums = all_momentums.reshape((num_samples, -1))

    assert (
        len(all_positions.shape) == 2
    ), f"Received all_positions of shape: {all_positions.shape}"
    assert (
        len(all_momentums.shape) == 2
    ), f"Received all_momentums of shape: {all_momentums.shape}"

    # Train-test split.
    if config.split_on == "times":
        assert train_split_proportion + test_split_proportion <= 1

        num_train_samples = int(num_samples * train_split_proportion)
        num_test_samples = int(num_samples * test_split_proportion)

        train_positions = all_positions[:num_train_samples]
        test_positions = all_positions[-num_test_samples:]
        train_momentums = all_momentums[:num_train_samples]
        test_momentums = all_momentums[-num_test_samples:]

        train_simulation_parameters = simulation_parameters
        test_simulation_parameters = simulation_parameters
    else:
        raise ValueError(f"Unsupported feature for split: {config.split_on}.")

    # Create scaler to normalize data.
    scaler = train.create_scaler(config)
    scaler = train.fit_scaler(train_positions, train_momentums, scaler)
    train_positions, train_momentums = train.transform_with_scaler(
        train_positions, train_momentums, scaler
    )
    test_positions, test_momentums = train.transform_with_scaler(
        test_positions, test_momentums, scaler
    )

    # Initialize model.
    logging.info("Constructing model.")
    rng, state_rng = jax.random.split(rng)
    state = train.create_train_state(
        config, state_rng, (train_positions[:1], train_momentums[:1], time_delta)
    )
    best_state = state
    parameter_overview.log_parameter_overview(state.params)

    # Setup for coordinates and time deltas.
    coordinates_fn = train.get_coordinates_fn(config)
    time_deltas_fn = train.get_time_deltas_fn(config)

    # Setup sampling for time jumps.
    sample_time_jump_fn = train.get_sample_time_jump_fn(config)

    # Set up checkpointing of the model.
    # We only save the model, and never load.
    # Thus, training always begins from scratch.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)

    # Save the config for reproducibility.
    config_path = os.path.join(workdir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    min_train_loss = jnp.inf
    all_train_metrics = {}
    all_test_metrics = {}

    # Start training!
    logging.info("Starting training.")
    for step in range(num_train_steps):
        step_rng = jax.random.fold_in(rng, step)

        # Sample time jump.
        step_rng, jump_rng = jax.random.split(step_rng)
        jump = sample_time_jump_fn(step, jump_rng)
        # Setup inputs and targets on all trajectories.
        (
            train_curr_positions,
            train_curr_momentums,
            train_target_positions,
            train_target_momentums,
        ) = coordinates_fn(train_positions, train_momentums, jump)
        time_deltas = time_deltas_fn(jump)

        # Sample indices.
        num_samples_on_trajectory = train_curr_positions.shape[0]
        batch_size = min(config.batch_size, num_samples_on_trajectory)
        sample_indices = jax.random.choice(
            step_rng, num_samples_on_trajectory, (batch_size,)
        )
        batch_curr_positions = train_curr_positions[sample_indices]
        batch_curr_momentums = train_curr_momentums[sample_indices]
        batch_target_positions = train_target_positions[sample_indices]
        batch_target_momentums = train_target_momentums[sample_indices]

        # Update parameters.
        grads = train.compute_updates(
            state,
            batch_curr_positions,
            batch_curr_momentums,
            time_deltas,
            batch_target_positions,
            batch_target_momentums,
            regularizations,
        )
        state = state.apply_gradients(grads=grads)

        # Indicate that training is happening.
        logging.log_first_n(logging.INFO, "Finished training step %d.", 10, step)

        # Evaluate, if required.
        is_last_step = step == num_train_steps - 1
        if step % eval_cadence == (eval_cadence - 1) or is_last_step:
            train_metrics = train.compute_metrics_helper(
                state,
                train_positions,
                train_momentums,
                jump,
                time_delta,
                scaler,
                compute_hamiltonian_fn,
                train_simulation_parameters,
                regularizations,
            )
            train.log_metrics(step, train_metrics, summary_writer, prefix="train_")
            all_train_metrics[step] = train_metrics

            test_metrics = {}
            for test_jump in test_time_jumps:
                test_metrics[test_jump] = train.compute_metrics_helper(
                    state,
                    test_positions,
                    test_momentums,
                    test_jump,
                    time_delta,
                    scaler,
                    compute_hamiltonian_fn,
                    test_simulation_parameters,
                    regularizations,
                )
                train.log_metrics(
                    step,
                    test_metrics[test_jump],
                    summary_writer,
                    prefix=f"test_jump_{test_jump}_",
                )
            all_test_metrics[step] = test_metrics

            # Save best state seen so far.
            if train_metrics["total_loss"] < min_train_loss:
                min_train_loss = train_metrics["total_loss"]
                best_state = state

    auxiliary_data = {
        "train": {
            "positions": train_positions,
            "momentums": train_momentums,
            "simulation_parameters": train_simulation_parameters,
            "metrics": all_train_metrics,
        },
        "test": {
            "positions": test_positions,
            "momentums": test_momentums,
            "simulation_parameters": test_simulation_parameters,
            "metrics": all_test_metrics,
        },
    }

    # Save everything we computed during training.
    ckpt.save(
        {
            "scaler": scaler,
            "best_state": best_state,
            "auxiliary_data": auxiliary_data,
        }
    )
    return scaler, best_state, auxiliary_data
