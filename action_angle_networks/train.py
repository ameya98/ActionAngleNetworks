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

"""Training code."""

import functools
import os
from typing import Callable, Dict, Optional, Tuple

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from absl import logging
from clu import checkpoint, metric_writers, parameter_overview
from flax.training import train_state

from action_angle_networks import (
    harmonic_motion_simulation,
    models,
    orbit_simulation,
    scalers,
)


def get_generate_canonical_coordinates_fn(
    config: ml_collections.ConfigDict,
) -> Callable[[chex.Array, Dict[str, chex.Array]], Tuple[chex.Array, chex.Array]]:
    """Returns a function that creates trajectories of shape [num_samples, num_trajectories]."""
    if config.simulation == "harmonic":
        return jax.jit(
            jax.vmap(
                harmonic_motion_simulation.generate_canonical_coordinates,
                in_axes=(0, None),
                out_axes=0,
            )
        )

    if config.simulation == "orbit":
        return jax.jit(
            jax.vmap(
                orbit_simulation.generate_canonical_coordinates,
                in_axes=(0, None),
                out_axes=0,
            )
        )

    raise ValueError(f"Unsupported simulation: {config.simulation}.")


def get_compute_hamiltonian_fn(
    config: ml_collections.ConfigDict,
) -> Callable[[chex.Array, chex.Array, Dict[str, chex.Array]], chex.Array]:
    """Returns a function that computes the Hamiltonian over trajectories of shape [num_samples, num_trajectories]."""
    if config.simulation == "harmonic":
        return jax.jit(
            jax.vmap(
                harmonic_motion_simulation.compute_hamiltonian, in_axes=(0, 0, None)
            )
        )

    if config.simulation == "orbit":
        return jax.jit(
            jax.vmap(orbit_simulation.compute_hamiltonian, in_axes=(0, 0, None))
        )

    raise ValueError(f"Unsupported simulation: {config.simulation}.")


def get_sample_simulation_parameters_fn(config: ml_collections.ConfigDict):
    """Returns a function that samples simulation parameters."""
    if config.simulation == "harmonic":
        return harmonic_motion_simulation.sample_simulation_parameters

    if config.simulation == "orbit":
        return orbit_simulation.sample_simulation_parameters

    raise ValueError(f"Unsupported simulation: {config.simulation}.")


def create_scaler(config: ml_collections.ConfigDict) -> scalers.Scaler:
    """Constructs the scaler for normalizing the data."""
    if config.scaler == "standard":
        return scalers.StandardScaler()
    if config.scaler == "identity":
        return scalers.IdentityScaler()
    raise ValueError(f"Unsupported scaler: {config.scaler}.")


def create_model(config: ml_collections.ConfigDict) -> nn.Module:
    """Creates the model."""
    activation = getattr(jax.nn, config.activation, None)
    latent_size = config.latent_size

    if config.encoder_decoder_type == "mlp":
        encoder = models.MLPEncoder(
            position_encoder=models.MLP(
                [latent_size], activation, skip_connections=False
            ),
            momentum_encoder=models.MLP(
                [latent_size], activation, skip_connections=False
            ),
            transform_fn=models.MLP([latent_size], activation, skip_connections=True),
            latent_position_decoder=models.MLP(
                [latent_size, 1], activation, skip_connections=True
            ),
            latent_momentum_decoder=models.MLP(
                [latent_size, 1], activation, skip_connections=True
            ),
            name="encoder",
        )
        decoder = models.MLPDecoder(
            latent_position_encoder=models.MLP(
                [latent_size], activation, skip_connections=False
            ),
            latent_momentum_encoder=models.MLP(
                [latent_size], activation, skip_connections=False
            ),
            transform_fn=models.MLP([latent_size], activation, skip_connections=True),
            position_decoder=models.MLP(
                [latent_size, 1], activation, skip_connections=True
            ),
            momentum_decoder=models.MLP(
                [latent_size, 1], activation, skip_connections=True
            ),
            name="decoder",
        )
    if config.encoder_decoder_type == "flow":
        flow = models.create_flow(
            config, init_shape=(config.batch_size, 2 * config.num_trajectories)
        )
        encoder = models.FlowEncoder(flow)
        decoder = models.FlowDecoder(flow)

    # Action-Angle Neural Network.
    if config.model == "action-angle-network":
        return models.ActionAngleNetwork(
            encoder=encoder,
            angular_velocity_net=models.MLP(
                [latent_size, config.num_trajectories],
                activation,
                skip_connections=True,
                name="angular_velocity_net",
            ),
            decoder=decoder,
            polar_action_angles=config.polar_action_angles,
            single_step_predictions=config.single_step_predictions,
        )

    # Baseline Euler Update Network.
    if config.model == "euler-update-network":
        return models.EulerUpdateNetwork(
            encoder=encoder,
            derivative_net=models.MLP(
                [latent_size, latent_size, latent_size, 2],
                activation,
                skip_connections=True,
                name="derivative",
            ),
            decoder=decoder,
        )

    raise ValueError("Unsupported model.")


def create_train_state(
    config: ml_collections.ConfigDict,
    rng: chex.PRNGKey,
    init_samples: chex.Array,
) -> train_state.TrainState:
    """Creates the training state."""
    model = create_model(config)
    params = model.init(rng, *init_samples)
    tx = optax.adam(learning_rate=config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.jit
def compute_predictions(
    state: train_state.TrainState,
    positions: chex.Array,
    momentums: chex.Array,
    time_deltas: chex.Numeric,
) -> Tuple[chex.Array, chex.Array, Optional[Dict[str, chex.Array]]]:
    """Computes model predictions."""
    return state.apply_fn(state.params, positions, momentums, time_deltas)


@jax.jit
def compute_loss(
    predicted_positions: chex.Array,
    predicted_momentums: chex.Array,
    target_positions: chex.Array,
    target_momentums: chex.Array,
    auxiliary_predictions: Optional[Dict[str, chex.Array]] = None,
    regularizations: Optional[Dict[str, chex.Array]] = None,
) -> chex.Numeric:
    """Computes the loss for the given predictions."""
    assert (
        predicted_positions.shape == target_positions.shape
    ), f"Got predicted_positions: {predicted_positions.shape}, target_positions: {target_positions.shape}"
    assert (
        predicted_momentums.shape == target_momentums.shape
    ), f"Got predicted_momentums: {predicted_momentums.shape}, target_momentums: {target_momentums.shape}"

    loss = optax.l2_loss(predictions=predicted_positions, targets=target_positions)
    loss += optax.l2_loss(predictions=predicted_momentums, targets=target_momentums)
    loss = jnp.mean(loss)

    if auxiliary_predictions is not None:
        angular_velocities = auxiliary_predictions["angular_velocities"]
        angular_velocities_variances = jnp.var(angular_velocities, axis=0).sum()
        loss += regularizations["angular_velocities"] * angular_velocities_variances

        actions = auxiliary_predictions["actions"]
        actions_variances = jnp.var(actions, axis=0).sum()
        loss += regularizations["actions"] * actions_variances
    return loss


@jax.jit
def compute_updates(
    state: train_state.TrainState,
    positions: chex.Array,
    momentums: chex.Array,
    time_deltas: chex.Numeric,
    target_positions: chex.Array,
    target_momentums: chex.Array,
    regularizations: Dict[str, chex.Numeric],
    compute_encoded_decoded_loss: bool = False,
) -> optax.Updates:
    """Computes gradients for a single batch."""

    def loss_fn(params):
        curr_state = state.replace(params=params)
        (
            predicted_positions,
            predicted_momentums,
            auxiliary_predictions,
        ) = compute_predictions(curr_state, positions, momentums, time_deltas)
        regularized_prediction_loss = compute_loss(
            predicted_positions,
            predicted_momentums,
            target_positions,
            target_momentums,
            auxiliary_predictions,
            regularizations,
        )
        total_loss = regularized_prediction_loss

        if compute_encoded_decoded_loss:
            (
                encoded_decoded_positions,
                encoded_decoded_momentums,
                _,
            ) = compute_predictions(curr_state, positions, momentums, 0.0)
            encoded_decoded_loss = compute_loss(
                encoded_decoded_positions,
                encoded_decoded_momentums,
                positions,
                momentums,
            )
            total_loss += (
                regularizations["encoded_decoded_differences"] * encoded_decoded_loss
            )

        return total_loss

    return jax.grad(loss_fn)(state.params)


@functools.partial(jax.jit, static_argnames="compute_hamiltonian_fn")
def compute_mean_change_in_hamiltonians(
    curr_positions: chex.Array,
    curr_momentums: chex.Array,
    predicted_positions: chex.Array,
    predicted_momentums: chex.Array,
    compute_hamiltonian_fn: Callable[
        [chex.Array, chex.Array, Dict[str, chex.Array]], chex.Array
    ],
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Numeric:
    """Computes the mean change in Hamiltonian for current coordinates versus predicted coordinates."""
    curr_hamiltonians = compute_hamiltonian_fn(
        curr_positions, curr_momentums, simulation_parameters
    )
    predicted_hamiltonians = compute_hamiltonian_fn(
        predicted_positions, predicted_momentums, simulation_parameters
    )
    return jnp.mean(jnp.abs(curr_hamiltonians - predicted_hamiltonians))


@jax.jit
def compute_metrics(
    predicted_positions: chex.Array,
    predicted_momentums: chex.Array,
    target_positions: chex.Array,
    target_momentums: chex.Array,
    current_positions: chex.Array,
    current_momentums: chex.Array,
    encoded_decoded_positions: chex.Array,
    encoded_decoded_momentums: chex.Array,
    auxiliary_predictions: chex.Array,
    regularizations: chex.Array,
    compute_encoded_decoded_loss: bool = False,
) -> Dict[str, chex.Numeric]:
    """Computes loss and other metrics."""
    prediction_loss = compute_loss(
        predicted_positions, predicted_momentums, target_positions, target_momentums
    )
    regularized_prediction_loss = compute_loss(
        predicted_positions,
        predicted_momentums,
        target_positions,
        target_momentums,
        auxiliary_predictions,
        regularizations,
    )
    total_loss = regularized_prediction_loss
    if compute_encoded_decoded_loss:
        encoded_decoded_loss = compute_loss(
            encoded_decoded_positions,
            encoded_decoded_momentums,
            current_positions,
            current_momentums,
        )
        total_loss += (
            regularizations["encoded_decoded_differences"] * encoded_decoded_loss
        )
    return {
        "prediction_loss": prediction_loss,
        "regularized_prediction_loss": regularized_prediction_loss,
        "total_loss": total_loss,
    }


@functools.partial(jax.jit, static_argnames=["jump", "compute_hamiltonian_fn"])
def compute_metrics_helper(
    state: train_state.TrainState,
    positions: chex.Array,
    momentums: chex.Array,
    jump: int,
    time_delta: float,
    scaler: scalers.Scaler,
    compute_hamiltonian_fn: Callable[
        [chex.Array, chex.Array, Dict[str, chex.Array]], chex.Array
    ],
    simulation_parameters: Dict[str, chex.Array],
    regularizations: Dict[str, float],
) -> Dict[str, chex.Numeric]:
    """Helper for compute_metrics that evaluates the current training state."""
    (
        curr_positions,
        curr_momentums,
        target_positions,
        target_momentums,
    ) = get_coordinates_for_time_jump(positions, momentums, jump)
    (
        predicted_positions,
        predicted_momentums,
        auxiliary_predictions,
    ) = compute_predictions(state, curr_positions, curr_momentums, jump * time_delta)
    encoded_decoded_positions, encoded_decoded_momentums, _ = compute_predictions(
        state, curr_positions, curr_momentums, 0
    )
    metrics = compute_metrics(
        predicted_positions,
        predicted_momentums,
        target_positions,
        target_momentums,
        curr_positions,
        curr_momentums,
        encoded_decoded_positions,
        encoded_decoded_momentums,
        auxiliary_predictions,
        regularizations,
    )

    # Rescale to original units before computing Hamiltonian.
    curr_positions, curr_momentums = inverse_transform_with_scaler(
        curr_positions, curr_momentums, scaler
    )
    predicted_positions, predicted_momentums = inverse_transform_with_scaler(
        predicted_positions, predicted_momentums, scaler
    )
    metrics["mean_change_in_hamiltonians"] = compute_mean_change_in_hamiltonians(
        curr_positions,
        curr_momentums,
        predicted_positions,
        predicted_momentums,
        compute_hamiltonian_fn,
        simulation_parameters,
    )

    return metrics


def log_metrics(
    step: int,
    metrics: Dict[str, float],
    summary_writer: metric_writers.MetricWriter,
    prefix: str = "",
) -> None:
    """Logs all metrics."""
    # Formatting for accuracy.
    for metric in metrics:
        if "accuracy" in metric:
            metrics[metric] *= 100

    # Add prefix to all metric names.
    metrics = {prefix + metric: metric_val for metric, metric_val in metrics.items()}

    for metric, metric_val in metrics.items():
        logging.info("num_steps: % 3d, %s: %.4f", step, metric, metric_val)

    summary_writer.write_scalars(step, metrics)
    summary_writer.flush()


@jax.jit
def fit_scaler(
    positions: chex.Array, momentums: chex.Array, scaler: scalers.Scaler
) -> scalers.Scaler:
    """Fits the scaler."""
    if positions.ndim == 1:
        positions = positions[jnp.newaxis, ...]
        momentums = momentums[jnp.newaxis, ...]

    assert positions.ndim == 2, f"Got positions of shape {positions.shape}."
    assert momentums.ndim == 2, f"Got momentums of shape {momentums.shape}."

    coords = jnp.concatenate((positions, momentums), axis=1)
    return scaler.fit(coords)


@jax.jit
def transform_with_scaler(
    positions: chex.Array, momentums: chex.Array, scaler: scalers.Scaler
) -> Tuple[chex.Array, chex.Array]:
    """Performs the rescaling to normalized coordinates."""
    if positions.ndim == 1:
        positions = positions[jnp.newaxis, ...]
        momentums = momentums[jnp.newaxis, ...]

    assert positions.ndim == 2, f"Got positions of shape {positions.shape}."
    assert momentums.ndim == 2, f"Got momentums of shape {momentums.shape}."

    coords = jnp.concatenate((positions, momentums), axis=1)
    coords = scaler.transform(coords)
    num_positions = coords.shape[1] // 2
    positions = coords[:, :num_positions]
    momentums = coords[:, num_positions:]
    return positions, momentums


@jax.jit
def inverse_transform_with_scaler(
    positions: chex.Array, momentums: chex.Array, scaler: scalers.Scaler
) -> Tuple[chex.Array, chex.Array]:
    """Performs the inverse rescaling to unnormalized coordinates."""
    if positions.ndim == 1:
        positions = positions[jnp.newaxis, ...]
        momentums = momentums[jnp.newaxis, ...]

    assert positions.ndim == 2, f"Got positions of shape {positions.shape}."
    assert momentums.ndim == 2, f"Got momentums of shape {momentums.shape}."

    coords = jnp.concatenate((positions, momentums), axis=1)
    coords = scaler.inverse_transform(coords)
    num_positions = coords.shape[1] // 2
    positions = coords[:, :num_positions]
    momentums = coords[:, num_positions:]
    return positions, momentums


@functools.partial(jax.jit, static_argnames="jump")
def get_coordinates_for_time_jump(
    positions: chex.Array, momentums: chex.Array, jump: int
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Returns the current and target coordinates for the given jump."""
    # Input arrays are of shape [num_samples, num_trajectories].
    assert positions.ndim == 2, f"Got positions of shape {positions.shape}."
    assert momentums.ndim == 2, f"Got momentums of shape {momentums.shape}."

    curr_positions, target_positions = positions[:-jump], positions[jump:]
    curr_momentums, target_momentums = momentums[:-jump], momentums[jump:]
    return curr_positions, curr_momentums, target_positions, target_momentums


@functools.partial(jax.jit, static_argnames="max_jump")
def get_coordinates_for_time_jumps(
    positions: chex.Array, momentums: chex.Array, max_jump: int
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array]:
    """Returns the current and target coordinates for jumps upto max_jump."""
    # Input arrays are of shape [num_samples, num_trajectories].
    assert positions.ndim == 2, f"Got positions of shape {positions.shape}."
    assert momentums.ndim == 2, f"Got momentums of shape {momentums.shape}."

    def map_to_target_indices(index):
        return index + jnp.arange(1, max_jump + 1)

    num_samples = positions.shape[1]
    curr_indices = jnp.arange(0, num_samples - max_jump)
    target_indices = jax.vmap(map_to_target_indices)(curr_indices)
    curr_positions = positions[curr_indices]
    target_positions = positions[target_indices]
    curr_momentums = momentums[curr_indices]
    target_momentums = momentums[target_indices]
    return curr_positions, curr_momentums, target_positions, target_momentums


def get_coordinates_fn(
    config: ml_collections.ConfigDict,
) -> Callable[
    [chex.Array, chex.Array, int], Tuple[chex.Array, chex.Array, chex.Array, chex.Array]
]:
    """Returns a function that sets up coordinates."""
    # Setup for one-step predictions.
    if config.single_step_predictions:
        return get_coordinates_for_time_jump

    # Setup for multi-step predictions.
    return get_coordinates_for_time_jumps


def sample_constant_time_jump(step: int, rng: chex.PRNGKey, constant_jump: int) -> int:
    """Returns a constant jump size."""
    del step, rng
    return constant_jump


def sample_time_jump_with_linear_increase(
    step: int,
    rng: chex.PRNGKey,
    num_train_steps: int,
    min_jump: int,
    max_jump: int,
) -> int:
    """Returns a stochastic jump size, with linearly increasing mean."""
    max_time_jump_for_step = min_jump + (step / num_train_steps) * (max_jump - min_jump)
    max_time_jump_for_step = jnp.round(max_time_jump_for_step)
    jump = jax.random.randint(rng, (), min_jump, max_time_jump_for_step + 1)
    jump = int(jump)
    return jump


def get_sample_time_jump_fn(
    config: ml_collections.ConfigDict,
) -> Callable[[int, chex.PRNGKey], int]:
    if config.train_time_jump_schedule == "constant":
        return functools.partial(
            sample_constant_time_jump,
            constant_jump=config.train_time_jump,
        )

    if config.train_time_jump_schedule == "linear":
        min_jump, max_jump = config.train_time_jump_range
        return functools.partial(
            sample_time_jump_with_linear_increase,
            min_jump=min_jump,
            max_jump=max_jump,
            num_train_steps=config.num_train_steps,
        )

    raise ValueError(
        f"Unsupported config.time_jump_schedule: {config.time_jump_schedule}"
    )


def get_time_deltas_fn(config: ml_collections.ConfigDict) -> Callable[[int], float]:
    """Returns a function that sets up time deltas."""
    # Setup for one-step predictions.
    if config.single_step_predictions:
        return lambda jump: jump * config.time_delta

    # Setup for multi-step predictions.
    return lambda jump: jnp.arange(1, jump + 1) * config.time_delta


@functools.partial(jax.jit, static_argnames="index")
def get_trajectory_with_parameters(
    index: int,
    positions: chex.Array,
    momentums: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> Tuple[chex.Array, chex.Array, Dict[str, chex.Array]]:
    """Gets the trajectory and simulation parameters for a particular index."""
    return jax.tree_map(
        lambda arr: arr[index], (positions, momentums, simulation_parameters)
    )


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> Tuple[chex.Array, chex.Array, Dict[str, Dict[str, chex.Array]]]:
    """Performs training and evaluation with the given configuration."""
    # Set up logging.
    summary_writer = metric_writers.create_default_writer(workdir)
    summary_writer.write_hparams(config.to_dict())

    time_delta = config.time_delta
    test_time_jumps = config.test_time_jumps
    num_trajectories = config.num_trajectories
    num_samples = config.num_samples
    train_split_proportion = config.train_split_proportion
    num_train_steps = config.num_train_steps
    regularizations = config.regularizations.to_dict()
    eval_cadence = config.eval_cadence

    # Get simulation functions.
    generate_canonical_coordinates_fn = get_generate_canonical_coordinates_fn(config)
    compute_hamiltonian_fn = get_compute_hamiltonian_fn(config)
    sample_simulation_parameters_fn = get_sample_simulation_parameters_fn(config)

    # Generate data.
    logging.info("Generating data.")
    rng = jax.random.PRNGKey(config.rng_seed)
    rng, simulation_parameters_rng = jax.random.split(rng)
    simulation_parameters = sample_simulation_parameters_fn(
        config.simulation_parameter_ranges.to_dict(),
        num_trajectories,
        simulation_parameters_rng,
    )
    times = jnp.arange(num_samples) * time_delta
    all_positions, all_momentums = generate_canonical_coordinates_fn(
        times, simulation_parameters
    )

    # Train-test split.
    if config.split_on == "times":
        num_train_samples = int(num_samples * train_split_proportion)
        train_positions = all_positions[:num_train_samples]
        test_positions = all_positions[num_train_samples:]
        train_momentums = all_momentums[:num_train_samples]
        test_momentums = all_momentums[num_train_samples:]

        train_simulation_parameters = simulation_parameters
        test_simulation_parameters = simulation_parameters
    else:
        raise ValueError(f"Unsupported feature for split: {config.split_on}.")

    # Rescale.
    scaler = create_scaler(config)
    scaler = fit_scaler(train_positions, train_momentums, scaler)
    train_positions, train_momentums = transform_with_scaler(
        train_positions, train_momentums, scaler
    )
    test_positions, test_momentums = transform_with_scaler(
        test_positions, test_momentums, scaler
    )

    # Initialize model.
    logging.info("Constructing model.")
    state = create_train_state(
        config, rng, (train_positions[:1], train_momentums[:1], time_delta)
    )
    best_state = state
    parameter_overview.log_parameter_overview(state.params)

    # Setup for coordinates and time deltas.
    coordinates_fn = get_coordinates_fn(config)
    time_deltas_fn = get_time_deltas_fn(config)

    # Setup sampling for time jumps.
    sample_time_jump_fn = get_sample_time_jump_fn(config)

    # Set up checkpointing of the model.
    # We only save the model, and never load.
    # Thus, training always begins from scratch.
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpt = checkpoint.Checkpoint(checkpoint_dir, max_to_keep=2)

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
        sample_indices = jax.random.choice(
            step_rng, num_samples_on_trajectory, (config.batch_size,)
        )
        batch_curr_positions = train_curr_positions[sample_indices]
        batch_curr_momentums = train_curr_momentums[sample_indices]
        batch_target_positions = train_target_positions[sample_indices]
        batch_target_momentums = train_target_momentums[sample_indices]

        # Update parameters.
        grads = compute_updates(
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
            train_metrics = compute_metrics_helper(
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
            log_metrics(step, train_metrics, summary_writer, prefix="train_")
            all_train_metrics[step] = train_metrics

            test_metrics = {}
            for test_jump in test_time_jumps:
                test_metrics[test_jump] = compute_metrics_helper(
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
                log_metrics(
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
