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

"""Simulation of Keplerian orbits."""

from typing import Dict, Mapping, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np

_SIMULATION_PARAMETERS = ["t0", "a", "m", "e", "k"]


def sample_simulation_parameters(
    simulation_parameter_ranges: Mapping[str, Tuple[chex.Numeric, chex.Numeric]],
    num_trajectories: int,
    rng: chex.PRNGKey,
) -> Dict[str, chex.Array]:
    """Samples simulation parameters."""
    # Check that all simulation parameter ranges are available.
    for simulation_parameter in _SIMULATION_PARAMETERS:
        if not simulation_parameter in simulation_parameter_ranges:
            raise ValueError(f"Missing simulation parameter: {simulation_parameter}")

    is_tuple = lambda val: isinstance(val, tuple)
    ranges_flat, ranges_treedef = jax.tree_flatten(
        simulation_parameter_ranges, is_leaf=is_tuple
    )
    rng, shuffle_rng, *rngs = jax.random.split(rng, len(ranges_flat) + 2)
    shuffle_indices = jax.random.permutation(shuffle_rng, num_trajectories)
    rng_tree = jax.tree_unflatten(ranges_treedef, rngs)

    def sample_simulation_parameter(
        simulation_parameter_range: Tuple[chex.Numeric, ...],
        parameter_rng: chex.PRNGKey,
    ):
        """Sample a single simulation parameter."""
        del parameter_rng

        if len(simulation_parameter_range) == 1:
            common_value = simulation_parameter_range[0]
            return jnp.full((num_trajectories,), common_value)

        minval, maxval = simulation_parameter_range
        samples = jnp.linspace(minval, maxval, num=num_trajectories)
        return jnp.sort(samples)[shuffle_indices]

    return jax.tree_map(
        sample_simulation_parameter,
        simulation_parameter_ranges,
        rng_tree,
        is_leaf=is_tuple,
    )


def generate_canonical_coordinates(
    t: chex.Numeric,
    simulation_parameters: Mapping[str, chex.Array],
    check_convergence: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Generates positions and momentums in polar coordinates for all trajectories."""
    coordinates = jax.vmap(
        generate_canonical_coordinates_for_trajectory, in_axes=(None, 0, None)
    )(t, simulation_parameters, check_convergence)
    coordinates = jax.tree_map(lambda arr: arr.squeeze(axis=0), coordinates)
    return coordinates


def generate_canonical_coordinates_for_trajectory(
    t: chex.Numeric,
    simulation_parameters: Mapping[str, chex.Array],
    check_convergence: bool = False,
) -> Tuple[chex.Array, chex.Array]:
    """Generates positions and momentums in polar coordinates for one trajectory."""

    def eccentric_anomaly_to_time(eccentric_anomaly):
        """Maps eccentricity to time."""
        mean_anomaly = eccentric_anomaly - e * jnp.sin(eccentric_anomaly)
        return period * mean_anomaly / (2 * jnp.pi) + t0

    def fixed_point_func(eccentric_anomaly):
        """Defines a function f such that its fixed point E satisfies eccentricity_to_time(E) = t."""
        return e * jnp.sin(eccentric_anomaly) + (2 * jnp.pi) * (t - t0) / period

    t0, a, m, e, k = (
        simulation_parameters["t0"],
        simulation_parameters["a"],
        simulation_parameters["m"],
        simulation_parameters["e"],
        simulation_parameters["k"],
    )

    # First, compute the eccentric_anomaly at this instant.
    period = (2 * jnp.pi * jnp.power(a, 1.5)) / jnp.sqrt(k)
    solver = jaxopt.FixedPointIteration(fixed_point_func, maxiter=20, verbose=False)
    # Guess a solution for the eccentric anomaly.
    eccentric_anomaly_init = t
    eccentric_anomaly = solver.run(eccentric_anomaly_init).params

    # Checks that the solver has converged.
    if check_convergence:
        assert jnp.allclose(eccentric_anomaly_to_time(eccentric_anomaly), t)

    # Then, compute the position in polar coordinates.
    r = a * (1 - e * jnp.cos(eccentric_anomaly))
    phi = 2 * jnp.arctan2(
        jnp.sqrt(1 + e) * jnp.sin(eccentric_anomaly / 2),
        jnp.sqrt(1 - e) * jnp.cos(eccentric_anomaly / 2),
    )

    # Finally, compute the radial and angular momentum.
    f = 2 * jnp.pi * a / (period * jnp.sqrt(1 - (e**2)))
    v_r = f * (e * jnp.sin(phi))
    v_phi = f * (1 + e * jnp.cos(phi))
    p_r = m * v_r
    p_phi = m * r * v_phi

    # Bundle everything up.
    position = jnp.asarray([r, phi])
    momentum = jnp.asarray([p_r, p_phi])
    return position, momentum


def compute_angular_momentum(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the angular momentum at these coordinates."""
    del position, simulation_parameters
    p_phi = momentum[1]
    return p_phi


def compute_hamiltonian(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the Hamiltonian at these coordinates."""
    m, k = simulation_parameters["m"], simulation_parameters["k"]
    r = position[0]
    p_r, p_phi = momentum
    return (p_r**2) / (2 * m) + (p_phi**2) / (2 * m * (r**2)) - k / r


def polar_to_cartesian(
    position: chex.Array,
    momentum: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Converts positions and momentums from polar to Cartesian coordinates."""
    r, phi = position
    p_r, p_phi = momentum
    position_cartesian = jnp.asarray([r * jnp.cos(phi), r * jnp.sin(phi)])
    momentum_cartesian = jnp.asarray(
        [
            p_r * jnp.cos(phi) - (p_phi / r) * jnp.sin(phi),
            p_r * jnp.sin(phi) + (p_phi / r) * jnp.cos(phi),
        ]
    )
    return position_cartesian, momentum_cartesian


def static_plot_coordinates_in_phase_space(
    positions: chex.Array,
    momentums: chex.Array,
    title: str,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    max_x_position: Optional[chex.Numeric] = None,
    max_y_position: Optional[chex.Numeric] = None,
) -> plt.Figure:
    """Plots a static phase space diagram of the given coordinates."""
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

    # Convert to Cartesian coordinates.
    qs, ps = jax.vmap(polar_to_cartesian)(qs, ps)

    if fig is None:
        # Create new Figure.
        fig = plt.figure(figsize=(8, 6))

    if ax is None:
        # Add a subplot.
        ax = plt.subplot(frameon=False)
    else:
        ax.set_frame_on(False)

    # Add title.
    fig.text(
        x=0.5,
        y=0.9,
        s=title,
        ha="center",
        va="center",
        fontsize=16,
        transform=ax.transAxes,
    )

    ax.plot(
        qs[:, 0],
        qs[:, 1],
        marker="o",
        markersize=2,
        linestyle="None",
        color=plt.cm.inferno(0.1),
        zorder=1,
    )
    ax.scatter(qs[0, 0], qs[0, 1], marker="o", s=30, color="gray", zorder=2)

    if max_x_position is None:
        qx_max = np.max(np.abs(qs[:, 0]))
    else:
        qx_max = max_x_position

    if max_y_position is None:
        qy_max = np.max(np.abs(qs[:, 1]))
    else:
        qy_max = max_y_position

    ax.text(0, qy_max * 1.65, r"$q_y$", ha="center", va="center", size=14)
    ax.text(qx_max * 1.6, 0, r"$q_x$", ha="center", va="center", size=14)

    ax.plot(
        [-qx_max * 1.5, qx_max * 1.5],
        [0, 0],
        linestyle="dashed",
        color="black",
    )
    ax.plot(
        [0, 0],
        [-qy_max * 1.5, qy_max * 1.5],
        linestyle="dashed",
        color="black",
    )

    ax.set_xlim(-(qx_max * 2), (qx_max * 2))
    ax.set_ylim(-(qy_max * 2.5), (qy_max * 2.5))

    # No ticks.
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
