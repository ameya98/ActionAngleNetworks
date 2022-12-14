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

"""Simulation of coupled harmonic motion."""

from typing import Dict, Mapping, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

_SIMULATION_PARAMETERS = ["A", "phi", "m", "k_wall", "k_pair"]


def sample_simulation_parameters(
    simulation_parameter_ranges: Mapping[str, Tuple[float, float]],
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


def compute_normal_modes(
    simulation_parameters: Mapping[str, chex.Array]
) -> Tuple[chex.Array, chex.Array]:
    """Returns the angular frequencies and eigenvectors for the normal modes."""
    m, k_wall, k_pair = (
        simulation_parameters["m"],
        simulation_parameters["k_wall"],
        simulation_parameters["k_pair"],
    )
    num_particles = m.shape[0]

    # Construct coupling matrix.
    coupling_matrix = -(k_wall + num_particles * k_pair) * jnp.eye(
        num_particles
    ) + k_pair * jnp.ones((num_particles, num_particles))
    coupling_matrix = jnp.diag(1 / m) @ coupling_matrix

    # Compute eigenvalues and eigenvectors.
    eigvals, eigvecs = jnp.linalg.eig(coupling_matrix)
    w = jnp.sqrt(-eigvals)
    w = jnp.real(w)
    eigvecs = jnp.real(eigvecs)
    return w, eigvecs


def generate_canonical_coordinates(
    t: chex.Array, simulation_parameters: Mapping[str, chex.Array]
):
    """Returns q (position) and p (momentum) coordinates at time t."""
    w, eigvecs = compute_normal_modes(simulation_parameters)
    m = simulation_parameters["m"]
    normal_mode_simulation_parameters = {
        "A": simulation_parameters["A"],
        "phi": simulation_parameters["phi"],
        # We will scale momentums by mass later.
        "m": jnp.ones_like(m),
        "w": w,
    }
    normal_mode_trajectories = generate_canonical_coordinates_for_normal_mode(
        t, normal_mode_simulation_parameters
    )
    trajectories = jax.tree_map(lambda arr: eigvecs @ arr, normal_mode_trajectories)
    positions, momentums = trajectories
    # Scale momentums by mass here.
    momentums = momentums * m
    return positions, momentums


def generate_canonical_coordinates_for_normal_mode(
    t: chex.Array,
    mode_simulation_parameters: Mapping[str, chex.Array],
):
    """Returns q (position) and p (momentum) coordinates at instant t."""
    phi, a, m, w = (
        mode_simulation_parameters["phi"],
        mode_simulation_parameters["A"],
        mode_simulation_parameters["m"],
        mode_simulation_parameters["w"],
    )
    position = a * jnp.cos(w * t + phi)
    momentum = -m * w * a * jnp.sin(w * t + phi)
    return position, momentum


def compute_hamiltonian(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Mapping[str, chex.Array],
) -> chex.Array:
    """Computes the Hamiltonian at the given coordinates."""

    def _squared_l2_distance(u: chex.Array, v: chex.Array) -> chex.Array:
        """Returns the squared L2 distance between two vectors."""
        return jnp.square(u - v).sum()

    m, k_wall, k_pair = (
        simulation_parameters["m"],
        simulation_parameters["k_wall"],
        simulation_parameters["k_pair"][0],
    )
    q, p = position, momentum
    squared_distance_matrix = jax.vmap(
        jax.vmap(_squared_l2_distance, in_axes=(None, 0)), in_axes=(0, None)
    )(q, q)
    squared_distances = jnp.sum(squared_distance_matrix) / 2
    hamiltonian = ((p**2) / (2 * m)).sum()
    hamiltonian += (k_wall * (q**2)).sum() / 2
    hamiltonian += (k_pair * squared_distances) / 2
    return hamiltonian


def plot_coordinates(
    positions: chex.Array,
    momentums: chex.Array,
    simulation_parameters: Mapping[str, chex.Array],
    title: str,
) -> animation.FuncAnimation:
    """Plots coordinates in the canonical basis."""
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

    # Create new Figure with black background
    fig = plt.figure(figsize=(8, 6), facecolor="black")

    # Add a subplot with no frame
    ax = plt.subplot(frameon=False)

    # Compute Hamiltonians.
    num_steps = qs.shape[0]
    q_max = np.max(np.abs(qs))
    p_max = np.max(np.abs(ps))
    p_scale = (q_max / p_max) / 5
    hs = jax.vmap(compute_hamiltonian, in_axes=(0, 0, None))(
        qs, ps, simulation_parameters
    )
    hs_formatted = np.round(hs.squeeze(), 5)

    def update(t):
        # Update data
        ax.clear()

        # 2 part titles to get different font weights
        ax.text(
            0.5,
            1.0,
            title + " ",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="w",
            family="sans-serif",
            fontweight="light",
            fontsize=16,
        )
        ax.text(
            0.5,
            0.93,
            "VISUALIZED",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            color="w",
            family="sans-serif",
            fontweight="bold",
            fontsize=16,
        )

        for qs_series, ps_series in zip(qs.T, ps.T):
            ax.scatter(qs_series[t], 10, marker="o", s=40, color="white")
            ax.annotate(
                r"$q$",
                xy=(qs_series[t], 8),
                ha="center",
                va="center",
                size=12,
                color="white",
            )
            ax.annotate(
                r"$p$",
                xy=(qs_series[t], 10 - 0.15),
                xytext=(qs_series[t] + ps_series[t] * p_scale, 10 - 0.15),
                arrowprops=dict(arrowstyle="<-", color="white"),
                ha="center",
                va="center",
                size=12,
                color="white",
            )

        ax.plot([0, 0], [5, 15], linestyle="dashed", color="white")

        ax.annotate(
            r"$H$ = %0.5f" % hs_formatted[t],
            xy=(0, 40),
            ha="center",
            va="center",
            size=14,
            color="white",
        )

        ax.set_xlim(-(q_max * 1.1), (q_max * 1.1))
        ax.set_ylim(-1, 50)

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Construct the animation with the update function as the animation director.
    anim = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=100, blit=False
    )
    plt.close()
    return anim


def plot_coordinates_in_phase_space(
    positions: chex.Array,
    momentums: chex.Array,
    title: str,
    hamiltonians: Optional[chex.Array] = None,
    max_position: Optional[chex.Numeric] = None,
    max_momentum: Optional[chex.Numeric] = None,
) -> animation.FuncAnimation:
    """Plots a phase space diagram of the given coordinates."""
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

    # Add a subplot.
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(frameon=False)

    # Compute Hamiltonians.
    num_steps = qs.shape[0]
    if hamiltonians is not None:
        hs_formatted = np.round(hamiltonians.squeeze(), 5)

    if max_position is None:
        q_max = np.max(np.abs(qs))
    else:
        q_max = max_position

    if max_momentum is None:
        p_max = np.max(np.abs(ps))
    else:
        p_max = max_momentum

    colors = plt.cm.inferno(np.linspace(0.1, 0.8, qs.shape[1]))

    def update(t):
        # Update data.
        ax.clear()

        # Add title.
        fig.text(
            x=0.5,
            y=1.0,
            s=title,
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )

        for qs_series, ps_series, color in zip(qs.T, ps.T, colors):
            ax.plot(
                qs_series,
                ps_series,
                marker="o",
                markersize=2,
                linestyle="None",
                zorder=1,
                alpha=0.2,
                color=color,
            )
            ax.scatter(
                qs_series[t],
                ps_series[t],
                marker="o",
                s=30,
                zorder=2,
                color=color,
                edgecolors="black",
            )

        ax.text(0, p_max * 1.65, r"$p$", ha="center", va="center", size=12)
        ax.text(q_max * 1.6, 0, r"$q$", ha="center", va="center", size=12)

        ax.plot(
            [-q_max * 1.5, q_max * 1.5],
            [0, 0],
            linestyle="dashed",
            color="black",
        )
        ax.plot(
            [0, 0],
            [-p_max * 1.5, p_max * 1.5],
            linestyle="dashed",
            color="black",
        )

        if hamiltonians is not None:
            ax.annotate(
                r"$H$ = %0.5f" % hs_formatted[t],
                xy=(0, p_max * 2),
                ha="center",
                va="center",
                size=12,
            )

        ax.set_xlim(-(q_max * 2), (q_max * 2))
        ax.set_ylim(-(p_max * 2.5), (p_max * 2.5))

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

    # Construct the animation with the update function as the animation director.
    anim = animation.FuncAnimation(
        fig, update, frames=num_steps, interval=100, blit=False
    )
    plt.close()
    return anim


def static_plot_coordinates_in_phase_space(
    positions: chex.Array,
    momentums: chex.Array,
    title: str,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    max_position: Optional[chex.Numeric] = None,
    max_momentum: Optional[chex.Numeric] = None,
) -> plt.Figure:
    """Plots a static phase space diagram of the given coordinates."""
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

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
        fontsize=22,
        transform=ax.transAxes,
    )

    colors = plt.cm.inferno(np.linspace(0.1, 0.8, qs.shape[1]))
    for qs_series, ps_series, color in zip(qs.T, ps.T, colors):
        ax.plot(
            qs_series,
            ps_series,
            marker="o",
            markersize=2,
            linestyle="None",
            zorder=1,
            alpha=0.5,
            color=color,
        )
        ax.scatter(
            qs_series[0],
            ps_series[0],
            marker="o",
            s=30,
            zorder=2,
            color=color,
            edgecolors="black",
        )

    if max_position is None:
        q_max = np.max(np.abs(qs))
    else:
        q_max = max_position

    if max_momentum is None:
        p_max = np.max(np.abs(ps))
    else:
        p_max = max_momentum

    ax.text(0, p_max * 1.4, r"$p$", ha="center", va="center", size=20)
    ax.text(q_max * 1.35, 0, r"$q$", ha="center", va="center", size=20)

    ax.plot(
        [-q_max * 1.2, q_max * 1.2],
        [0, 0],
        linestyle="dashed",
        color="black",
    )
    ax.plot(
        [0, 0],
        [-p_max * 1.2, p_max * 1.2],
        linestyle="dashed",
        color="black",
    )

    ax.set_xlim(-(q_max * 2), (q_max * 2))
    ax.set_ylim(-(p_max * 2.5), (p_max * 2.5))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
