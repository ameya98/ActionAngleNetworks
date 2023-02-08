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

"""Simulation of a double pendulum."""

from typing import Dict, Mapping, Optional, Tuple

import chex
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


_SIMULATION_PARAMETERS = ["l1", "l2", "m1", "theta1_init", "theta2_init"]
_GRAVITY_CONSTANT = 9.81


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
    rng, *rngs = jax.random.split(rng, len(ranges_flat) + 1)
    rng_tree = jax.tree_unflatten(ranges_treedef, rngs)

    def sample_simulation_parameter(
        simulation_parameter_range: Tuple[chex.Numeric, ...],
        parameter_rng: chex.PRNGKey,
    ):
        """Sample a single simulation parameter."""

        if len(simulation_parameter_range) == 1:
            return simulation_parameter_range[0]

        minval, maxval = simulation_parameter_range
        sample = jax.random.uniform(parameter_rng, minval=minval, maxval=maxval)
        return sample

    return jax.tree_map(
        sample_simulation_parameter,
        simulation_parameter_ranges,
        rng_tree,
        is_leaf=is_tuple,
    )


def generate_canonical_coordinates(
    times: chex.Numeric,
    simulation_parameters: Mapping[str, chex.Array],
) -> Tuple[chex.Array, chex.Array]:
    """Generates positions and momentums across instants."""
    l1 = simulation_parameters["l1"]
    l2 = simulation_parameters["l2"]
    m1 = simulation_parameters["m1"]
    m2 = simulation_parameters["m2"]
    theta1_init = simulation_parameters["theta1_init"]
    theta2_init = simulation_parameters["theta2_init"]
    theta1_dot_init = 0.0
    theta2_dot_init = 0.0
    g = _GRAVITY_CONSTANT

    # Following the notation of:
    # https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
    # We use the diffrax convention:
    # y = (theta1, theta2, theta1_dot, theta2_dot)
    def alpha1(y):
        theta1, theta2 = y[0], y[1]
        return (l2 / l1) * (m2 / (m1 + m2)) * jnp.cos(theta1 - theta2)

    def alpha2(y):
        theta1, theta2 = y[0], y[1]
        return (l1 / l2) * jnp.cos(theta1 - theta2)

    def f1(y):
        theta1, theta2, theta2_dot = y[0], y[1], y[3]
        return -(l2 / l1) * (m2 / (m1 + m2)) * (theta2_dot**2) * jnp.sin(
            theta1 - theta2
        ) - ((g * jnp.sin(theta1) / l1))

    def f2(y):
        theta1, theta2, theta1_dot = y[0], y[1], y[2]
        return (l1 / l2) * (theta1_dot**2) * jnp.sin(theta1 - theta2) - (
            (g * jnp.sin(theta2) / l2)
        )

    def g1(y):
        return (f1(y) - alpha1(y) * f2(y)) / (1 - alpha1(y) * alpha2(y))

    def g2(y):
        return (-alpha2(y) * f1(y) + f2(y)) / (1 - alpha1(y) * alpha2(y))

    # Solve differential equation to get final state.
    def compute_derivative(t, y, args):
        del t, args
        _, _, theta1_dot, theta2_dot = y
        theta1_ddot = g1(y)
        theta2_ddot = g2(y)
        return (theta1_dot, theta2_dot, theta1_ddot, theta2_ddot)

    term = diffrax.ODETerm(compute_derivative)
    solver = diffrax.Dopri5()
    t0 = float(times[0])
    t1 = float(times[-1])
    dt = 0.01
    y0 = (theta1_init, theta2_init, theta1_dot_init, theta2_dot_init)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        max_steps=10 * int((t1 - t0) / dt),
        saveat=diffrax.SaveAt(ts=times),
        y0=y0,
    )
    coords = solution.ys

    # Canonical momentums.
    def compute_canonical_momentums(y):
        theta1, theta2, theta1_dot, theta2_dot = y
        p1 = (m1 + m2) * (l1**2) * theta1_dot + m2 * l1 * l2 * theta2_dot * jnp.cos(
            theta1 - theta2
        )
        p2 = m2 * (l2**2) * theta2_dot + m2 * l1 * l2 * theta1_dot * jnp.cos(
            theta1 - theta2
        )
        return p1, p2

    positions = jnp.stack((coords[0], coords[1])).T
    momentums = jax.vmap(compute_canonical_momentums)(coords)
    momentums = jnp.stack(momentums).T
    return positions, momentums


def compute_potential_energy(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the potential energy at these coordinates."""
    del momentum
    l1 = simulation_parameters["l1"]
    l2 = simulation_parameters["l2"]
    m1 = simulation_parameters["m1"]
    m2 = simulation_parameters["m2"]
    theta1, theta2 = position[0], position[1]
    g = _GRAVITY_CONSTANT

    potential1 = -m1 * g * l1 * jnp.cos(theta1)
    potential2 = -m2 * g * (l1 * jnp.cos(theta1) + l2 * jnp.cos(theta2))
    return jnp.asarray([potential1, potential2])


def compute_kinetic_energy(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the kinetic energy at these coordinates."""
    l1 = simulation_parameters["l1"]
    l2 = simulation_parameters["l2"]
    m1 = simulation_parameters["m1"]
    m2 = simulation_parameters["m2"]
    theta1, theta2 = position[0], position[1]
    p1, p2 = momentum[0], momentum[1]

    factor = l1 * l2 * (m1 + m2 * jnp.sin(theta1 - theta2) ** 2)
    theta1_dot = (l2 * p1 - l1 * p2 * jnp.cos(theta1 - theta2)) / (l1 * factor)
    theta2_dot = (l1 * (m1 + m2) * p2 - l2 * m2 * p1 * jnp.cos(theta1 - theta2)) / (
        l2 * m2 * factor
    )
    kinetic1 = 0.5 * m1 * (l1 * theta1_dot) ** 2
    kinetic2 = (
        0.5
        * m2
        * (
            (l1 * theta1_dot) ** 2
            + (l2 * theta2_dot) ** 2
            + 2 * l1 * l2 * theta1_dot * theta2_dot * jnp.cos(theta1 - theta2)
        )
    )
    return jnp.asarray([kinetic1, kinetic2])


def compute_hamiltonian(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the Hamiltonian at these coordinates."""
    kinetic = jnp.nansum(
        compute_kinetic_energy(position, momentum, simulation_parameters)
    )
    potential = jnp.sum(
        compute_potential_energy(position, momentum, simulation_parameters)
    )
    return kinetic + potential


def polar_to_cartesian(
    position: chex.Array, simulation_parameters: Mapping[str, chex.Array]
) -> chex.Array:
    """Converts position in polar coordinates to Cartesian coordinates"""
    l1 = simulation_parameters["l1"]
    l2 = simulation_parameters["l2"]
    theta1, theta2 = position[0], position[1]
    x1 = l1 * jnp.sin(theta1)
    y1 = -l1 * jnp.cos(theta1)
    x2 = x1 + l2 * jnp.sin(theta2)
    y2 = y1 + -l2 * jnp.cos(theta2)
    return jnp.asarray([x1, y1]), jnp.asarray([x2, y2])


def plot_coordinates(
    positions: chex.Array,
    momentums: chex.Array,
    simulation_parameters: Mapping[str, chex.Array],
    title: str,
    max_position: Optional[chex.Numeric] = None,
):
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

    # Add a subplot.
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(frameon=False)

    # Compute Hamiltonians.
    num_steps = qs.shape[0]
    hs = jax.vmap(compute_hamiltonian, in_axes=(0, 0, None))(
        qs, ps, simulation_parameters
    )
    hs_formatted = np.round(hs.squeeze(), 5)

    # Convert to Cartesian coordinates.
    positions1, positions2 = jax.vmap(polar_to_cartesian, in_axes=(0, None))(
        positions, simulation_parameters
    )

    if max_position is None:
        x_max = max(np.max(np.abs(positions1[:, 0])), np.max(np.abs(positions2[:, 0])))
        y_max = max(np.max(np.abs(positions1[:, 1])), np.max(np.abs(positions2[:, 1])))
        q_max = max(x_max, y_max)
    else:
        q_max = max_position

    def update(t):
        # Update data
        ax.clear()

        # Add title
        ax.text(
            0.5,
            0.83,
            title,
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            family="sans-serif",
            fontweight="light",
            fontsize=16,
        )

        # First pendulum.
        ax.plot((0, positions1[t, 0]), (0, positions1[t, 1]), color="black", zorder=-1)
        ax.plot(
            positions1[:, 0],
            positions1[:, 1],
            marker="o",
            markersize=2,
            linestyle="None",
            alpha=0.2,
            zorder=-2,
        )
        ax.scatter(
            positions1[t, 0],
            positions1[t, 1],
            marker="o",
            edgecolors="black",
            s=40,
            zorder=0,
        )

        # Second pendulum.
        ax.plot(
            (positions1[t, 0], positions2[t, 0]),
            (positions1[t, 1], positions2[t, 1]),
            color="black",
            zorder=-1,
        )
        ax.plot(
            positions2[:, 0],
            positions2[:, 1],
            marker="o",
            markersize=2,
            linestyle="None",
            alpha=0.2,
            zorder=-2,
        )
        ax.scatter(
            positions2[t, 0],
            positions2[t, 1],
            marker="o",
            edgecolors="black",
            s=40,
            zorder=0,
        )

        ax.annotate(
            r"$H$ = %0.5f" % hs_formatted[t],
            xy=(0, 0.70),
            ha="center",
            va="center",
            size=14,
        )

        ax.set_xlim(-(q_max * 1), (q_max * 1))
        ax.set_ylim(-(q_max * 1.1), (q_max * 0.5))

        # No ticks
        ax.set_aspect("equal")
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
    simulation_parameters: Mapping[str, chex.Array],
    title: str,
    max_position: Optional[chex.Numeric] = None,
    max_momentum: Optional[chex.Numeric] = None,
):
    assert len(positions) == len(momentums)

    qs, ps = positions, momentums
    qs, ps = np.asarray(qs), np.asarray(ps)
    if qs.ndim == 1:
        qs, ps = qs[Ellipsis, np.newaxis], ps[Ellipsis, np.newaxis]

    assert qs.ndim == 2, f"Got positions of shape {qs.shape}."
    assert ps.ndim == 2, f"Got momentums of shape {ps.shape}."

    # Add a subplot.
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    pos = ax.get_position()
    pos = [pos.x0, pos.y0 - 0.15, pos.width, pos.height]
    ax.set_position(pos)

    # Compute Hamiltonians.
    num_steps = qs.shape[0]
    hs = jax.vmap(compute_hamiltonian, in_axes=(0, 0, None))(
        qs, ps, simulation_parameters
    )
    hs_formatted = np.round(hs.squeeze(), 5)

    if max_position is None:
        q_max = np.max(np.abs(qs))
    else:
        q_max = max_position

    if max_momentum is None:
        p_max = np.max(np.abs(ps))
    else:
        p_max = max_momentum

    def update(t):
        # Update data
        ax.clear()

        # 2 part titles to get different font weights
        ax.text(
            0.5,
            0.83,
            title + " ",
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            color="w",
            family="sans-serif",
            fontweight="light",
            fontsize=16,
        )
        ax.text(
            0.5,
            0.78,
            "PHASE SPACE VISUALIZED",
            transform=fig.transFigure,
            ha="center",
            va="bottom",
            color="w",
            family="sans-serif",
            fontweight="bold",
            fontsize=16,
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
                color=color,
            )
            ax.scatter(
                qs_series[0],
                ps_series[0],
                marker="o",
                s=30,
                zorder=2,
                edgecolors="black",
                color="gray",
            )

        ax.text(
            0, p_max * 1.7, r"$p$", ha="center", va="center", size=14, color="white"
        )
        ax.text(
            q_max * 1.7, 0, r"$q$", ha="center", va="center", size=14, color="white"
        )

        ax.plot([-q_max * 1.5, q_max * 1.5], [0, 0], linestyle="dashed", color="white")
        ax.plot([0, 0], [-p_max * 1.5, p_max * 1.5], linestyle="dashed", color="white")

        ax.annotate(
            r"$H$ = %0.5f" % hs_formatted[t],
            xy=(0, p_max * 2.4),
            ha="center",
            va="center",
            size=14,
            color="white",
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
        fontsize=16,
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
            color=color,
        )
        ax.scatter(qs_series[0], ps_series[0], marker="o", s=30, zorder=2, color="gray")

    if max_position is None:
        q_max = np.max(np.abs(qs))
    else:
        q_max = max_position

    if max_momentum is None:
        p_max = np.max(np.abs(ps))
    else:
        p_max = max_momentum

    ax.text(0, p_max * 1.65, r"$p$", ha="center", va="center", size=14)
    ax.text(q_max * 1.6, 0, r"$q$", ha="center", va="center", size=14)

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

    ax.set_xlim(-(q_max * 2), (q_max * 2))
    ax.set_ylim(-(p_max * 2.5), (p_max * 2.5))

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
