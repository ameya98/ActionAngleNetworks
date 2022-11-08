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


_SIMULATION_PARAMETERS = ["l_1", "l_2", "m_1", "theta_1_init", "theta_2_init"]
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

    simulation_parameters = jax.tree_map(
        sample_simulation_parameter,
        simulation_parameter_ranges,
        rng_tree,
        is_leaf=is_tuple,
    )
    return jax.tree_map(jnp.squeeze, simulation_parameters)


def generate_canonical_coordinates(
    times: chex.Numeric,
    simulation_parameters: Mapping[str, chex.Array],
) -> Tuple[chex.Array, chex.Array]:
    """Generates positions and momentums across instants."""
    l_1 = simulation_parameters["l_1"]
    l_2 = simulation_parameters["l_2"]
    m_1 = simulation_parameters["m_1"]
    m_2 = simulation_parameters["m_2"]
    theta_1_init = simulation_parameters["theta_1_init"]
    theta_2_init = simulation_parameters["theta_2_init"]
    theta_1_dot_init = 0.0
    theta_2_dot_init = 0.0
    g = _GRAVITY_CONSTANT

    # Following the notation of:
    # https://diego.assencio.com/?index=1500c66ae7ab27bb0106467c68feebc6
    # We use the diffrax convention:
    # y = (theta_1, theta_2, theta_1_dot, theta_2_dot)
    def alpha_1(y):
        theta_1, theta_2 = y[0], y[1]
        return (l_2 / l_1) * (m_2 / (m_1 + m_2)) * jnp.cos(theta_1 - theta_2)

    def alpha_2(y):
        theta_1, theta_2 = y[0], y[1]
        return (l_1 / l_2) * jnp.cos(theta_1 - theta_2)

    def f_1(y):
        theta_1, theta_2, theta_2_dot = y[0], y[1], y[3]
        return -(l_2 / l_1) * (m_2 / (m_1 + m_2)) * (theta_2_dot**2) * jnp.sin(
            theta_1 - theta_2
        ) - ((g * jnp.sin(theta_1) / l_1))

    def f_2(y):
        theta_1, theta_2, theta_1_dot = y[0], y[1], y[2]
        return (l_1 / l_2) * (theta_1_dot**2) * jnp.sin(theta_1 - theta_2) - (
            (g * jnp.sin(theta_2) / l_2)
        )

    def g_1(y):
        return (f_1(y) - alpha_1(y) * f_2(y)) / (1 - alpha_1(y) * alpha_2(y))

    def g_2(y):
        return (-alpha_2(y) * f_1(y) + f_2(y)) / (1 - alpha_1(y) * alpha_2(y))

    # Solve differential equation to get final state.
    def compute_derivative(t, y, args):
        del t, args
        _, _, theta_1_dot, theta_2_dot = y
        jax.debug.print(
            "f_1={f_1}, f_2={f_2}, alpha_1={alpha_1}, alpha_2={alpha_2}",
            f_1=f_1(y),
            f_2=f_2(y),
            alpha_1=alpha_1(y),
            alpha_2=alpha_2(y),
        )
        theta_1_ddot = g_1(y)
        theta_2_ddot = g_2(y)
        return (theta_1_dot, theta_2_dot, theta_1_ddot, theta_2_ddot)

    term = diffrax.ODETerm(compute_derivative)
    solver = diffrax.Dopri5()
    t0 = float(times[0])
    t1 = float(times[-1])
    t1 = 1
    dt = 0.01
    times = jnp.arange(t0, t1, dt)
    y0 = (theta_1_init, theta_2_init, theta_1_dot_init, theta_2_dot_init)
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        max_steps=5000,
        saveat=diffrax.SaveAt(t0=t0, t1=t1, ts=times),
        y0=y0,
    )
    coords = solution.ys

    # Canonical momentums.
    def compute_momentums(y):
        theta_1, theta_2, theta_1_dot, theta_2_dot = y
        p_1 = (m_1 + m_2) * (
            l_1**2
        ) * theta_1_dot + m_2 * l_1 * l_2 * theta_2_dot * jnp.cos(theta_1 - theta_2)
        p_2 = m_2 * (l_2**2) * theta_2_dot + m_2 * l_1 * l_2 * theta_1_dot * jnp.cos(
            theta_1 - theta_2
        )
        return p_1, p_2

    positions = jnp.stack((coords[0], coords[1])).T
    momentums = jax.vmap(compute_momentums)(coords)
    momentums = jnp.stack(momentums).T
    print(
        "pot",
        jax.vmap(compute_potential_energy, in_axes=(0, 0, None))(
            positions, momentums, simulation_parameters
        ),
    )
    print(
        "kines",
        jax.vmap(compute_kinetic_energy, in_axes=(0, 0, None))(
            positions, momentums, simulation_parameters
        ),
    )
    print(
        "hams",
        jax.vmap(compute_hamiltonian, in_axes=(0, 0, None))(
            positions, momentums, simulation_parameters
        ),
    )
    return positions, momentums


def compute_potential_energy(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the potential energy at these coordinates."""
    del momentum
    l_1 = simulation_parameters["l_1"]
    l_2 = simulation_parameters["l_2"]
    m_1 = simulation_parameters["m_1"]
    m_2 = simulation_parameters["m_2"]
    theta_1, theta_2 = position[0], position[1]
    g = _GRAVITY_CONSTANT

    potential_1 = -m_1 * g * l_1 * jnp.cos(theta_1)
    potential_2 = -m_2 * g * (l_1 * jnp.cos(theta_1) + l_2 * jnp.cos(theta_2))
    return jnp.asarray([potential_1, potential_2])


def compute_kinetic_energy(
    position: chex.Array,
    momentum: chex.Array,
    simulation_parameters: Dict[str, chex.Array],
) -> chex.Array:
    """Computes the kinetic energy at these coordinates."""
    l_1 = simulation_parameters["l_1"]
    l_2 = simulation_parameters["l_2"]
    m_1 = simulation_parameters["m_1"]
    m_2 = simulation_parameters["m_2"]
    theta_1, theta_2 = position[0], position[1]
    p_1, p_2 = momentum[0], momentum[1]

    factor = l_1 * l_2 * (m_1 + m_2 * jnp.sin(theta_1 - theta_2) ** 2)
    theta_1_dot = (l_2 * p_1 - l_1 * p_2 * jnp.cos(theta_1 - theta_2)) / (l_1 * factor)
    theta_2_dot = (
        l_1 * (m_1 + m_2) * p_2 - l_2 * m_2 * p_1 * jnp.cos(theta_1 - theta_2)
    ) / (l_2 * m_2 * factor)
    kinetic_1 = 0.5 * m_1 * (l_1 * theta_1_dot) ** 2
    kinetic_2 = (
        0.5
        * m_2
        * (
            (l_1 * theta_1_dot) ** 2
            + (l_2 * theta_2_dot) ** 2
            + 2 * l_1 * l_2 * theta_1_dot * theta_2_dot * jnp.cos(theta_1 - theta_2)
        )
    )
    return jnp.asarray([kinetic_1, kinetic_2])


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
