"""Loads precomputed MD17 trajectories for molecules."""

import os

import numpy as np
import scipy.interpolate

_MD17_BASE_PATH = "/Users/ameyad/Documents/google-research/data/MD17/npz_data/"


def charge_to_element(nuclear_charge: int) -> str:
    """Returns the atomic symbol for a given nuclear charge."""
    charge_to_element_map = {
        1: "H",
        6: "C",
        7: "N",
        8: "O",
    }
    return charge_to_element_map.get(nuclear_charge)


def charge_to_mass(nuclear_charge: int) -> int:
    """Returns the nuclear mass for a given nuclear charge."""
    element_to_mass_map = {
        "H": 1,
        "C": 12,
        "N": 14,
        "O": 16,
    }
    element = charge_to_element(nuclear_charge)
    return element_to_mass_map.get(element)


def load_trajectory(molecule: str, num_samples: int, resample: bool):
    """Loads a molecular dynamics trajectory."""

    # Load sample of trajectory.
    molecule_path = os.path.join(_MD17_BASE_PATH, f"rmd17_{molecule}.npz")
    with np.load(molecule_path) as data:
        indices = data["old_indices"]
        print("indices", indices, len(indices))
        return
        timesteps = np.sort(indices)[: num_samples + 1]
        indices = np.argsort(indices)[: num_samples + 1]
        positions = data["coords"][indices]
        nuclear_charges = data["nuclear_charges"]

    print(positions.shape)
    # Resample with uniform time sampling.
    if resample:
        positions_fn = scipy.interpolate.interp1d(
            timesteps, positions, kind="cubic", axis=0
        )
        resampled_timesteps = np.linspace(timesteps[0], timesteps[-1], num_samples + 1)
        positions = positions_fn(resampled_timesteps)

    # Ignore dividing by timestep of finite differences for momentums, because we usually normalize features.
    nuclear_masses = np.vectorize(charge_to_mass)(nuclear_charges)
    momentums = np.diff(positions, axis=0) * nuclear_masses[np.newaxis, :, np.newaxis]
    return positions[:-1], momentums, nuclear_charges
