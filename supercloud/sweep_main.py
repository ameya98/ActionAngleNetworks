"""Script to begin training on MIT SuperCloud."""

import itertools
import os
import sys
from typing import Any, Dict, Optional, Sequence

import jax
import ml_collections
import tensorflow as tf
import yaml
from absl import app, flags, logging
from clu import platform
from ml_collections import config_flags

sys.path.append("..")
from action_angle_networks import train


_TASK_INDEX = flags.DEFINE_integer("index", None, "Task index for sweeps.")
_SWEEP_FILE = flags.DEFINE_string(
    "sweep_file", None, "YAML file for defining parameter sweeps."
)
_BASE_WORKDIR = flags.DEFINE_string(
    "base_workdir", None, "Base directory to store model data."
)
_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def read_sweep_from_file(sweepfile: str):
    with open(sweepfile, "r") as f:
        return yaml.unsafe_load(f)


def get_updates_at_index(
    index: Optional[int], sweep: Dict[str, Sequence[Any]]
) -> Dict[str, Any]:
    """Get the config updates at the given index."""
    if index is None:
        return {}

    values = None
    for current_index, current_values in enumerate(itertools.product(*sweep.values())):
        if current_index != index:
            continue

        values = current_values
        break

    if values is None:
        raise ValueError("Index is too large for chosen sweep.")

    keys = list(sweep.keys())
    return dict(zip(keys, values))


def update_config(
    config: ml_collections.ConfigDict, updates: Dict[Any, Any]
) -> ml_collections.ConfigDict:
    """Updates the config."""
    return config.update_from_flattened_dict(updates)


def update_workdir(base_workdir: str, sweep_file: str, updates: Dict[Any, Any]) -> str:
    """Updates the workdir."""
    sweep_file = os.path.splitext(sweep_file)[0]
    workdir = os.path.join(base_workdir, sweep_file)

    for update_key, update_val in updates.items():
        workdir = os.path.join(workdir, f"{update_key}={update_val}")
    return workdir


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    config = _CONFIG.value
    sweep_file = _SWEEP_FILE.value
    task_index = _TASK_INDEX.value
    base_workdir = _BASE_WORKDIR.value

    # Get updates.
    sweep = read_sweep_from_file(sweep_file)
    updates = get_updates_at_index(task_index, sweep)

    # Update config with these updates.
    config = update_config(config, updates)
    workdir = update_workdir(base_workdir, sweep_file, updates)

    # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    # This example only supports single-host training on a single device.
    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX local devices: %r", jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"process_index: {jax.process_index()}, "
        f"process_count: {jax.process_count()}"
    )
    platform.work_unit().create_artifact(
        platform.ArtifactType.DIRECTORY, workdir, "workdir"
    )

    train.train_and_evaluate(config, workdir)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "base_workdir", "sweep_file", "index"])
    app.run(main)
