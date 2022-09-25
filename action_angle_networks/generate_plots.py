"""Generates the plots for the paper."""

import glob
import os
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from action_angle_networks import analysis

PLT_STYLE_CONTEXT = ["science", "ieee", "grid"]


def get_dirs_for_plot_performance_against_parameters(config: str):
    """Returns input and output directories for the performance against number of parameters plot, for this config."""
    input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/performance_vs_parameters/{config}/**/config.yml"
    if config == "neural_ode":
        input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/performance_vs_parameters/neural_ode/**/num_derivative_net_layers=3/**/config.yml"

    input_configs = glob.glob(input_config_regex, recursive=True)
    input_dirs = [os.path.dirname(input_config) for input_config in input_configs]
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_parameters/action_angle_networks/configs/harmonic_motion/{config}/"
    return input_dirs, output_dir


def plot_performance_against_parameters(input_dirs: str, output_dir: str) -> None:
    """Plots test performance against number of training samples."""
    print(input_dirs)
    os.makedirs(output_dir, exist_ok=True)
    (
        all_prediction_losses,
        all_delta_hamiltonians,
    ) = analysis.get_performance_against_parameters(input_dirs)

    num_parameters = sorted(all_delta_hamiltonians.keys())
    jumps = next(iter(all_delta_hamiltonians.values())).keys()
    colors = plt.cm.viridis(np.linspace(0, 1, len(jumps)))

    with plt.style.context(PLT_STYLE_CONTEXT):
        for jump, color in zip(jumps, colors):
            delta_hamiltonians_for_jump = np.asarray(
                [
                    all_delta_hamiltonians[num_parameter][jump]
                    for num_parameter in num_parameters
                ]
            )
            plt.plot(
                num_parameters, delta_hamiltonians_for_jump, label=jump, color=color
            )

        plt.legend(title="Jump Size")
        plt.xlabel("Number of Parameters")
        plt.ylabel("Mean Relative \n Change in Hamiltonian")
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.savefig(os.path.join(output_dir, "relative_change_in_hamiltonian.pdf"))
        plt.close()

    with plt.style.context(PLT_STYLE_CONTEXT):
        for jump, color in zip(jumps, colors):
            prediction_loss_for_jump = np.asarray(
                [
                    all_prediction_losses[num_parameter][jump]
                    for num_parameter in num_parameters
                ]
            )
            plt.plot(num_parameters, prediction_loss_for_jump, label=jump, color=color)

        plt.legend(title="Jump Size")
        plt.xlabel("Number of Parameters")
        plt.ylabel("Mean Prediction Error")
        plt.yscale("log")
        plt.ylim(5e-6, 1e3)
        plt.savefig(os.path.join(output_dir, "prediction_error.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_samples(config: str) -> Tuple[str, str]:
    """Returns input and output directories for the performance against samples plot, for this config."""
    input_dir = f"/Users/ameyad/Documents/google-research/workdirs/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=0.5"
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=0.5"
    return input_dir, output_dir


def plot_performance_against_samples(input_dir: str, output_dir: str) -> None:
    """Plots test performance against number of training samples."""
    os.makedirs(output_dir, exist_ok=True)

    workdirs = [os.path.join(input_dir, workdir) for workdir in os.listdir(input_dir)]
    (
        all_prediction_losses,
        all_delta_hamiltonians,
    ) = analysis.get_performance_against_samples(workdirs)

    num_samples = sorted(all_delta_hamiltonians.keys())
    jumps = next(iter(all_delta_hamiltonians.values())).keys()
    colors = plt.cm.viridis(np.linspace(0, 1, len(jumps)))

    with plt.style.context(PLT_STYLE_CONTEXT):
        for jump, color in zip(jumps, colors):
            delta_hamiltonians_for_jump = np.asarray(
                [all_delta_hamiltonians[num_sample][jump] for num_sample in num_samples]
            )
            plt.plot(num_samples, delta_hamiltonians_for_jump, label=jump, color=color)

        plt.legend(title="Jump Size")
        plt.xlabel("Training Samples")
        plt.ylabel("Mean Relative \n Change in Hamiltonian")
        plt.yscale("log")
        plt.ylim(1e-3, 1e4)
        plt.savefig(os.path.join(output_dir, "relative_change_in_hamiltonian.pdf"))
        plt.close()

    with plt.style.context(PLT_STYLE_CONTEXT):
        for jump, color in zip(jumps, colors):
            prediction_loss_for_jump = np.asarray(
                [all_prediction_losses[num_sample][jump] for num_sample in num_samples]
            )
            plt.plot(num_samples, prediction_loss_for_jump, label=jump, color=color)

        plt.legend(title="Jump Size")
        plt.xlabel("Training Samples")
        plt.ylabel("Mean Prediction Error")
        plt.yscale("log")
        plt.ylim(5e-6, 1e3)
        plt.savefig(os.path.join(output_dir, "prediction_error.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_steps(config: str) -> Tuple[str, str]:
    """Returns input and output directories for the performance against steps plot, for this config."""
    input_dir = f"/Users/ameyad/Documents/google-research/workdirs/performance_vs_steps/action_angle_networks/configs/harmonic_motion/{config}/num_samples=100"
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_steps/action_angle_networks/configs/harmonic_motion/{config}/num_samples=100"
    return input_dir, output_dir


def plot_performance_against_steps(input_dir: str, output_dir: str) -> None:
    """Plots test performance against number of training steps."""

    for workdir in os.listdir(input_dir):
        output_workdir = os.path.join(output_dir, workdir)
        os.makedirs(output_workdir, exist_ok=True)

        full_workdir = os.path.join(input_dir, workdir)
        print(full_workdir)
        (
            prediction_losses,
            delta_hamiltonians,
            steps,
        ) = analysis.get_performance_against_steps(full_workdir)

        jumps = sorted(delta_hamiltonians.keys())
        colors = plt.cm.viridis(np.linspace(0, 1, len(jumps)))

        with plt.style.context(PLT_STYLE_CONTEXT):
            for jump, color in zip(jumps, colors):
                plt.plot(steps, delta_hamiltonians[jump], label=jump, color=color)

            plt.xlabel("Training Steps")
            plt.ylabel("Mean Relative \n Change in Hamiltonian")
            plt.yscale("log")
            plt.ylim(1e-3, 1e4)
            plt.legend(title="Jump Size")
            plt.savefig(
                os.path.join(output_workdir, "relative_change_in_hamiltonian.pdf")
            )
            plt.close()

        with plt.style.context(PLT_STYLE_CONTEXT):
            for jump, color in zip(jumps, colors):
                plt.plot(steps, prediction_losses[jump], label=jump, color=color)

            plt.xlabel("Training Steps")
            plt.ylabel("Mean Prediction Error")
            plt.yscale("log")
            plt.ylim(5e-6, 1e3)
            plt.legend(title="Jump Size")
            plt.savefig(os.path.join(output_workdir, "prediction_error.pdf"))
            plt.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Performance against parameters.
    config = "action_angle_flow"
    dirs = get_dirs_for_plot_performance_against_parameters(config)
    plot_performance_against_parameters(*dirs)

    config = "neural_ode"
    dirs = get_dirs_for_plot_performance_against_parameters(config)
    plot_performance_against_parameters(*dirs)

    # # Performance against training samples.
    # config = "action_angle_flow"
    # dirs = get_dirs_for_plot_performance_against_samples(config)
    # plot_performance_against_samples(*dirs)

    # config = "euler_update_flow"
    # dirs = get_dirs_for_plot_performance_against_samples(config)
    # plot_performance_against_samples(*dirs)

    # config = "neural_ode"
    # dirs = get_dirs_for_plot_performance_against_samples(config)
    # plot_performance_against_samples(*dirs)

    # # Performance against training steps.
    # config = "action_angle_flow"
    # dirs = get_dirs_for_plot_performance_against_steps(config)
    # plot_performance_against_steps(*dirs)

    # config = "euler_update_flow"
    # dirs = get_dirs_for_plot_performance_against_steps(config)
    # plot_performance_against_steps(*dirs)

    # config = "neural_ode"
    # dirs = get_dirs_for_plot_performance_against_steps(config)
    # plot_performance_against_steps(*dirs)


if __name__ == "__main__":
    app.run(main)
