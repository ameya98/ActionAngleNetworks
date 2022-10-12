"""Generates the plots for the paper."""

import glob
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from action_angle_networks import analysis, harmonic_motion_simulation


PLT_STYLE_CONTEXT = ["science", "ieee", "grid"]


def get_label_from_config(config: str) -> str:
    """Returns the label for the config on the plot."""
    if config == "action_angle_flow":
        return "Action-Angle Network"

    if config == "euler_update_flow":
        return "Euler Update Network"

    if config == "neural_ode":
        return "Neural ODE"

    if config == "hamiltonian_neural_network":
        return "Hamiltonian Neural Network"

    raise ValueError(f"Unsupported config: {config}")


def get_dirs_for_plot_trajectories(
    configs: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    def get_input_dir_for_config(config: str) -> str:
        return f"/Users/ameyad/Documents/google-research/workdirs/local/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=0.5/num_samples=200"

    output_dir = f"/Users/ameyad/Documents/google-research/paper/trajectories/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_samples=200"
    return {config: get_input_dir_for_config(config) for config in configs}, output_dir


def plot_trajectories(input_dirs: Dict[str, str], output_dir: str) -> None:
    """Plots test performance against number of training samples."""
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(
        ncols=len(input_dirs) + 1, figsize=((len(input_dirs) + 1) * 6, 5)
    )
    jump = 50

    with plt.style.context(PLT_STYLE_CONTEXT):
        # Choose one of the input directories to plot the true trajectories.
        input_dir = next(iter(input_dirs.values()))
        test_positions, test_momentums = analysis.get_true_trajectories(input_dir, jump)
        harmonic_motion_simulation.static_plot_coordinates_in_phase_space(
            test_positions,
            test_momentums,
            title="True Trajectory",
            fig=fig,
            ax=axs[0],
        )

        # Plot all of the predictions.
        for ax, (config, input_dir) in zip(axs[1:], input_dirs.items()):
            (
                predicted_positions,
                predicted_momentums,
            ) = analysis.get_predicted_trajectories(input_dir, jump)
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space(
                predicted_positions,
                predicted_momentums,
                title=get_label_from_config(config),
                fig=fig,
                ax=ax,
            )

        fig.suptitle(f"Predictions for Jump Size: {jump}", y=0.9, fontsize=18)
        fig.savefig(os.path.join(output_dir, "trajectories.pdf"))


def get_dirs_for_plot_inference_times(
    configs: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    def get_input_dir_for_config(config: str) -> str:
        return f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion=0.1/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"

    output_dir = f"/Users/ameyad/Documents/google-research/paper/inference_times/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_samples=100"
    return {config: get_input_dir_for_config(config) for config in configs}, output_dir


def plot_inference_times(input_dirs: Dict[str, str], output_dir: str) -> None:
    """Plots inference times against jump size."""
    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(input_dirs)))

    with plt.style.context(PLT_STYLE_CONTEXT):
        for color, (config, input_dir) in zip(colors, input_dirs.items()):
            inference_times = analysis.get_inference_times(input_dir)
            jumps = sorted(inference_times.keys())
            inference_times_for_jumps = [inference_times[jump] for jump in jumps]
            plt.plot(
                jumps,
                inference_times_for_jumps,
                "--o",
                color=color,
                label=get_label_from_config(config),
            )

        plt.legend(title="Model", title_fontsize="large", fontsize="large")
        plt.xlabel("Jump Size", fontsize="x-large")
        plt.ylabel("Inference Time\n(seconds)", fontsize="x-large")
        plt.yscale("log")
        plt.savefig(os.path.join(output_dir, "inference_times.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_time(
    configs: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    def get_input_dir_for_config(config: str) -> str:
        return f"/Users/ameyad/Documents/google-research/workdirs/local/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=0.5/num_samples=100"

    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_time/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_samples=100"
    return {config: get_input_dir_for_config(config) for config in configs}, output_dir


def plot_performance_against_time(input_dirs: Dict[str, str], output_dir: str) -> None:
    """Plots prediction errors against time."""

    os.makedirs(output_dir, exist_ok=True)
    colors = plt.cm.Dark2(np.linspace(0, 1, len(input_dirs)))

    with plt.style.context(PLT_STYLE_CONTEXT):
        for color, (config, input_dir) in zip(colors, input_dirs.items()):
            errors = analysis.get_performance_against_time(input_dir)
            jumps = sorted(errors.keys())
            errors_for_jumps = [errors[jump] for jump in jumps]
            plt.plot(
                jumps,
                errors_for_jumps,
                "--o",
                color=color,
                label=get_label_from_config(config),
            )

        plt.legend(title="Model", title_fontsize="large", fontsize="large")
        plt.xlabel("Jump Size", fontsize="x-large")
        plt.ylabel("Prediction Error", fontsize="x-large")
        plt.xscale("log")
        plt.yscale("log")
        plt.savefig(os.path.join(output_dir, "prediction_error.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_parameters(
    config: str,
) -> Tuple[List[str], str]:
    """Returns input and output directories for the performance against number of parameters plot, for this config."""
    input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/local/performance_vs_parameters/{config}/**/config.yml"
    if config == "neural_ode":
        input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/local/performance_vs_parameters/neural_ode/**/num_derivative_net_layers=3/**/config.yml"

    input_configs = glob.glob(input_config_regex, recursive=True)
    input_dirs = [os.path.dirname(input_config) for input_config in input_configs]
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_parameters/action_angle_networks/configs/harmonic_motion/{config}/"
    return input_dirs, output_dir


def plot_performance_against_parameters(input_dirs: str, output_dir: str) -> None:
    """Plots test performance against number of model parameters."""
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

        plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
        plt.xlabel("Number of Parameters", fontsize="x-large")
        plt.ylabel("Mean Relative \n Change in Hamiltonian", fontsize="x-large")
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

        plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
        plt.xlabel("Number of Parameters", fontsize="x-large")
        plt.ylabel("Mean Prediction Error", fontsize="x-large")
        plt.yscale("log")
        plt.ylim(5e-6, 1e3)
        plt.savefig(os.path.join(output_dir, "prediction_error.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_samples(config: str) -> Tuple[List[str], str]:
    """Returns input and output directories for the performance against samples plot, for this config."""
    input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/**/simulation_parameter_ranges.k_pair=1.0/config.yml"
    input_configs = glob.glob(input_config_regex, recursive=True)
    input_dirs = [os.path.dirname(input_config) for input_config in input_configs]
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=1.0"
    return input_dirs, output_dir


def plot_performance_against_samples(input_dirs: str, output_dir: str) -> None:
    """Plots test performance against number of training samples."""
    os.makedirs(output_dir, exist_ok=True)

    (
        all_prediction_losses,
        all_delta_hamiltonians,
    ) = analysis.get_performance_against_samples(input_dirs)

    num_samples = sorted(all_delta_hamiltonians.keys())
    jumps = next(iter(all_delta_hamiltonians.values())).keys()
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(jumps)))

    with plt.style.context(PLT_STYLE_CONTEXT):
        for jump, color in zip(jumps, colors):
            delta_hamiltonians_for_jump = np.asarray(
                [all_delta_hamiltonians[num_sample][jump] for num_sample in num_samples]
            )
            plt.plot(num_samples, delta_hamiltonians_for_jump, label=jump, color=color)

        plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
        plt.xlabel("Training Samples", fontsize="x-large")
        plt.ylabel("Mean Relative \n Change in Hamiltonian", fontsize="x-large")
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

        plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
        plt.xlabel("Training Samples", fontsize="x-large")
        plt.ylabel("Mean Prediction Error", fontsize="x-large")
        plt.yscale("log")
        plt.ylim(5e-6, 1e3)
        plt.savefig(os.path.join(output_dir, "prediction_error.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_steps(config: str) -> Tuple[str, str]:
    """Returns input and output directories for the performance against steps plot, for this config."""
    input_dir = f"/Users/ameyad/Documents/google-research/workdirs/local/performance_vs_steps/action_angle_networks/configs/harmonic_motion/{config}/num_samples=100"
    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_steps/action_angle_networks/configs/harmonic_motion/{config}/num_samples=100"
    return input_dir, output_dir


def plot_performance_against_steps(input_dir: str, output_dir: str) -> None:
    """Plots test performance against number of training steps."""

    for workdir in os.listdir(input_dir):
        output_workdir = os.path.join(output_dir, workdir)
        os.makedirs(output_workdir, exist_ok=True)

        full_workdir = os.path.join(input_dir, workdir)
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

            plt.xlabel("Training Steps", fontsize="x-large")
            plt.ylabel("Mean Relative \n Change in Hamiltonian", fontsize="x-large")
            plt.yscale("log")
            plt.ylim(1e-3, 1e4)
            plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
            plt.savefig(
                os.path.join(output_workdir, "relative_change_in_hamiltonian.pdf")
            )
            plt.close()

        with plt.style.context(PLT_STYLE_CONTEXT):
            for jump, color in zip(jumps, colors):
                plt.plot(steps, prediction_losses[jump], label=jump, color=color)

            plt.xlabel("Training Steps", fontsize="x-large")
            plt.ylabel("Mean Prediction Error", fontsize="x-large")
            plt.yscale("log")
            plt.ylim(5e-6, 1e3)
            plt.legend(title="Jump Size", title_fontsize="large", fontsize="large")
            plt.savefig(os.path.join(output_workdir, "prediction_error.pdf"))
            plt.close()


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Trajectories.
    # dirs = get_dirs_for_plot_trajectories(
    #     configs=["action_angle_flow", "euler_update_flow", "neural_ode"]
    # )
    # plot_trajectories(*dirs)

    # # Performance against time.
    # dirs = get_dirs_for_plot_performance_against_time(
    #     configs=["action_angle_flow", "euler_update_flow", "neural_ode"])
    # plot_performance_against_time(*dirs)

    # Inference times.
    dirs = get_dirs_for_plot_inference_times(
        configs=[
            "action_angle_flow",
            "euler_update_flow",
            "hamiltonian_neural_network",
            "neural_ode",
        ]
    )
    plot_inference_times(*dirs)

    # # Performance against parameters.
    # config = "action_angle_flow"
    # dirs = get_dirs_for_plot_performance_against_parameters(config)
    # plot_performance_against_parameters(*dirs)

    # config = "neural_ode"
    # dirs = get_dirs_for_plot_performance_against_parameters(config)
    # plot_performance_against_parameters(*dirs)

    # Performance against training samples.
    for config in [
        "action_angle_flow",
        "euler_update_flow",
        "neural_ode",
        "hamiltonian_neural_network",
    ]:
        dirs = get_dirs_for_plot_performance_against_samples(config)
        plot_performance_against_samples(*dirs)

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
