"""Generates the plots for the paper."""

import glob
import os
from typing import Dict, List, Sequence, Tuple

import jax

import matplotlib.pyplot as plt
import numpy as np
from absl import app

from action_angle_networks import analysis
from action_angle_networks.simulation import (
    harmonic_motion_simulation,
    orbit_simulation,
)


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
        return "HNN"

    raise ValueError(f"Unsupported config: {config}")


def get_dirs_for_plot_static_trajectories(
    configs: Sequence[str],
    simulation: str,
    num_train_samples: int,
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the trajectories plot for a list of configs."""
    if simulation == "harmonic_motion":

        def get_input_dir_for_config(config: str) -> str:
            return f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"

        output_dir = f"/Users/ameyad/Documents/google-research/paper/static_trajectories/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_train_samples={num_train_samples}"
        return {
            config: get_input_dir_for_config(config) for config in configs
        }, output_dir

    if simulation == "orbit":

        def get_input_dir_for_config(config: str) -> str:
            return f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/orbit/orbit/performance_vs_samples/orbit/{config}.py/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/"

        output_dir = f"/Users/ameyad/Documents/google-research/paper/static_trajectories/action_angle_networks/configs/orbit/num_train_samples={num_train_samples}"
        return {
            config: get_input_dir_for_config(config) for config in configs
        }, output_dir


def plot_static_trajectories(
    input_dirs: Dict[str, str],
    output_dir: str,
    simulation: str,
    jump: int,
    transparent: bool = False,
) -> None:
    """Plots a static view of the trajectories."""
    output_dir = os.path.join(output_dir, f"jump={jump}")
    os.makedirs(output_dir, exist_ok=True)

    if simulation == "harmonic_motion":
        plot_coordinates_fn = (
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space
        )
    if simulation == "orbit":
        plot_coordinates_fn = orbit_simulation.static_plot_coordinates_in_phase_space

    fig, axs = plt.subplots(
        ncols=len(input_dirs) + 1, figsize=((len(input_dirs) + 1) * 6, 5)
    )

    with plt.style.context(PLT_STYLE_CONTEXT):
        # Choose one of the input directories to plot the true trajectories.
        input_dir = next(iter(input_dirs.values()))
        test_positions, test_momentums, _ = analysis.get_test_trajectories(
            input_dir, jump
        )
        plot_coordinates_fn(
            test_positions[:200],
            test_momentums[:200],
            title="True Trajectory",
            fig=fig,
            ax=axs[0],
        )

        # Plot all of the predictions.
        for ax, (config, input_dir) in zip(axs[1:], input_dirs.items()):
            (
                predicted_positions,
                predicted_momentums,
                _,
            ) = analysis.get_recursive_multi_step_predicted_trajectories(
                input_dir, jump
            )
            plot_coordinates_fn(
                predicted_positions[:200],
                predicted_momentums[:200],
                title=get_label_from_config(config),
                fig=fig,
                ax=ax,
                max_position=np.abs(test_positions[:200]).max(),
                max_momentum=np.abs(test_momentums[:200]).max(),
            )

        # fig.suptitle(f"Predictions for Jump Size: {jump}", y=0.9, fontsize=18)
        # fig.suptitle(f"True Trajectory and Model Predictions", y=0.9, fontsize=18)
        print("Transparent:", transparent)
        fig.savefig(
            os.path.join(output_dir, "recursive_multi_step_trajectories.pdf"),
            transparent=transparent,
        )


def get_dirs_for_plot_static_trajectory(
    config: str, num_train_samples: int
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    input_dir = f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"
    output_dir = f"/Users/ameyad/Documents/google-research/paper/single_static_trajectories/action_angle_networks/configs/harmonic_motion/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/k_pair=0.5/{config}"
    return input_dir, output_dir


def plot_static_trajectory(
    input_dir: str,
    output_dir: str,
    simulation: str,
    jump: int,
    title: str,
    plot_true_trajectory: bool = False,
) -> None:
    """Plots an animation of a single trajectory."""
    output_dir = os.path.join(output_dir, f"jump={jump}")
    os.makedirs(output_dir, exist_ok=True)

    if simulation == "harmonic_motion":
        plot_coordinates_fn = (
            harmonic_motion_simulation.static_plot_coordinates_in_phase_space
        )
    else:
        raise NotImplementedError

    with plt.style.context(PLT_STYLE_CONTEXT):
        test_positions, test_momentums, _ = analysis.get_test_trajectories(
            input_dir, jump
        )
        (
            predicted_positions,
            predicted_momentums,
            _,
        ) = analysis.get_recursive_multi_step_predicted_trajectories(input_dir, jump)

        # Plot true trajectory?
        if plot_true_trajectory:
            fig = plot_coordinates_fn(
                test_positions[:200],
                test_momentums[:200],
                title=title,
            )
            fig.savefig(
                os.path.join(output_dir, "test_trajectories.pdf"), transparent=True
            )

        # Plot predictions.
        else:
            fig = plot_coordinates_fn(
                predicted_positions[:200],
                predicted_momentums[:200],
                title=title,
                max_position=np.abs(test_positions[:200]).max(),
                max_momentum=np.abs(test_momentums[:200]).max(),
            )
            fig.savefig(
                os.path.join(
                    output_dir, "recursive_multi_step_predicted_trajectories.pdf"
                ),
                transparent=True,
            )


def get_dirs_for_plot_animated_trajectory(
    config: str, num_train_samples: int, plot_phase_space: bool
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""
    if plot_phase_space:
        plot_type = "phase"
    else:
        plot_type = "original"
    input_dir = f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"
    output_dir = f"/Users/ameyad/Documents/google-research/paper/animated_trajectories/action_angle_networks/configs/harmonic_motion/{config}/num_samples=1000/train_split_proportion={num_train_samples / 1000}/num_train_steps=50000/k_pair=0.5/{plot_type}"
    return input_dir, output_dir


def plot_animated_trajectory(
    input_dir: str,
    output_dir: str,
    simulation: str,
    jump: int,
    title: str,
    plot_true_trajectory: bool = False,
    plot_phase_space: bool = True,
) -> None:
    """Plots an animation of a single trajectory."""
    output_dir = os.path.join(output_dir, f"jump={jump}")
    os.makedirs(output_dir, exist_ok=True)

    if simulation == "harmonic_motion":
        if plot_phase_space:
            plot_coordinates_fn = (
                harmonic_motion_simulation.plot_coordinates_in_phase_space
            )
        else:
            plot_coordinates_fn = harmonic_motion_simulation.plot_coordinates
    else:
        raise NotImplementedError

    with plt.style.context(PLT_STYLE_CONTEXT):
        (
            test_positions,
            test_momentums,
            true_hamiltonians,
        ) = analysis.get_test_trajectories(input_dir, jump)
        (
            predicted_positions,
            predicted_momentums,
            hamiltonians,
        ) = analysis.get_recursive_multi_step_predicted_trajectories(input_dir, jump)

        # Plot true trajectory?
        if plot_true_trajectory:
            anim = plot_coordinates_fn(
                test_positions[:200],
                test_momentums[:200],
                title=title,
            )
            anim.save(os.path.join(output_dir, "test_trajectories.mp4"), dpi=500)

        # Plot predictions.
        else:
            anim = plot_coordinates_fn(
                predicted_positions[:200],
                predicted_momentums[:200],
                title=title,
                max_position=np.abs(test_positions[:200]).max(),
                max_momentum=np.abs(test_momentums[:200]).max(),
            )
            anim.save(
                os.path.join(
                    output_dir, "recursive_multi_step_predicted_trajectories.mp4"
                ),
                dpi=500,
            )


def get_dirs_for_plot_inference_times(
    configs: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    def get_input_dir_for_config(config: str) -> str:
        return f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion=0.1/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"

    output_dir = f"/Users/ameyad/Documents/google-research/paper/inference_times/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_samples=100"
    return {config: get_input_dir_for_config(config) for config in configs}, output_dir


def plot_inference_times(
    input_dirs: Dict[str, str], output_dir: str, transparent: bool = False
) -> None:
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

        plt.legend(title="Model", title_fontsize="medium", fontsize="medium")
        plt.xlabel("Jump Size", fontsize="x-large")
        plt.ylabel("Inference Time\n(seconds)", fontsize="x-large")
        plt.yscale("log")
        if transparent:
            plt.savefig(
                os.path.join(output_dir, "inference_times_transparent.pdf"),
                transparent=True,
            )
        else:
            plt.savefig(os.path.join(output_dir, "inference_times.pdf"))
        plt.close()


def get_dirs_for_plot_performance_against_time(
    configs: Sequence[str],
) -> Tuple[Dict[str, str], str]:
    """Returns input and output directories for the inference times plot, for this config."""

    def get_input_dir_for_config(config: str) -> str:
        return f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/num_samples=1000/train_split_proportion=0.1/num_train_steps=50000/simulation_parameter_ranges.k_pair=0.5"

    output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_time/action_angle_networks/configs/harmonic_motion/k_pair=0.5/num_samples=100"
    return {config: get_input_dir_for_config(config) for config in configs}, output_dir


def plot_performance_against_time(
    input_dirs: Dict[str, str], output_dir: str, transparent: bool = False
) -> None:
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

        plt.legend(title="Model", title_fontsize="medium", fontsize="medium")
        plt.xlabel("Jump Size", fontsize="x-large")
        plt.ylabel("Prediction Error", fontsize="x-large")
        plt.xscale("log")
        plt.yscale("log")
        if transparent:
            plt.savefig(
                os.path.join(output_dir, "prediction_error_transparent.pdf"),
                transparent=True,
            )
        else:
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


def get_dirs_for_plot_performance_against_samples(
    config: str, simulation: str
) -> Tuple[List[str], str]:
    """Returns input and output directories for the performance against samples plot, for this config."""
    if simulation == "harmonic_motion":
        input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/harmonic_motion/harmonic_motion/performance_vs_samples/harmonic_motion/{config}.py/**/simulation_parameter_ranges.k_pair=1.0/config.yml"
        input_configs = glob.glob(input_config_regex, recursive=True)
        input_dirs = [os.path.dirname(input_config) for input_config in input_configs]
        output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_samples/action_angle_networks/configs/harmonic_motion/{config}/k_pair=0.5"
        return input_dirs, output_dir
    if simulation == "orbit":
        input_config_regex = f"/Users/ameyad/Documents/google-research/workdirs/supercloud/sweeps/orbit/orbit/performance_vs_samples/orbit/{config}.py/**/config.yml"
        input_configs = glob.glob(input_config_regex, recursive=True)
        input_dirs = [os.path.dirname(input_config) for input_config in input_configs]
        output_dir = f"/Users/ameyad/Documents/google-research/paper/performance_vs_samples/action_angle_networks/configs/orbit/{config}"
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

    # # Combined static trajectories.
    # dirs = get_dirs_for_plot_static_trajectories(
    #     configs=[
    #         "action_angle_flow",
    #         # "euler_update_flow",
    #         "neural_ode",
    #         "hamiltonian_neural_network",
    #     ],
    #     simulation="harmonic_motion",
    #     num_train_samples=100,
    # )
    # for jump in [1, 2, 5, 10]:
    #     plot_static_trajectories(
    #         *dirs, simulation="harmonic_motion", jump=jump, transparent=True
    #     )

    # # Single static trajectories.
    # for config in [
    #     "action_angle_flow",
    #     "euler_update_flow",
    #     "hamiltonian_neural_network",
    #     "neural_ode",
    # ]:
    #     dirs = get_dirs_for_plot_static_trajectory(config, num_train_samples=100)
    #     plot_static_trajectory(
    #         *dirs,
    #         simulation="harmonic_motion",
    #         jump=1,
    #         title=get_label_from_config(config),
    #     )
    #     if config == "action_angle_flow":
    #         plot_static_trajectory(
    #             *dirs,
    #             simulation="harmonic_motion",
    #             jump=1,
    #             title="True Trajectory",
    #             plot_true_trajectory=True,
    #         )

    # # Animated trajectories.
    # for config in [
    #     "action_angle_flow",
    #     "euler_update_flow",
    #     "hamiltonian_neural_network",
    #     "neural_ode",
    # ]:
    #     dirs = get_dirs_for_plot_animated_trajectory(config, num_train_samples=100, plot_phase_space=True)
    #     plot_animated_trajectory(
    #         *dirs,
    #         simulation="harmonic_motion",
    #         plot_phase_space=True,
    #         jump=1,
    #         title=get_label_from_config(config),
    #     )
    #     if config == "action_angle_flow":
    #         plot_animated_trajectory(
    #             *dirs,
    #             simulation="harmonic_motion",
    #             jump=1,
    #             title="True Trajectory",
    #             plot_true_trajectory=True,
    #         )

    # # Performance against time jumps.
    # dirs = get_dirs_for_plot_performance_against_time(
    #     configs=[
    #         "action_angle_flow",
    #         "euler_update_flow",
    #         "neural_ode",
    #         "hamiltonian_neural_network",
    #     ]
    # )
    # plot_performance_against_time(*dirs)

    # # Inference times.
    # dirs = get_dirs_for_plot_inference_times(
    #     configs=[
    #         "action_angle_flow",
    #         "euler_update_flow",
    #         "hamiltonian_neural_network",
    #         "neural_ode",
    #     ]
    # )
    # plot_inference_times(*dirs)

    # # Performance against parameters.
    # for config in [
    #     "action_angle_flow",
    #     "neural_ode",
    # ]:
    #     dirs = get_dirs_for_plot_performance_against_parameters(config)
    #     plot_performance_against_parameters(*dirs)

    # # Performance against training samples.
    # for config in [
    #     "action_angle_flow",
    #     "euler_update_flow",
    #     "neural_ode",
    #     "hamiltonian_neural_network",
    # ]:
    #     dirs = get_dirs_for_plot_performance_against_samples(config, "harmonic_motion")
    #     plot_performance_against_samples(*dirs)

    # # Performance against training steps.
    # for config in [
    #     "action_angle_flow",
    #     "euler_update_flow",
    #     "neural_ode",
    #     "hamiltonian_neural_network",
    # ]:
    #     dirs = get_dirs_for_plot_performance_against_steps(config)
    #     plot_performance_against_steps(*dirs)

    # Animated trajectories.
    for config in [
        "action_angle_flow",
        "euler_update_flow",
        "hamiltonian_neural_network",
        "neural_ode",
    ]:
        dirs = get_dirs_for_plot_animated_trajectory(
            config, num_train_samples=100, plot_phase_space=True
        )
        plot_animated_trajectory(
            *dirs,
            simulation="harmonic_motion",
            jump=1,
            title=get_label_from_config(config),
            plot_phase_space=True,
        )
        if config == "action_angle_flow":
            plot_animated_trajectory(
                *dirs,
                simulation="harmonic_motion",
                jump=1,
                title="True Trajectory",
                plot_true_trajectory=True,
                plot_phase_space=True,
            )


if __name__ == "__main__":
    app.run(main)
