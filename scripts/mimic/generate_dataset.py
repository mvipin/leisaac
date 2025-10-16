# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument("--task_type", type=str, default=None, help="Specify task type. If your annotated dataset is recorded with keyboard, you should set it to 'keyboard', otherwise not to set it and keep default value None.")
parser.add_argument(
    "--ingredient_type",
    type=str,
    default=None,
    choices=["bread_slice_1", "bread_slice_2", "cheese_slice", "patty"],
    help="Specify the ingredient type for sandwich assembly task. This dynamically sets the object_ref in subtask configs. Options: bread_slice_1, bread_slice_2, cheese_slice, patty. If not specified, uses the default from the environment config.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

import leisaac  # noqa: F401

from leisaac.utils.env_utils import get_task_type


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )
    setattr(env_cfg, 'task_type', get_task_type(task_name, args_cli.task_type))

    # Dynamically set ingredient type for sandwich assembly task
    if args_cli.ingredient_type is not None and "AssembleSandwich" in env_name:
        # Mapping from ingredient type to human-readable names
        ingredient_display_names = {
            "bread_slice_1": "bread slice",
            "bread_slice_2": "bread slice",
            "cheese_slice": "cheese slice",
            "patty": "patty",
        }

        ingredient_name = ingredient_display_names.get(args_cli.ingredient_type, args_cli.ingredient_type)

        # Modify the first subtask config to use the specified ingredient
        if hasattr(env_cfg, 'subtask_configs') and len(env_cfg.subtask_configs) > 0:
            # Update object_ref for the first subtask (grasp ingredient)
            env_cfg.subtask_configs["so101_follower"][0].object_ref = args_cli.ingredient_type
            # Update descriptions
            env_cfg.subtask_configs["so101_follower"][0].description = f"Grasp {ingredient_name} from cartridge"
            env_cfg.subtask_configs["so101_follower"][0].next_subtask_description = f"Place {ingredient_name} on plate"
            # Update second subtask description (place on plate)
            if len(env_cfg.subtask_configs["so101_follower"]) > 1:
                env_cfg.subtask_configs["so101_follower"][1].description = f"Place {ingredient_name} on plate"

            print(f"[INFO] Ingredient type set to: {args_cli.ingredient_type} ({ingredient_name})")
        else:
            omni.log.warn(
                f"Could not find subtask_configs in environment config. "
                f"The --ingredient_type argument will be ignored."
            )

    # create environment
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # reset before starting
    env.reset()

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
    )

    try:
        asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # close sim app
    simulation_app.close()
