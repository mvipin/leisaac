from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .assemble_sandwich_env_cfg import AssembleSandwichEnvCfg


@configclass
class SandwichIngredientMimicEnvCfg(AssembleSandwichEnvCfg, MimicEnvCfg):
    """
    Generalized configuration for sandwich ingredient manipulation with mimic environment.
    
    This environment supports language prompt-based differentiation between ingredient types
    (bread, patty, cheese) using the same physical manipulation skills. The specific ingredient
    type is determined by language prompts during training and inference.
    """

    def __post_init__(self):
        super().__post_init__()

        # Data generation configuration
        self.datagen_config.name = "sandwich_ingredient_leisaac_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 20  # More trials for better generalization
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 42

        # Generalized subtask configurations for ingredient manipulation
        # These work for any ingredient type (bread, patty, cheese) with language prompt differentiation
        subtask_configs = []
        
        # Subtask 1: Grasp ingredient from cartridge
        # Note: The specific ingredient (bread/patty/cheese) is determined by language prompts
        # during training data generation and inference
        subtask_configs.append(
            SubTaskConfig(
                object_ref="ingredient",  # Generic reference - maps to specific ingredient via USD
                subtask_term_signal="grasp_ingredient",  # Generic termination signal
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 5},  # More neighbors for better selection
                action_noise=0.002,  # Lower noise for precise manipulation
                num_interpolation_steps=8,  # More steps for smoother motion
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Grasp ingredient from cartridge",  # Generic description
                next_subtask_description="Place ingredient on plate",
            )
        )
        
        # Subtask 2: Place ingredient on plate (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                object_ref="plate",  # Reference to destination plate
                subtask_term_signal=None,  # No termination signal for final subtask
                subtask_term_offset_range=(0, 0),  # No offset for final subtask
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 5},
                action_noise=0.002,
                num_interpolation_steps=8,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place ingredient on plate",  # Generic description
            )
        )
        
        # Assign subtask configurations to the SO101 follower robot
        self.subtask_configs["so101_follower"] = subtask_configs
