"""
Optimized camera configurations for sandwich assembly task.

This file contains camera position and orientation settings specifically
optimized for the simplified table workspace (1.2m × 0.8m × 0.85m height)
with ingredients holder on the left and plate on the right.

Usage:
1. Copy these configurations to your environment config file
2. Or import and apply them in __post_init__ method
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg

# =============================================================================
# OPTIMIZED CAMERA CONFIGURATIONS FOR SANDWICH ASSEMBLY
# =============================================================================

# Workspace context:
# - Table: 1.2m × 0.8m × 0.85m height, centered at (0, 0, 0.85)
# - Ingredients holder: (-0.3, 0, 0.90) - Left side
# - Plate: (0.3, 0, 0.90) - Right side  
# - Robot base: (0, -1.0, 0) - In front of table

SANDWICH_WRIST_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
    offset=TiledCameraCfg.OffsetCfg(
        # Position: Slightly forward and up from gripper for better ingredient view
        pos=(0.02, 0.08, -0.03),
        # Rotation: Angled down toward table surface for optimal manipulation view
        rot=(-0.35, -0.93, -0.05, 0.08),  # Quaternion (w,x,y,z)
        convention="ros"
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=32.0,  # Slightly wider view for ingredients
        focus_distance=400.0,
        horizontal_aperture=40.0,  # ~80° FOV for better ingredient coverage
        clipping_range=(0.005, 20.0),  # Closer near plane for detailed view
        lock_camera=True
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,  # 30 FPS
)

SANDWICH_FRONT_CAMERA_CFG = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base/front_camera",
    offset=TiledCameraCfg.OffsetCfg(
        # Position: Higher and angled for complete table overview
        pos=(-0.2, -0.8, 0.7),
        # Rotation: Looking down at table workspace from front-diagonal
        rot=(0.2, -0.98, 0.0, 0.0),  # Quaternion (w,x,y,z)
        convention="ros"
    ),
    data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=24.0,  # Wider view for full workspace
        focus_distance=400.0,
        horizontal_aperture=45.0,  # ~85° FOV for complete table view
        clipping_range=(0.1, 30.0),
        lock_camera=True
    ),
    width=640,
    height=480,
    update_period=1 / 30.0,  # 30 FPS
)

# Alternative camera positions for different viewing preferences
ALTERNATIVE_WRIST_POSITIONS = {
    "close_detail": {
        "pos": (0.01, 0.06, -0.02),
        "rot": (-0.4, -0.91, -0.04, 0.05),
        "focal_length": 36.0,
        "aperture": 35.0,  # ~75° FOV
    },
    "wide_view": {
        "pos": (0.03, 0.10, -0.04),
        "rot": (-0.3, -0.95, -0.06, 0.10),
        "focal_length": 28.0,
        "aperture": 45.0,  # ~85° FOV
    },
    "angled_down": {
        "pos": (0.02, 0.05, -0.01),
        "rot": (-0.45, -0.89, -0.03, 0.06),
        "focal_length": 32.0,
        "aperture": 38.0,  # ~78° FOV
    }
}

ALTERNATIVE_FRONT_POSITIONS = {
    "high_overview": {
        "pos": (-0.1, -1.0, 1.0),
        "rot": (0.3, -0.95, 0.0, 0.0),
        "focal_length": 20.0,
        "aperture": 50.0,  # ~90° FOV
    },
    "side_angle": {
        "pos": (-0.5, -0.6, 0.8),
        "rot": (0.15, -0.85, 0.35, -0.35),
        "focal_length": 26.0,
        "aperture": 42.0,  # ~82° FOV
    },
    "close_workspace": {
        "pos": (-0.3, -0.6, 0.6),
        "rot": (0.25, -0.97, 0.0, 0.0),
        "focal_length": 30.0,
        "aperture": 40.0,  # ~80° FOV
    }
}

# Viewer camera positions for development and debugging
SANDWICH_VIEWER_POSITIONS = {
    "diagonal_overview": {
        "eye": (1.5, -1.5, 1.8),
        "lookat": (0.0, 0.0, 0.9),
    },
    "side_view": {
        "eye": (2.0, 0.0, 1.5),
        "lookat": (0.0, 0.0, 0.9),
    },
    "front_elevated": {
        "eye": (0.0, -2.0, 2.0),
        "lookat": (0.0, 0.0, 0.9),
    },
    "close_action": {
        "eye": (0.5, -1.0, 1.2),
        "lookat": (0.0, 0.0, 0.9),
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_sandwich_camera_config(env_cfg):
    """
    Apply optimized camera configurations to environment config.
    
    Args:
        env_cfg: Environment configuration object
    """
    env_cfg.scene.wrist = SANDWICH_WRIST_CAMERA_CFG
    env_cfg.scene.front = SANDWICH_FRONT_CAMERA_CFG
    return env_cfg

def apply_alternative_cameras(env_cfg, wrist_style="close_detail", front_style="high_overview"):
    """
    Apply alternative camera configurations.
    
    Args:
        env_cfg: Environment configuration object
        wrist_style: Style from ALTERNATIVE_WRIST_POSITIONS
        front_style: Style from ALTERNATIVE_FRONT_POSITIONS
    """
    if wrist_style in ALTERNATIVE_WRIST_POSITIONS:
        wrist_config = ALTERNATIVE_WRIST_POSITIONS[wrist_style]
        env_cfg.scene.wrist.offset.pos = wrist_config["pos"]
        env_cfg.scene.wrist.offset.rot = wrist_config["rot"]
        env_cfg.scene.wrist.spawn.focal_length = wrist_config["focal_length"]
        env_cfg.scene.wrist.spawn.horizontal_aperture = wrist_config["aperture"]
    
    if front_style in ALTERNATIVE_FRONT_POSITIONS:
        front_config = ALTERNATIVE_FRONT_POSITIONS[front_style]
        env_cfg.scene.front.offset.pos = front_config["pos"]
        env_cfg.scene.front.offset.rot = front_config["rot"]
        env_cfg.scene.front.spawn.focal_length = front_config["focal_length"]
        env_cfg.scene.front.spawn.horizontal_aperture = front_config["aperture"]
    
    return env_cfg

def apply_viewer_position(env_cfg, style="diagonal_overview"):
    """
    Apply viewer camera position.
    
    Args:
        env_cfg: Environment configuration object
        style: Style from SANDWICH_VIEWER_POSITIONS
    """
    if style in SANDWICH_VIEWER_POSITIONS:
        viewer_config = SANDWICH_VIEWER_POSITIONS[style]
        env_cfg.viewer.eye = viewer_config["eye"]
        env_cfg.viewer.lookat = viewer_config["lookat"]
    
    return env_cfg

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

def example_usage():
    """
    Example of how to use these camera configurations.
    """
    # In your environment configuration file:
    
    # Method 1: Direct replacement
    # from sandwich_camera_config import SANDWICH_WRIST_CAMERA_CFG, SANDWICH_FRONT_CAMERA_CFG
    # 
    # @configclass
    # class YourEnvCfg(SingleArmTaskEnvCfg):
    #     def __post_init__(self):
    #         super().__post_init__()
    #         self.scene.wrist = SANDWICH_WRIST_CAMERA_CFG
    #         self.scene.front = SANDWICH_FRONT_CAMERA_CFG
    
    # Method 2: Using utility functions
    # from sandwich_camera_config import apply_sandwich_camera_config, apply_viewer_position
    # 
    # @configclass
    # class YourEnvCfg(SingleArmTaskEnvCfg):
    #     def __post_init__(self):
    #         super().__post_init__()
    #         apply_sandwich_camera_config(self)
    #         apply_viewer_position(self, "diagonal_overview")
    
    # Method 3: Alternative configurations
    # from sandwich_camera_config import apply_alternative_cameras
    # 
    # @configclass
    # class YourEnvCfg(SingleArmTaskEnvCfg):
    #     def __post_init__(self):
    #         super().__post_init__()
    #         apply_alternative_cameras(self, "wide_view", "side_angle")
    
    pass

if __name__ == "__main__":
    print("Sandwich Assembly Camera Configuration")
    print("=====================================")
    print(f"Available wrist camera styles: {list(ALTERNATIVE_WRIST_POSITIONS.keys())}")
    print(f"Available front camera styles: {list(ALTERNATIVE_FRONT_POSITIONS.keys())}")
    print(f"Available viewer positions: {list(SANDWICH_VIEWER_POSITIONS.keys())}")
