from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from leisaac.assets.scenes.kitchen import KITCHEN_WITH_SANDWICH_CFG, KITCHEN_WITH_SANDWICH_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import SingleArmTaskSceneCfg, SingleArmTaskEnvCfg, SingleArmTerminationsCfg, SingleArmObservationsCfg


@configclass
class AssembleSandwichSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the assemble sandwich task."""

    scene: AssetBaseCfg = KITCHEN_WITH_SANDWICH_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class AssembleSandwichEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the assemble sandwich environment."""

    scene: AssembleSandwichSceneCfg = AssembleSandwichSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    terminations: SingleArmTerminationsCfg = SingleArmTerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Optimized viewer camera for simplified table workspace
        self.viewer.eye = (1.5, -1.5, 1.8)     # Elevated diagonal view
        self.viewer.lookat = (0.0, 0.0, 0.9)   # Looking at table center

        # Robot position for simplified table workspace
        self.scene.robot.init_state.pos = (2.7, -2, 0.81)  # Closer to desk/table

        # Optimized camera positions for sandwich assembly task
        # Wrist camera: Better view of ingredients and manipulation
        self.scene.wrist.offset.pos = (0.02, 0.08, -0.03)
        self.scene.wrist.offset.rot = (-0.35, -0.93, -0.05, 0.08)
        self.scene.wrist.spawn.focal_length = 32.0
        self.scene.wrist.spawn.horizontal_aperture = 40.0  # ~80° FOV
        self.scene.wrist.spawn.clipping_range = (0.005, 20.0)

        # Front camera: Complete table workspace overview
        self.scene.front.offset.pos = (-0.2, -0.8, 0.7)
        self.scene.front.offset.rot = (0.2, -0.98, 0.0, 0.0)
        self.scene.front.spawn.focal_length = 24.0
        self.scene.front.spawn.horizontal_aperture = 45.0  # ~85° FOV

        parse_usd_and_create_subassets(KITCHEN_WITH_SANDWICH_USD_PATH, self)
