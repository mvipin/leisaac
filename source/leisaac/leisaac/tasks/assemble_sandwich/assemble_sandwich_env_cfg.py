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

        self.viewer.eye = (2.5, -1.0, 1.3)
        self.viewer.lookat = (3.6, -0.4, 1.0)

        self.scene.robot.init_state.pos = (3.6, -0.65, 0.89)

        parse_usd_and_create_subassets(KITCHEN_WITH_SANDWICH_USD_PATH, self)
