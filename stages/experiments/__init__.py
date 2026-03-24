"""Experiment index builders for Stage 3 training.

Each module defines the channel selection for one experiment configuration.
Import all builders from here for convenience.

    from crop_mapping_pipeline.stages.experiments import (
        parse_date, build_local_band_map,
        build_exp_A_indices, build_exp_A_v2_indices,
        build_exp_B_indices,
        build_exp_C_indices, build_exp_C_indices_projected,
        build_exp_C_v2_indices, build_exp_C_v2_indices_projected,
        build_exp_C_v2_rf_indices,
        build_exp_C_v3_indices,
        build_exp_D_indices,
        build_exp_D_v2_indices,
    )
"""

from crop_mapping_pipeline.stages.experiments.base import (
    parse_date,
    build_local_band_map,
)
from crop_mapping_pipeline.stages.experiments.exp_a import build_exp_A_indices
from crop_mapping_pipeline.stages.experiments.exp_a_v2 import build_exp_A_v2_indices
from crop_mapping_pipeline.stages.experiments.exp_b import build_exp_B_indices
from crop_mapping_pipeline.stages.experiments.exp_c import (
    build_exp_C_indices,
    build_exp_C_indices_projected,
)
from crop_mapping_pipeline.stages.experiments.exp_c_v2 import (
    build_exp_C_v2_indices,
    build_exp_C_v2_indices_projected,
)
from crop_mapping_pipeline.stages.experiments.exp_c_v2_rf import build_exp_C_v2_rf_indices
from crop_mapping_pipeline.stages.experiments.exp_c_v3 import build_exp_C_v3_indices
from crop_mapping_pipeline.stages.experiments.exp_d import build_exp_D_indices
from crop_mapping_pipeline.stages.experiments.exp_d_v2 import build_exp_D_v2_indices
from crop_mapping_pipeline.stages.experiments.registry import (
    ExperimentConfig,
    build_registry,
    expand_exp_keys,
)

__all__ = [
    "parse_date",
    "build_local_band_map",
    "build_exp_A_indices",
    "build_exp_A_v2_indices",
    "build_exp_B_indices",
    "build_exp_C_indices",
    "build_exp_C_indices_projected",
    "build_exp_C_v2_indices",
    "build_exp_C_v2_indices_projected",
    "build_exp_C_v2_rf_indices",
    "build_exp_C_v3_indices",
    "build_exp_D_indices",
    "build_exp_D_v2_indices",
    "ExperimentConfig",
    "build_registry",
    "expand_exp_keys",
]
