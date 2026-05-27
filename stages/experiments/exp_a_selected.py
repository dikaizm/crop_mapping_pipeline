# Shim — logic merged into exp_a.py.
from crop_mapping_pipeline.stages.experiments.exp_a import (
    build_single_date_selected_indices,
    build_single_date_selected_indices as build_exp_A_selected_indices,
)

__all__ = ["build_single_date_selected_indices", "build_exp_A_selected_indices"]
