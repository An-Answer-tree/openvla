"""Shared constants for LIBERO HDF5 finetuning."""

from pathlib import Path


DEFAULT_OPENVLA_CHECKPOINT = "openvla/openvla-7b"
DEFAULT_LIBERO_DATA_ROOT = Path("/mnt/HDD-6940GB/Dataset/LIBERO-datasets-multiview")

SUPPORTED_BENCHMARKS = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
    "libero_90",
)

SUPPORTED_CAMERA_VIEWS = (
    "agentview_rgb",
    "operation_topview_rgb",
    "operation_leftview_rgb",
    "operation_rightview_rgb",
    "operation_backview_rgb",
)

ACTION_NORMALIZATION_MASK = (True, True, True, True, True, True, False)
NORMALIZATION_EPS = 1e-8
