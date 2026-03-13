"""Utility helpers for LIBERO HDF5 finetuning."""

import json
import os
import random
from pathlib import Path

import draccus
import numpy as np
import torch
import yaml

from libero_finetune.config import LiberoFinetuneConfig


def set_random_seed(seed: int) -> None:
    """Sets process-local random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """Creates a directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def build_experiment_id(cfg: LiberoFinetuneConfig) -> str:
    """Builds a stable experiment identifier for a training run."""
    model_name = cfg.vla_path.split("/")[-1]
    effective_batch_size = cfg.batch_size * cfg.grad_accumulation_steps
    parts = [
        model_name,
        cfg.benchmark_name,
        cfg.camera_view,
        f"b{effective_batch_size}",
        f"lr-{cfg.learning_rate}",
    ]

    if cfg.use_lora:
        parts.append(f"lora-r{cfg.lora_rank}")
    if cfg.use_quantization:
        parts.append("q-4bit")
    if cfg.image_aug:
        parts.append("image_aug")
    if cfg.run_id_note:
        parts.append(cfg.run_id_note)

    return "+".join(parts)


def save_config_artifacts(cfg: LiberoFinetuneConfig, run_dir: Path) -> None:
    """Saves both YAML and JSON copies of the active config."""
    yaml_path = run_dir / "config.yaml"
    json_path = run_dir / "config.json"

    with open(yaml_path, "w", encoding="utf-8") as file_obj:
        draccus.dump(cfg, file_obj)

    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        yaml_config = yaml.safe_load(yaml_file)

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(yaml_config, json_file, indent=2)
