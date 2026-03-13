"""Configuration objects for LIBERO HDF5 finetuning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from libero_finetune.constants import (
    DEFAULT_LIBERO_DATA_ROOT,
    DEFAULT_OPENVLA_CHECKPOINT,
    SUPPORTED_BENCHMARKS,
    SUPPORTED_CAMERA_VIEWS,
)


@dataclass
class LiberoFinetuneConfig:
    """Config for single-view OpenVLA finetuning on LIBERO HDF5 datasets."""

    vla_path: str = DEFAULT_OPENVLA_CHECKPOINT

    data_root_dir: Path = DEFAULT_LIBERO_DATA_ROOT
    benchmark_name: str = "libero_spatial"
    camera_view: str = "agentview_rgb"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")

    batch_size: int = 8
    max_steps: int = 50_000
    save_steps: int = 5_000
    log_every_n_steps: int = 10
    learning_rate: float = 5e-4
    grad_accumulation_steps: int = 2
    image_aug: bool = True
    rotate_image_180: bool = True
    num_workers: int = 4
    random_seed: int = 7
    save_latest_checkpoint_only: bool = True

    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False

    use_wandb: bool = False
    wandb_project: str = "openvla"
    wandb_entity: str = "stanford-voltron"
    run_id_note: Optional[str] = None

    def validate(self) -> None:
        """Validates the config before training starts."""
        if self.benchmark_name not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark `{self.benchmark_name}`. Choose from {SUPPORTED_BENCHMARKS}."
            )
        if self.camera_view not in SUPPORTED_CAMERA_VIEWS:
            raise ValueError(
                f"Unsupported camera view `{self.camera_view}`. Choose from {SUPPORTED_CAMERA_VIEWS}."
            )
        if self.batch_size <= 0:
            raise ValueError("`batch_size` must be positive.")
        if self.max_steps <= 0:
            raise ValueError("`max_steps` must be positive.")
        if self.save_steps <= 0:
            raise ValueError("`save_steps` must be positive.")
        if self.grad_accumulation_steps <= 0:
            raise ValueError("`grad_accumulation_steps` must be positive.")
        if self.num_workers < 0:
            raise ValueError("`num_workers` must be non-negative.")
        if self.use_quantization and not self.use_lora:
            raise ValueError("Quantized finetuning is only supported together with LoRA.")
