"""Command-line entrypoint for LIBERO HDF5 finetuning."""

import draccus

from libero_finetune.config import LiberoFinetuneConfig
from libero_finetune.trainer import LiberoFinetuneTrainer


@draccus.wrap()
def main(cfg: LiberoFinetuneConfig) -> None:
    """Runs OpenVLA finetuning on a single LIBERO benchmark and camera view."""
    trainer = LiberoFinetuneTrainer(cfg)
    trainer.run()
