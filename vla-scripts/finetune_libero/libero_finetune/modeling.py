"""Model-loading helpers for LIBERO HDF5 finetuning."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

from libero_finetune.config import LiberoFinetuneConfig
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_autoclasses() -> None:
    """Registers OpenVLA custom classes with Hugging Face auto classes."""
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def build_quantization_config(cfg: LiberoFinetuneConfig) -> Optional[BitsAndBytesConfig]:
    """Builds a 4-bit quantization config when requested."""
    if not cfg.use_quantization:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )


def load_processor_and_model(
    cfg: LiberoFinetuneConfig,
    device_id: int,
) -> Tuple[PrismaticProcessor, OpenVLAForActionPrediction]:
    """Loads the OpenVLA processor and model for finetuning."""
    register_openvla_autoclasses()

    quantization_config = build_quantization_config(cfg)
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    return processor, vla


def unwrap_model(model):
    """Returns the underlying model when wrapped in DDP."""
    return model.module if hasattr(model, "module") else model


def merge_lora_weights(vla_path: str, adapter_dir: Path) -> OpenVLAForActionPrediction:
    """Loads base model plus adapters and returns a merged HF model."""
    register_openvla_autoclasses()
    base_vla = AutoModelForVision2Seq.from_pretrained(
        vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
    return merged_vla.merge_and_unload()
