"""Training loop for LIBERO HDF5 finetuning."""

import os
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import tqdm
import wandb
from accelerate import PartialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers.modeling_outputs import CausalLMOutputWithPast

from libero_finetune.config import LiberoFinetuneConfig
from libero_finetune.dataset import LiberoMultiviewHDF5Dataset
from libero_finetune.modeling import (
    load_processor_and_model,
    merge_lora_weights,
    unwrap_model,
)
from libero_finetune.utils import build_experiment_id, ensure_dir, save_config_artifacts, set_random_seed
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LiberoFinetuneTrainer:
    """Single-view LIBERO LoRA finetuner for OpenVLA."""

    def __init__(self, cfg: LiberoFinetuneConfig) -> None:
        cfg.validate()
        self.cfg = cfg

        if not torch.cuda.is_available():
            raise RuntimeError("LIBERO finetuning requires at least one CUDA device.")

        self.distributed_state = PartialState()
        self.device_id = self.distributed_state.local_process_index
        self.world_size = self.distributed_state.num_processes

        if self.world_size > 1 and self.cfg.use_quantization:
            raise NotImplementedError("Quantized finetuning is only supported in single-process mode.")

        torch.cuda.set_device(self.device_id)
        torch.cuda.empty_cache()
        set_random_seed(self.cfg.random_seed + self.distributed_state.process_index)

        self.exp_id = build_experiment_id(self.cfg)
        self.run_dir = self.cfg.run_root_dir / self.exp_id
        self.adapter_dir = self.cfg.adapter_tmp_dir / self.exp_id

    def run(self) -> None:
        """Runs the full finetuning workflow."""
        ensure_dir(self.run_dir)
        ensure_dir(self.adapter_dir)

        if self.distributed_state.is_main_process:
            save_config_artifacts(self.cfg, self.run_dir)

        processor, vla = load_processor_and_model(self.cfg, self.device_id)

        prompt_builder_fn = (
            PurePromptBuilder if "v01" not in self.cfg.vla_path else VicunaV15ChatPromptBuilder
        )
        action_tokenizer = ActionTokenizer(processor.tokenizer)
        dataset = LiberoMultiviewHDF5Dataset(
            data_root_dir=self.cfg.data_root_dir,
            benchmark_name=self.cfg.benchmark_name,
            camera_view=self.cfg.camera_view,
            action_tokenizer=action_tokenizer,
            base_tokenizer=processor.tokenizer,
            image_transform=processor.image_processor.apply_transform,
            prompt_builder_fn=prompt_builder_fn,
            rotate_image_180=self.cfg.rotate_image_180,
        )

        if self.distributed_state.is_main_process:
            save_dataset_statistics(deepcopy(dataset.dataset_statistics), self.run_dir)

        sampler: Optional[DistributedSampler]
        sampler = None
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.distributed_state.process_index,
                shuffle=True,
            )
            vla = DDP(vla, device_ids=[self.device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

        collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collator,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

        optimizer = AdamW([param for param in vla.parameters() if param.requires_grad], lr=self.cfg.learning_rate)

        if self.distributed_state.is_main_process and self.cfg.use_wandb:
            wandb.init(
                entity=self.cfg.wandb_entity,
                project=self.cfg.wandb_project,
                name=f"ft+{self.exp_id}",
            )

        recent_losses = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_action_accuracies = deque(maxlen=self.cfg.grad_accumulation_steps)
        recent_l1_losses = deque(maxlen=self.cfg.grad_accumulation_steps)

        global_step = 0
        micro_step = 0
        progress = tqdm.tqdm(
            total=self.cfg.max_steps,
            leave=False,
            disable=not self.distributed_state.is_main_process,
        )

        try:
            vla.train()
            optimizer.zero_grad()
            epoch = 0

            while global_step < self.cfg.max_steps:
                if sampler is not None:
                    sampler.set_epoch(epoch)

                for batch in dataloader:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output: CausalLMOutputWithPast = vla(
                            input_ids=batch["input_ids"].to(self.device_id),
                            attention_mask=batch["attention_mask"].to(self.device_id),
                            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(self.device_id),
                            labels=batch["labels"].to(self.device_id),
                        )
                        loss = output.loss

                    (loss / self.cfg.grad_accumulation_steps).backward()
                    metrics = self._compute_batch_metrics(output, batch, action_tokenizer, loss.item(), vla)
                    recent_losses.append(metrics["loss"])
                    recent_action_accuracies.append(metrics["action_accuracy"])
                    recent_l1_losses.append(metrics["l1_loss"])

                    micro_step += 1
                    if micro_step % self.cfg.grad_accumulation_steps != 0:
                        continue

                    global_step += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    progress.update(1)

                    smoothed_metrics = {
                        "train_loss": sum(recent_losses) / len(recent_losses),
                        "action_accuracy": sum(recent_action_accuracies) / len(recent_action_accuracies),
                        "l1_loss": sum(recent_l1_losses) / len(recent_l1_losses),
                    }

                    if self.distributed_state.is_main_process and global_step % self.cfg.log_every_n_steps == 0:
                        print(
                            f"[step {global_step}] "
                            f"loss={smoothed_metrics['train_loss']:.4f} "
                            f"acc={smoothed_metrics['action_accuracy']:.4f} "
                            f"l1={smoothed_metrics['l1_loss']:.4f}"
                        )
                        if self.cfg.use_wandb:
                            wandb.log(smoothed_metrics, step=global_step)

                    if global_step % self.cfg.save_steps == 0:
                        self._save_checkpoint(global_step, vla, processor, dataset.dataset_statistics)

                    if global_step >= self.cfg.max_steps:
                        break

                epoch += 1

            self._save_checkpoint(global_step, vla, processor, dataset.dataset_statistics, is_final=True)

        finally:
            progress.close()
            dataset.close()
            if self.distributed_state.is_main_process and self.cfg.use_wandb:
                wandb.finish()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def _compute_batch_metrics(
        self,
        output: CausalLMOutputWithPast,
        batch: Dict[str, torch.Tensor],
        action_tokenizer: ActionTokenizer,
        loss_value: float,
        model,
    ) -> Dict[str, float]:
        """Computes training metrics for logging."""
        unwrapped_model = unwrap_model(model)
        num_patches = unwrapped_model.vision_backbone.featurizer.patch_embed.num_patches
        action_logits = output.logits[:, num_patches:-1]
        action_preds = action_logits.argmax(dim=2)
        action_gt = batch["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        continuous_actions_pred = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].detach().cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_gt[mask].detach().cpu().numpy())
        )
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

        return {
            "loss": loss_value,
            "action_accuracy": action_accuracy.item(),
            "l1_loss": action_l1_loss.item(),
        }

    def _save_checkpoint(
        self,
        step: int,
        model,
        processor,
        dataset_statistics: Dict[str, Dict],
        is_final: bool = False,
    ) -> None:
        """Saves either the latest checkpoint or a step-specific checkpoint."""
        if self.cfg.save_latest_checkpoint_only:
            target_dir = self.run_dir
            adapter_save_dir = self.adapter_dir
        else:
            suffix = "final" if is_final else f"step-{step:06d}"
            target_dir = self.run_dir / suffix
            adapter_save_dir = self.adapter_dir / suffix
            ensure_dir(target_dir)
            ensure_dir(adapter_save_dir)

        if self.cfg.use_lora:
            if self.distributed_state.is_main_process:
                unwrap_model(model).save_pretrained(adapter_save_dir)
            self._barrier()

            merged_vla = merge_lora_weights(self.cfg.vla_path, adapter_save_dir)
            if self.distributed_state.is_main_process:
                processor.save_pretrained(target_dir)
                save_dataset_statistics(deepcopy(dataset_statistics), target_dir)
                merged_vla.save_pretrained(target_dir)
                print(f"Saved merged HF checkpoint at: {target_dir}")
        else:
            if self.distributed_state.is_main_process:
                processor.save_pretrained(target_dir)
                save_dataset_statistics(deepcopy(dataset_statistics), target_dir)
                unwrap_model(model).save_pretrained(target_dir)
                print(f"Saved HF checkpoint at: {target_dir}")

        self._barrier()

    @staticmethod
    def _barrier() -> None:
        """Runs a distributed barrier when multiple processes are active."""
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
