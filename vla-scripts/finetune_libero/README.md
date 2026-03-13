# LIBERO Single-View Finetuning

This directory adds a minimal OpenVLA finetuning pipeline for the multiview LIBERO
HDF5 dataset at `/mnt/HDD-6940GB/Dataset/LIBERO-datasets-multiview`.

What exists in the upstream repository:

- `experiments/robot/libero/run_libero_eval.py`: LIBERO evaluation.
- `experiments/robot/libero/regenerate_libero_dataset.py`: LIBERO HDF5 replay / regeneration.
- `vla-scripts/finetune.py`: generic RLDS-based OpenVLA LoRA finetuning.

What does not exist upstream:

- A direct trainer that reads raw LIBERO HDF5 demonstrations and fine-tunes OpenVLA
  from a selected LIBERO camera view.

This directory fills that gap with a single-view HDF5 trainer. It keeps the official
OpenVLA LoRA recipe and writes merged Hugging Face checkpoints directly, so no extra
conversion step is needed for the models produced here.

## Default usage

```bash
cd /home/szliutong/Projects/openvla
python vla-scripts/finetune_libero/train.py \
  --config_path vla-scripts/finetune_libero/agentview_rgb/config.yaml
```

To switch view, point `--config_path` at one of:

- `vla-scripts/finetune_libero/agentview_rgb/config.yaml`
- `vla-scripts/finetune_libero/operation_topview_rgb/config.yaml`
- `vla-scripts/finetune_libero/operation_leftview_rgb/config.yaml`
- `vla-scripts/finetune_libero/operation_rightview_rgb/config.yaml`
- `vla-scripts/finetune_libero/operation_backview_rgb/config.yaml`

Each config defaults to `libero_spatial`. To train another benchmark, edit
`benchmark_name` in the chosen config file.
