#!/usr/bin/env bash

set -eo pipefail

source /home/liutong/.bashrc
conda create -n openvla python=3.10 -y
conda activate openvla
conda install nvidia::cuda==13.0.0 -y

export OPENVLA_ROOT="/home/liutong/Projects/openvla"
export LIBERO_ROOT="/home/liutong/Projects/lerobot-libero-NewViewCameras"
export MAX_JOBS="4"

export TOKENIZERS_PARALLELISM=false

cd "${OPENVLA_ROOT}"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e "${OPENVLA_ROOT}"
uv pip install packaging ninja
env MAX_JOBS="${MAX_JOBS}" uv pip install "flash-attn==2.8.3" --no-build-isolation --no-deps --verbose
uv pip install -r "${OPENVLA_ROOT}/experiments/robot/libero/libero_requirements.txt"
uv pip install -e "${LIBERO_ROOT}/third_party/robosuite"
uv pip install -e "${LIBERO_ROOT}"
python -c "import experiments.robot.libero.run_libero_eval as m; print(m.GenerateConfig)"
