source /home/szliutong/.zshrc

conda activate openvla
cd /home/szliutong/Projects/openvla

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_libero/finetune.py