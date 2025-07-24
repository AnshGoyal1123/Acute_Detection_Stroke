#!/bin/bash

#SBATCH --account=rsteven1_gpu
#SBATCH --job-name=auis
#SBATCH --nodes=4
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --mem=50G
#SBATCH --export=ALL

# Load any necessary modules (if needed)
# module load anaconda3 cuda/11.7

# Activate your conda environment (edit as needed)
source ~/.bashrc
conda activate strokeai

# Set environment variables for NCCL
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^lo,docker

# Path to your script
SCRIPT_PATH=/home/agoyal19/My_Work/Segmentation/train.py

# Launch the script using torchrun (recommended for PyTorch DDP)
torchrun --nnodes=${SLURM_JOB_NUM_NODES} \
         --nproc_per_node=4 \
         --rdzv_id=${SLURM_JOB_ID} \
         --rdzv_backend=c10d \
         --rdzv_endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):29500 \
         $SCRIPT_PATH --model_name AUIS --wandb