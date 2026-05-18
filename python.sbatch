#!/bin/bash
#SBATCH --job-name=Object_Detection_KD
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24-00:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

set -e

# Create logs folder
mkdir -p logs

# Load environment
. ~/abishek_env/bin/activate

# Debug info
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Go to project
cd ~/Detection_with_Distillation || exit 1

# Run training with optional checkpoint
# Usage: sbatch python.sbatch (starts from scratch)
# Usage: sbatch --export=CHECKPOINT=/home/aryan_s2/Detection_with_Distillation/checkpoints_distill/distill_final.pth python.sbatch (resumes from checkpoint)
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
    python3 train_distillation.py --checkpoint "$CHECKPOINT"
else
    echo "Starting training from scratch"
    python3 train_distillation.py
fi
