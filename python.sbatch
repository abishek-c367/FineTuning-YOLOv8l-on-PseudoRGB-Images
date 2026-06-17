#!/bin/bash
#SBATCH --job-name=Object_Detection_KD
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24-00:00:00
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log

# Exit on critical failures
set -e

# -------------------------------
# Create logs directory
# -------------------------------
mkdir -p logs

# -------------------------------
# Activate virtual environment
# -------------------------------
source ~/abishek_env/bin/activate

# -------------------------------
# Debug / System Information
# -------------------------------
echo "========================================"
echo "SLURM JOB INFORMATION"
echo "========================================"

echo "Job ID            : $SLURM_JOB_ID"
echo "Job Name          : $SLURM_JOB_NAME"
echo "Node Name         : $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

echo ""
echo "Hostname:"
hostname

echo ""
echo "Python Path:"
which python3

echo ""
echo "PyTorch Version:"
python3 -c "import torch; print(torch.__version__)"

echo ""
echo "CUDA Version Used By PyTorch:"
python3 -c "import torch; print(torch.version.cuda)"

echo ""
echo "CUDA Available:"
python3 -c "import torch; print(torch.cuda.is_available())"

echo ""
echo "GPU Name:"
python3 -c "
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
"

echo ""
echo "Kernel NVIDIA Driver Version:"
cat /proc/driver/nvidia/version || true

echo ""
echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH

echo ""
echo "Trying nvidia-smi..."
nvidia-smi || echo "WARNING: NVML driver/library mismatch detected. CUDA compute still appears functional."

echo "========================================"

# -------------------------------
# Go to project directory
# -------------------------------
cd ~/Detection_with_Distillation || exit 1


# -------------------------------
# Start training
# -------------------------------
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint:"
    echo "$CHECKPOINT"

    python3 train_distillation.py \
        --checkpoint "$CHECKPOINT"

else
    echo "Starting training from scratch"

    python3 train_distillation.py
fi

echo "========================================"
echo "Training job completed."
echo "========================================"