#!/bin/bash
#SBATCH --job-name=grpo_smoke
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvidia-gpu&cuda13"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0:45:00
#SBATCH -o slurm/logs/%x-%j.out

# Submit from the repo root:  sbatch slurm/quick_test_grpo.sbatch [extra quick_test_grpo.py flags]
# --constraint picks any NVIDIA GPU on a CUDA 13 driver (A100/A40/L40S/H100/H200): all support bf16,
# and the envs' torch builds need CUDA 13.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/scratch/conda_envs/math_grpo

export HF_HOME=~/scratch/hf_cache
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
python quick_test_grpo.py --output_dir "${TMPDIR:-/tmp}/grpo_smoke" "$@"
