#!/usr/bin/env bash
#SBATCH --job-name=aimers-w8a8-final
#SBATCH --partition=batch_eebme_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/w8a8-final-%j.out

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:?}"
cd "$REPO_DIR"

source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate aimers

echo "===== SOURCE VERSION ====="
git rev-parse HEAD
git branch --show-current

echo "===== ENVIRONMENT ====="
which python
python --version
nvidia-smi

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")
print("GPU:", torch.cuda.get_device_name(0))
PY

echo "===== SAFETY CHECK ====="

if [[ -e model/model || -L model/model ]]; then
    echo "ERROR: Existing model output detected."
    exit 1
fi

if [[ -e model/submit_lm_w8a8_spanish_newdata.zip ]]; then
    echo "ERROR: Existing ZIP detected."
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" && ! -f model/.env ]]; then
    echo "ERROR: HF_TOKEN is not configured."
    exit 1
fi

echo "===== QUANTIZATION ====="
cd model
python baseline_lm_w8a8_spanish.py

echo "===== COMPLETED ====="
