#!/usr/bin/bash

#SBATCH -J aimers_full_workflow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export HF_HOME="/data/$USER/.cache/huggingface"
export TORCH_HOME="/data/$USER/.cache/torch"
export PIP_CACHE_DIR="/data/$USER/.cache/pip"

source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate aimers

echo "================================================="
echo "[0/5] Hugging Face Assets Download"
echo "================================================="
BASE_MODEL="LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_1="m-a-p/Code-Feedback"
DATASET_2="MyeongHo0621/korean-quality-cleaned"

python -c "
from huggingface_hub import snapshot_download
import os
cache_dir = os.environ.get('HF_HOME')

def download_asset(repo_id, repo_type='model'):
    print(f'Starting download for {repo_type}: {repo_id}...')
    try:
        snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir_use_symlinks=False, cache_dir=cache_dir)
        print(f'[SUCCESS] {repo_id} download complete.\\n')
    except Exception as e:
        print(f'[ERROR] {repo_id} download failed: {e}')
        exit(1)

download_asset('$BASE_MODEL', repo_type='model')
download_asset('$DATASET_1', repo_type='dataset')
download_asset('$DATASET_2', repo_type='dataset')
"

echo "================================================="
echo "[1/5] Main Stage (Model Loading & Quantization)"
echo "================================================="
QUANT_CHECK_DIR="./model_q4_train"

if [ -d "$QUANT_CHECK_DIR" ] \
  && [ -f "$QUANT_CHECK_DIR/model.safetensors" ] \
  && python -c "import sys; from safetensors import safe_open; p=sys.argv[1]; f=safe_open(p, framework='pt'); next(iter(f.keys()), None)" "$QUANT_CHECK_DIR/model.safetensors" >/dev/null 2>&1; then
    echo "[SKIP] Reusing valid quantized checkpoint at $QUANT_CHECK_DIR"
else
    echo "[RUN] main.py (model loading and quantization)..."
    python main.py
fi

echo "================================================="
echo "[2/5] Fine-tuning Stage (train_qlora.py)"
echo "================================================="
python train_qlora.py \
    --model_dir "$QUANT_CHECK_DIR" \
    --dataset_name "$DATASET_2" \
    --dataset_config default \
    --output_dir ./adapter_model4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4

echo "================================================="
echo "[3/5] Pre-check (Adapter Check)"
echo "================================================="
if [ ! -d "./adapter_model4" ]; then
  echo "[FAIL] Missing adapter output: ./adapter_model4"
  exit 1
fi

echo "================================================="
echo "[4/5] Build standalone model + submit.zip"
echo "================================================="
python build_submission_from_adapter.py \
  --adapter_dir ./adapter_model4 \
  --output_dir ./model \
  --zip_name submit \
  --nsamples 384 \
  --seqlen 1024 \
  --iters 350

echo "================================================="
echo "[5/5] Preflight check"
echo "================================================="
if [ -f "submit.zip" ]; then
    python ../preflight_check.py --zip-path submit.zip
    echo "[DONE] Full workflow completed."
else
    echo "[ERROR] Failed to create submit.zip"
    exit 1
fi
