#!/usr/bin/bash

#SBATCH -J aimers_model5
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
echo "[0/5] Hugging Face assets download"
echo "================================================="
BASE_MODEL="LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_1="m-a-p/Code-Feedback"
DATASET_2="MyeongHo0621/korean-quality-cleaned"
DATASET_3="openai/gsm8k"

python -c "
from huggingface_hub import snapshot_download
import os
cache_dir = os.environ.get('HF_HOME')

def download_asset(repo_id, repo_type='model'):
    print(f'Starting download for {repo_type}: {repo_id}...')
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir_use_symlinks=False, cache_dir=cache_dir)
    print(f'[SUCCESS] {repo_id} download complete.\\n')

download_asset('$BASE_MODEL', repo_type='model')
download_asset('$DATASET_1', repo_type='dataset')
download_asset('$DATASET_2', repo_type='dataset')
download_asset('$DATASET_3', repo_type='dataset')
"

echo "================================================="
echo "[1/5] Build curated training dataset"
echo "================================================="
python build_curated_dataset.py \
  --base_model_id "$BASE_MODEL" \
  --output_dir ./curated_train_dataset \
  --target_size 10000

echo "================================================="
echo "[2/5] QLoRA training"
echo "================================================="
python train_qlora.py \
  --base_model_id "$BASE_MODEL" \
  --dataset_path ./curated_train_dataset \
  --adapter_output_dir ./adapter_model5 \
  --max_samples 10000 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 3e-5

echo "================================================="
echo "[3/5] Build standalone model + submit.zip"
echo "================================================="
python build_submission_from_adapter.py \
  --base_model_id "$BASE_MODEL" \
  --adapter_dir ./adapter_model5 \
  --output_dir ./model \
  --zip_name submit \
  --nsamples 512 \
  --seqlen 1024 \
  --iters 450

echo "================================================="
echo "[4/5] Preflight check"
echo "================================================="
if [ -f "submit.zip" ]; then
  python ../preflight_check.py --zip-path submit.zip
  echo "[DONE] Model5 workflow completed."
else
  echo "[ERROR] Failed to create submit.zip"
  exit 1
fi

