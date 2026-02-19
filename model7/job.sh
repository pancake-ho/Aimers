#!/usr/bin/bash

#SBATCH -J aimers_model7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [ ! -f "train_and_merge.py" ]; then
  if [ -f "model7/train_and_merge.py" ]; then
    cd "model7"
  elif [ -f "Aimers/model7/train_and_merge.py" ]; then
    cd "Aimers/model7"
  else
    echo "[ERROR] Could not locate model7 working directory from: $(pwd)"
    exit 1
  fi
fi

export HF_HOME="/data/$USER/.cache/huggingface"
export TORCH_HOME="/data/$USER/.cache/torch"
export PIP_CACHE_DIR="/data/$USER/.cache/pip"

source "/data/$USER/anaconda3/etc/profile.d/conda.sh"
conda activate aimers

echo "================================================="
echo "[0/5] Hugging Face assets download"
echo "================================================="
BASE_MODEL="LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_KO="MyeongHo0621/korean-quality-cleaned"
DATASET_CODE="m-a-p/Code-Feedback"

python -c "
from huggingface_hub import snapshot_download
import os
cache_dir = os.environ.get('HF_HOME')

def download_asset(repo_id, repo_type='model'):
    print(f'Starting download for {repo_type}: {repo_id}...')
    snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir_use_symlinks=False, cache_dir=cache_dir)
    print(f'[SUCCESS] {repo_id} download complete.\\n')

download_asset('$BASE_MODEL', repo_type='model')
download_asset('$DATASET_KO', repo_type='dataset')
download_asset('$DATASET_CODE', repo_type='dataset')
"

echo "================================================="
echo "[1/5] Train and merge (model7)"
echo "================================================="
python train_and_merge.py \
  --base_model_id "$BASE_MODEL" \
  --adapter_output_dir ./adapter_model7 \
  --merged_output_dir ./merged_model7 \
  --curated_output_dir ./curated_train_dataset_model7 \
  --korean_dataset "$DATASET_KO" \
  --code_dataset "$DATASET_CODE" \
  --korean_samples 25000 \
  --code_samples 25000 \
  --max_seq_length 512 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5

echo "================================================="
echo "[2/5] Final quantization for speed"
echo "================================================="
python quantize_for_speed.py \
  --merged_model_dir ./merged_model7 \
  --output_dir ./model \
  --calib_dataset ./curated_train_dataset_model7 \
  --calib_samples 512 \
  --seqlen 512 \
  --iters 1000 \
  --group_size 128 \
  --max_model_len 512 \
  --max_new_tokens 192 \
  --repetition_penalty 1.08 \
  --use-marlin

echo "================================================="
echo "[3/5] Package submit.zip"
echo "================================================="
python package_submit.py \
  --source_dir ./model \
  --zip_name submit \
  --staging_dir ./_submit_staging \
  --max_model_len 512 \
  --max_new_tokens 192 \
  --repetition_penalty 1.08

echo "================================================="
echo "[4/5] Preflight check"
echo "================================================="
if [ -f "submit.zip" ]; then
  python ../preflight_check.py --zip-path submit.zip
  echo "[DONE] Model7 workflow completed."
else
  echo "[ERROR] Failed to create submit.zip"
  exit 1
fi
