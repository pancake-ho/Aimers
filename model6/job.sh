#!/usr/bin/bash

#SBATCH -J aimers_model6
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
echo "[0/4] Hugging Face assets download"
echo "================================================="
BASE_MODEL="LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_KO="MyeongHo0621/korean-quality-cleaned"
DATASET_CODE="m-a-p/Code-Feedback"
DATASET_CALIB="LGAI-EXAONE/MANTA-1M"

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
download_asset('$DATASET_CALIB', repo_type='dataset')
"

echo "================================================="
echo "[1/4] Train and merge (model6)"
echo "================================================="
python train_and_merge.py \
  --base_model_id "$BASE_MODEL" \
  --adapter_output_dir ./adapter_model6 \
  --merged_output_dir ./merged_model6 \
  --korean_dataset "$DATASET_KO" \
  --code_dataset "$DATASET_CODE" \
  --korean_samples 15000 \
  --code_samples 15000 \
  --max_seq_length 512

echo "================================================="
echo "[2/4] Final quantization and packaging"
echo "================================================="
python quantize_final.py \
  --merged_model_dir ./merged_model6 \
  --output_dir ./model \
  --zip_name submit \
  --calib_dataset "$DATASET_CALIB" \
  --calib_samples 384 \
  --seqlen 512 \
  --iters 1000 \
  --max_model_len 512 \
  --use-marlin

echo "================================================="
echo "[3/4] Preflight check"
echo "================================================="
if [ -f "submit.zip" ]; then
  python ../preflight_check.py --zip-path submit.zip
  echo "[DONE] Model6 workflow completed."
else
  echo "[ERROR] Failed to create submit.zip"
  exit 1
fi

