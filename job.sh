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

# Seraph 권장 캐시 경로 설정
export HF_HOME=/data/$USER/.cache/huggingface
export TORCH_HOME=/data/$USER/.cache/torch
export PIP_CACHE_DIR=/data/$USER/.cache/pip

# 콘다 환경 활성화
source /data/$USER/anaconda3/etc/profile.d/conda.sh
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
        print(f'[SUCCESS] {repo_id} download complete.\n')
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
# main.py가 완료되었는지 확인할 기준 디렉토리나 파일 설정
QUANT_CHECK_DIR="./quantized_model"

if [ -d "$QUANT_CHECK_DIR" ]; then
    echo "[SKIP] $QUANT_CHECK_DIR 이 이미 존재합니다. main.py를 건너뛰고 파인튜닝으로 넘어갑니다."
else
    echo "[RUN] main.py 실행 중 (모델 로딩 및 양자화)..."
    python main.py \
        --model_id $BASE_MODEL \
        --output_dir $QUANT_CHECK_DIR
fi

echo "================================================="
echo "[2/5] Fine-tuning Stage (train_qlora.py)"
echo "================================================="
# train_qlora.py를 실행하여 파인튜닝을 진행합니다.
# 결과물은 ./adapter_model4에 저장되도록 설정합니다.
python train_qlora.py \
    --base_model_path $QUANT_CHECK_DIR \
    --dataset_name $DATASET_2 \
    --output_dir ./adapter_model4 \
    --epochs 3 \
    --batch_size 4

echo "================================================="
echo "[3/5] Pre-check (Adapter Check)"
echo "================================================="
if [ ! -d "./adapter_model4" ]; then
  echo "[FAIL] 파인튜닝 결과물(./adapter_model4)이 생성되지 않았습니다."
  exit 1
fi

echo "================================================="
echo "[4/5] Build standalone model + submit.zip"
echo "================================================="
python model4/build_submission_from_adapter.py \
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
    python preflight_check.py --zip-path submit.zip
    echo "[DONE] 전 과정(다운로드 -> 양자화 -> 파인튜닝 -> 빌드) 완료!"
else
    echo "[ERROR] 최종 submit.zip 생성 실패."
    exit 1
fi