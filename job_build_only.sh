#!/usr/bin/bash

#SBATCH -J aimers_full_pipeline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out  # 로그 파일을 logs 폴더에 저장

set -euo pipefail

# [중요] Seraph 권장사항: /home 대신 /data 영역에 캐시 저장
export HF_HOME=/data/$USER/.cache/huggingface
export TORCH_HOME=/data/$USER/.cache/torch
export PIP_CACHE_DIR=/data/$USER/.cache/pip

# 허깅페이스 토큰이 필요한 경우(Gated Model 등) 아래 주석을 해제하고 토큰을 입력하세요
# export HF_TOKEN="your_huggingface_token_here"

# 콘다 환경 활성화
source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate aimers

# 작업 디렉토리 이동
cd /data/$USER/repos/aimers/Aimers/model4
mkdir -p logs

echo "================================================="
echo "[0/3] Hugging Face Model Download"
echo "================================================="
# 다운로드할 베이스 모델명을 입력하세요 (예: "meta-llama/Llama-2-7b-hf")
BASE_MODEL="MLP-KTLim/llama-3-Korean-Bllama-8B" 

echo "Starting download for: $BASE_MODEL"
python -c "
from huggingface_hub import snapshot_download
import os
try:
    snapshot_download(
        repo_id='$BASE_MODEL',
        local_dir_use_symlinks=False,
        cache_dir=os.environ.get('HF_HOME')
    )
    print('[SUCCESS] Download complete.')
except Exception as e:
    print(f'[ERROR] Download failed: {e}')
    exit(1)
"

echo "================================================="
echo "[1/3] Pre-check (Adapter Check)"
echo "================================================="
if [ ! -d "./adapter_model4" ]; then
  echo "[FAIL] Missing adapter directory: ./adapter_model4"
  echo "학습된 어댑터 모델이 필요합니다. 학습을 먼저 진행하거나 경로를 확인하세요."
  exit 1
fi

echo "================================================="
echo "[2/3] Build standalone model + submit.zip"
echo "================================================="
# 다운로드한 베이스 모델과 어댑터를 병합하여 제출 파일을 생성합니다.
python model4/build_submission_from_adapter.py \
  --adapter_dir ./adapter_model4 \
  --output_dir ./model \
  --zip_name submit \
  --nsamples 384 \
  --seqlen 1024 \
  --iters 350

echo "================================================="
echo "[3/3] Preflight check"
echo "================================================="
if [ -f "submit.zip" ]; then
    python preflight_check.py --zip-path submit.zip
else
    echo "[ERROR] submit.zip file not found!"
    exit 1
fi

echo "[DONE] All processes (Download -> Build -> Check) complete."