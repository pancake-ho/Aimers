#!/usr/bin/bash

#SBATCH -J aimers_final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

# -------------------------------------------------------

# 1. 환경 활성화
source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate aimers

# 2. 프로젝트 폴더로 이동
cd /data/$USER/repos/aimers/Aimers/model

# 3. 실행
python3 main.py \
  --quant_method gptq \
  --do_kd \
  --search_budget 6 \
  --eval_dataset_id LGAI-EXAONE/MANTA-1M \
  --eval_dataset_split train \
  --eval_start 200000 \
  --eval_count 128 \
  --out_dir ./artifacts

exit 0
