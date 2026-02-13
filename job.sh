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
cd /data/$USER/repos/aimers/Aimers

# 3. 실행
python test.py

exit 0
