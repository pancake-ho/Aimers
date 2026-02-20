#!/usr/bin/bash

#SBATCH -J aimers_pancake
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_eebme_ugrad
#SBATCH -w moana-y6
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

QUANT=${QUANT:-gptq}
MIX_CONFIGS=${MIX_CONFIGS:-,}

source /data/$USER/anaconda3/etc/profile.d/conda.sh
conda activate aimers

cd /data/$USER/repos/aimers/model

python main.py \
  --base_model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --out_dir ./artifacts \
  --quant_method gptq \
  --do_kd --do_lora \
  --quant_small_grid \
  --quant_two_stage_eval \
  --quant_proxy_eval_count 32 \
  --quant_two_stage_top_k 2 \
  --skip_rehearsal \
  --skip_smoke


exit 0
