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
  --quant_method "$QUANT" \
  --do_kd \
  --do_lora \
  --search_budget 1 \
  --disable_kd_grid \
  --disable_quant_grid \
  --selection_metric lb_proxy \
  --score_perf_weight 0.5 \
  --score_speed_weight 0.5 \
  --eval_dataset_id LGAI-EXAONE/MANTA-1M \
  --eval_dataset_split train \
  --eval_start 200000 \
  --eval_count 128 \
  --strict_rehearsal \
  --rehearsal_mode full \
  --report_path metrics.csv \
  --use_external_mix \
  --mix_dataset_ids MyeongHo0621/korean-quality-cleaned,m-a-p/Code-Feedback \
  --mix_dataset_splits train,train \
  --mix_dataset_configs "$MIX_CONFIGS" \
  --mix_weights 0.60,0.25,0.15

exit 0
