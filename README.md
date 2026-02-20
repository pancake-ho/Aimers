# Aimers KD -> Quant Pipeline

Pipeline (fixed path):

`KD data generation -> LoRA SFT/merge -> GPTQ/AWQ quantization -> ./model + zip -> vLLM smoke`

## Files

- `model/generate_kd_data.py`
- `model/train_kd_lora.py`
- `model/quantize.py`
- `model/pipeline.py`
- `model/utils/*`
- `scripts/seraph_kd_quant.sbatch`

## Quick run

```bash
python model/pipeline.py \
  --student_model ./base_model \
  --base_model ./base_model \
  --quant_method gptq \
  --kd_num_samples 5000 \
  --out_dir ./model
```

## Stage commands

### 1) KD pseudo-label generation

```bash
python model/generate_kd_data.py \
  --student_model ./base_model \
  --dataset_id LGAI-EXAONE/MANTA-1M \
  --dataset_split train \
  --num_samples 5000 \
  --kd_format prompt_completion \
  --output_path ./kd_data/train.jsonl
```

### 2) KD LoRA training + merge

```bash
python model/train_kd_lora.py \
  --base_model ./base_model \
  --kd_data_path ./kd_data/train.jsonl \
  --data_format prompt_completion \
  --output_lora_dir ./distilled_lora \
  --output_merged_dir ./distilled_merged
```

### 3) Quantize + package

```bash
python model/quantize.py \
  --input_model_dir ./distilled_merged \
  --out_dir ./model \
  --quant_method gptq \
  --num_calibration_samples 1024 \
  --max_seq_length 1024 \
  --scheme W4A16 \
  --targets Linear \
  --ignore embed_tokens,lm_head \
  --zip_name submit \
  --run_vllm_smoke
```

## lm-eval (optional)

`--run_lm_eval` requires `--lm_eval_tasks`.

```bash
python model/pipeline.py \
  --run_lm_eval \
  --lm_eval_tasks "arc_challenge,hellaswag" \
  --lm_eval_model_backend vllm
```

You can also run directly:

```bash
python -m lm_eval \
  --model vllm \
  --model_args pretrained=./model \
  --tasks arc_challenge,hellaswag \
  --batch_size auto
```

## Memory metric

`model/quantize.py` writes and prints safetensors size:

- `./model/size_report.json`
- `total_bytes`
- `total_mib`

## Seraph usage

Heavy jobs must run on compute nodes (`srun` or `sbatch`), not on master node.

### sbatch

```bash
PARTITION=batch_eebme_ugrad sbatch -p batch_eebme_ugrad scripts/seraph_kd_quant.sbatch
```

### srun debug example

```bash
srun -p debug_eebme_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 04:00:00 --pty bash
python model/pipeline.py --stop_after kd --kd_num_samples 32 --kd_max_new_tokens 32
```

The sbatch template sets:

- `HF_HOME=/local_datasets/$USER/hf_home`
- `HF_DATASETS_CACHE=/local_datasets/$USER/hf_datasets`
- `TRANSFORMERS_CACHE=/local_datasets/$USER/hf_transformers`
