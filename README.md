# Aimers Phase3 Pipeline

Pipeline:

`KD (optional) -> LoRA (optional, merged) -> Quant (GPTQ/AWQ) -> submit.zip -> rehearsal`

## Run

```bash
cd model
python main.py \
  --base_model LGAI-EXAONE/EXAONE-4.0-1.2B \
  --out_dir ./artifacts \
  --quant_method gptq \
  --do_kd \
  --do_lora \
  --selection_metric lb_proxy \
  --strict_rehearsal
```

## Key options

- `--quant_method {gptq,awq}`
- `--calib_shortfall_policy {downscale,replacement}`
- `--gptq_scheme`, `--gptq_targets`, `--gptq_ignore_patterns`
- `--awq_group_size`, `--awq_symmetric`, `--awq_duo_scaling`, `--awq_offload_device`
- `--quant_small_grid`, `--gptq_grid_block_sizes`, `--gptq_grid_dampening_values`
- `--quant_two_stage_eval`, `--quant_proxy_eval_count`, `--quant_two_stage_top_k`
- `--mix_streaming`, `--teacher_model_id`, `--rehearsal`

## Artifacts

- KD trials: `artifacts/kd/trial_XX/`
- LoRA trial: `artifacts/lora/trial_01/`
- Quant trials: `artifacts/quant/{method}/trial_XX/`
- Final submit zip source: selected quant `final_model_dir`
- Submit zip: `artifacts/submit.zip`

Each trial directory writes:

- `config_used.json`
- `metrics.json`
- `eval.json`

Top-level reports:

- `metrics.csv`
- `metrics.jsonl`
- `metrics.json`
- `final_summary.json`
