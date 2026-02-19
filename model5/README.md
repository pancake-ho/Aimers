# model5

Improved pipeline based on model4:
- stronger mixed curated dataset
- higher-capacity LoRA targets (`q/k/v/o + MLP`)
- vLLM-safe final export (uniform 4-bit by default)

## Files
- `build_curated_dataset.py`: builds `./curated_train_dataset` from 3 datasets
- `train_qlora.py`: trains LoRA adapter (`./adapter_model5`)
- `build_submission_from_adapter.py`: merges adapter, quantizes, writes `submit.zip`
- `job.sh`: full Slurm workflow

## Run
From `Aimers/model5`:

```bash
sbatch job.sh
```

