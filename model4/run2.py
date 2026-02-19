#!/usr/bin/env python
"""
AutoRound quantization for EXAONE-4.0-1.2B (model4 strategy).

Requirements implemented:
- Hybrid precision:
  - MLP: gate_proj/up_proj/down_proj -> 8-bit
  - All other layers (including attention q/k/v/o_proj) -> 4-bit
- Calibration quality:
  - nsamples=1024
  - iters=1000
  - seqlen=2048
- Calibration dataset shuffle with seed=42
- Export format: auto_gptq
"""

import os
import shutil

import torch
from auto_round import AutoRound
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
OUT_DIR = "./model_q4_train"

BITS = 4
GROUP_SIZE = 128
SYMMETRIC = True
SEQLEN = 1024
NSAMPLES = 384
ITERS = 350
SEED = 42


def build_layer_config(model):
    """
    Hybrid precision policy:
    - Set only MLP projections to 8-bit.
    - Keep everything else at global 4-bit (including q/k/v/o_proj).
    """
    layer_config = {}
    for name, _ in model.named_modules():
        if (
            name.endswith("gate_proj")
            or name.endswith("up_proj")
            or name.endswith("down_proj")
        ):
            layer_config[name] = {"bits": 8}
    return layer_config


def main():
    print("[AutoRound] Loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    print("[AutoRound] Loading + shuffling calibration dataset...")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT).shuffle(seed=SEED)
    ds = ds.select(range(min(NSAMPLES, len(ds))))

    def preprocess(example):
        prompt = tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False,
        )
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=SEQLEN,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset_list = [ds[i] for i in range(len(ds))]

    layer_config = build_layer_config(model)
    print(f"[AutoRound] MLP 8-bit overrides: {len(layer_config)} modules")
    print(
        f"[AutoRound] Start quantization (bits={BITS}, nsamples={len(dataset_list)}, "
        f"seqlen={SEQLEN}, iters={ITERS})"
    )

    autoround = AutoRound(
        model,
        tokenizer,
        bits=BITS,
        group_size=GROUP_SIZE,
        sym=SYMMETRIC,
        dataset=dataset_list,
        seqlen=SEQLEN,
        nsamples=len(dataset_list),
        iters=ITERS,
        lr=1e-2,
        minmax_lr=1e-2,
        enable_quanted_input=True,
        enable_minmax_tuning=True,
        batch_size=1,
        gradient_accumulate_steps=8,
        scale_dtype=torch.float32,
        layer_config=layer_config,
    )

    autoround.quantize()
    print("[AutoRound] Quantization done.")

    os.makedirs(OUT_DIR, exist_ok=True)
    autoround.save_quantized(
        output_dir=OUT_DIR,
        format="auto_gptq",
        inplace=True,
    )
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[AutoRound] Saved quantized model to: {OUT_DIR}")

    # Optional convenience zip for quick checks
    zip_name = "submit"
    shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=OUT_DIR)
    print(f"[AutoRound] Created {zip_name}.zip")


if __name__ == "__main__":
    main()
