#!/usr/bin/env python
"""
AutoRound quantization pipeline for EXAONE-4.0-1.2B.

Score-oriented improvements:
- Preset profiles for fast/balanced/quality runs.
- Shuffled calibration sampling (avoid first-chunk bias).
- Correct AutoRound argument names for current API (`seqlen`, `nsamples`).
- Configurable output directory and optional zip packaging.
"""

import argparse
import os
import shutil

import torch
from auto_round import AutoRound
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"


def parse_args():
    p = argparse.ArgumentParser(description="AutoRound quantization for EXAONE")
    p.add_argument("--preset", choices=["fast", "balanced", "quality"], default="balanced")
    p.add_argument("--out_dir", default="./model_q4_train")
    p.add_argument("--zip_name", default="submit")
    p.add_argument("--make_zip", action="store_true", help="Create <zip_name>.zip from out_dir")

    p.add_argument("--num_calib", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--sym", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def apply_preset(args):
    if args.preset == "fast":
        num_calib = 256
        seq_len = 1024
        iters = 200
    elif args.preset == "quality":
        num_calib = 1024
        seq_len = 2048
        iters = 600
    else:
        num_calib = 512
        seq_len = 2048
        iters = 400

    args.num_calib = args.num_calib or num_calib
    args.seq_len = args.seq_len or seq_len
    args.iters = args.iters or iters
    return args


def main():
    args = apply_preset(parse_args())

    print("[AutoRound] Loading model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )

    print("[AutoRound] Loading and shuffling calibration data...")
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT).shuffle(seed=args.seed)
    ds = ds.select(range(min(args.num_calib, len(ds))))

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
            max_length=args.seq_len,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataset_list = [ds[i] for i in range(len(ds))]

    print(
        "[AutoRound] Start quantization: "
        f"preset={args.preset}, bits={args.bits}, calib={len(dataset_list)}, "
        f"seq_len={args.seq_len}, iters={args.iters}"
    )

    autoround = AutoRound(
        model,
        tokenizer,
        bits=args.bits,
        group_size=args.group_size,
        sym=args.sym,
        dataset=dataset_list,
        seqlen=args.seq_len,
        nsamples=len(dataset_list),
        iters=args.iters,
        lr=args.lr,
        minmax_lr=args.lr,
        enable_quanted_input=True,
        enable_minmax_tuning=True,
        batch_size=1,
        gradient_accumulate_steps=8,
        scale_dtype=torch.float32,
    )

    autoround.quantize()
    print("[AutoRound] Quantization complete.")

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[AutoRound] Saving GPTQ-compatible model to {args.out_dir} ...")
    autoround.save_quantized(output_dir=args.out_dir, format="auto_gptq", inplace=True)
    tokenizer.save_pretrained(args.out_dir)

    if args.make_zip:
        print(f"[AutoRound] Creating {args.zip_name}.zip ...")
        shutil.make_archive(
            base_name=args.zip_name,
            format="zip",
            root_dir=".",
            base_dir=args.out_dir,
        )
        print(f"[AutoRound] Created {args.zip_name}.zip")

    print("[AutoRound] Done.")


if __name__ == "__main__":
    main()
