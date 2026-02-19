#!/usr/bin/env python
"""
Build a standalone submission model from a LoRA adapter:
1) Merge LoRA adapter into EXAONE-4.0-1.2B base model
2) Re-quantize merged model with AutoRound (default: uniform 4-bit)
   - Optional hybrid mode: MLP gate/up/down -> 8-bit
3) Export in auto_gptq format
4) Package submit.zip with root directory exactly: model/
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil

import torch
from auto_round import AutoRound
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Merge adapter + AutoRound export + zip")
    p.add_argument("--base_model_id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--adapter_dir", default="./adapter_model4")
    p.add_argument("--output_dir", default="./model", help="Final standalone model directory")
    p.add_argument("--zip_name", default="submit")

    # Calibration settings (increase gradually if OOM)
    p.add_argument("--dataset_id", default="LGAI-EXAONE/MANTA-1M")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--nsamples", type=int, default=384)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--iters", type=int, default=350)
    p.add_argument("--group_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sym", action="store_true", default=True)
    p.add_argument(
        "--hybrid_mlp_8bit",
        action="store_true",
        help="Enable mixed-bit export (MLP gate/up/down at 8-bit, others at 4-bit).",
    )
    return p.parse_args()


def build_layer_config(model):
    """
    Hybrid-bit quantization:
    - MLP projections 8-bit
    - everything else remains global 4-bit
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
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/6] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    base_model.config.use_cache = False

    print("[3/6] Loading adapter and merging...")
    merged_model = PeftModel.from_pretrained(base_model, args.adapter_dir).merge_and_unload()
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("[4/6] Building shuffled calibration set...")
    ds = load_dataset(args.dataset_id, split=args.dataset_split).shuffle(seed=args.seed)
    ds = ds.select(range(min(args.nsamples, len(ds))))

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
            max_length=args.seqlen,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    calib_list = [ds[i] for i in range(len(ds))]

    layer_config = None
    if args.hybrid_mlp_8bit:
        layer_config = build_layer_config(merged_model)
        print(f"[INFO] Hybrid overrides (MLP->8bit): {len(layer_config)} modules")
    else:
        print("[INFO] Using uniform 4-bit quantization for vLLM compatibility.")

    print("[5/6] AutoRound quantization + auto_gptq export...")
    autoround_kwargs = {
        "bits": 4,
        "group_size": args.group_size,
        "sym": args.sym,
        "dataset": calib_list,
        "seqlen": args.seqlen,
        "nsamples": len(calib_list),
        "iters": args.iters,
        "lr": 1e-2,
        "minmax_lr": 1e-2,
        "enable_quanted_input": True,
        "enable_minmax_tuning": True,
        "batch_size": 1,
        "gradient_accumulate_steps": 8,
        "scale_dtype": torch.float32,
    }
    if layer_config:
        autoround_kwargs["layer_config"] = layer_config
    autoround = AutoRound(merged_model, tokenizer, **autoround_kwargs)
    autoround.quantize()

    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    autoround.save_quantized(
        output_dir=args.output_dir,
        format="auto_gptq",
        inplace=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    print("[6/6] Packaging submit.zip with root model/ ...")
    # Competition expects zip root directory named exactly "model/"
    if os.path.basename(os.path.abspath(args.output_dir)) != "model":
        if os.path.exists("./model"):
            shutil.rmtree("./model")
        shutil.copytree(args.output_dir, "./model")
        zip_base_dir = "model"
    else:
        zip_base_dir = "model"

    if os.path.exists(f"{args.zip_name}.zip"):
        os.remove(f"{args.zip_name}.zip")
    shutil.make_archive(
        base_name=args.zip_name,
        format="zip",
        root_dir=".",
        base_dir=zip_base_dir,
    )
    print(f"[DONE] Created {args.zip_name}.zip")


if __name__ == "__main__":
    main()
