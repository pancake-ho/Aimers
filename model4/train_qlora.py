#!/usr/bin/env python
"""
QLoRA fine-tuning for EXAONE-4.0-1.2B quantized base (model4 strategy).

Requirements implemented:
- LoRA capacity: r=32, lora_alpha=64
- Training: num_train_epochs=3, learning_rate=5e-5
- Flexible dataset loading:
  - local directory path -> load_from_disk()
  - otherwise -> load_dataset()
- Save adapter + tokenizer to output directory
- No physical merge (avoids GPTQ merge errors)
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Dict, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


TARGET_MODULES = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA training for EXAONE quantized model")
    p.add_argument("--model_dir", type=str, default="./model_q4_train")
    p.add_argument("--output_dir", type=str, default="./model_final_adapter")
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    p.add_argument("--dataset_config", type=str, default="main")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max_samples", type=int, default=4096)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument(
        "--fallback_base_model_id",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="Fallback base model id when mixed-bit GPTQ checkpoint cannot be loaded for training.",
    )
    p.add_argument(
        "--fallback_base_local_files_only",
        action="store_true",
        help="Use local files only for fallback base model loading.",
    )
    return p.parse_args()


def load_any_dataset(dataset_name: str, dataset_config: str, split: str):
    if os.path.isdir(dataset_name):
        ds = load_from_disk(dataset_name)
        if isinstance(ds, DatasetDict):
            if split not in ds:
                raise ValueError(f"Split '{split}' not found in local dataset: {list(ds.keys())}")
            return ds[split]
        if isinstance(ds, Dataset):
            return ds
        raise TypeError("Unsupported local dataset type.")
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def to_qa(example: Dict[str, str]) -> Tuple[str, str]:
    if "question" in example and "answer" in example:
        return str(example["question"]).strip(), str(example["answer"]).strip()
    if "problem" in example and "solution" in example:
        return str(example["problem"]).strip(), str(example["solution"]).strip()
    if "instruction" in example and "output" in example:
        return str(example["instruction"]).strip(), str(example["output"]).strip()
    if "prompt" in example and "response" in example:
        return str(example["prompt"]).strip(), str(example["response"]).strip()
    if "input" in example and "output" in example:
        return str(example["input"]).strip(), str(example["output"]).strip()
    if "text" in example:
        # already curated/templated text
        return str(example["text"]).strip(), ""
    raise ValueError(f"Unsupported dataset schema keys: {list(example.keys())}")


def format_example(example: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, str]:
    if "text" in example and len(example.keys()) == 1:
        return {"text": str(example["text"])}

    user_text, assistant_text = to_qa(example)
    if assistant_text:
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    else:
        messages = [{"role": "user", "content": user_text}]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        if assistant_text:
            text = f"<|user|>\n{user_text}\n<|assistant|>\n{assistant_text}"
        else:
            text = user_text
    return {"text": text}


def print_trainable_params(model: torch.nn.Module):
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100.0 * trainable / total if total else 0.0
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    print("[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/6] Loading training base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            local_files_only=True,
            device_map="auto",
        )
        print(f"[INFO] Loaded model from {args.model_dir}")
    except RuntimeError as e:
        msg = str(e)
        if "size mismatch for qweight" not in msg:
            raise
        print("[WARN] Mixed-bit GPTQ checkpoint is not train-loadable via transformers/auto_gptq.")
        print(f"[WARN] Falling back to full base model: {args.fallback_base_model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            args.fallback_base_model_id,
            trust_remote_code=True,
            local_files_only=args.fallback_base_local_files_only,
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map="auto",
        )
        print(f"[INFO] Loaded fallback model from {args.fallback_base_model_id}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("[3/6] Loading and formatting dataset...")
    dataset = load_any_dataset(args.dataset_name, args.dataset_config, args.split)
    dataset = dataset.shuffle(seed=42)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting dataset with chat template",
    )
    print(f"[INFO] Train samples: {len(dataset)}")

    print("[4/6] Applying LoRA config (r=32, alpha=64)...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_params(model)

    print("[5/6] Building SFTTrainer...")
    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_trainer_ckpt"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": dataset,
    }
    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sft_params:
        trainer_kwargs["formatting_func"] = lambda ex: ex["text"]
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    print("[6/6] Saving adapter + tokenizer (no merge)...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    meta = {
        "base_model_dir": args.model_dir,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
        "lora_r": 32,
        "lora_alpha": 64,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "note": "Adapter-only output. No physical merge.",
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved adapter+tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
