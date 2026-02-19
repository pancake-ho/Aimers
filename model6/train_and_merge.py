#!/usr/bin/env python
"""
Model6 stage-1 pipeline:
1) Build a balanced 30k SFT set (15k Korean + 15k Code)
2) Train LoRA on full-precision EXAONE-1.2B
3) Merge LoRA into base weights (zero adapter overhead)
4) Save merged model as a single safetensors file
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from typing import Any, Dict, List

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


TARGET_MODULES = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model6 train + merge pipeline")
    p.add_argument("--base_model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--adapter_output_dir", type=str, default="./adapter_model6")
    p.add_argument("--merged_output_dir", type=str, default="./merged_model6")

    p.add_argument("--korean_dataset", type=str, default="MyeongHo0621/korean-quality-cleaned")
    p.add_argument("--korean_config", type=str, default="default")
    p.add_argument("--korean_split", type=str, default="train")
    p.add_argument("--korean_samples", type=int, default=15000)

    p.add_argument("--code_dataset", type=str, default="m-a-p/Code-Feedback")
    p.add_argument("--code_config", type=str, default=None)
    p.add_argument("--code_split", type=str, default="train")
    p.add_argument("--code_samples", type=int, default=15000)

    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=4e-5)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    return p.parse_args()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]).strip())
                elif "content" in item:
                    parts.append(str(item["content"]).strip())
        return "\n".join([p for p in parts if p]).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return str(content["content"]).strip()
    return str(content).strip()


def _normalize_messages(raw_messages: Any) -> List[Dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []
    role_map = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    out: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", msg.get("from", "user"))).strip().lower()
        role = role_map.get(role, role)
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = _content_to_text(msg.get("content", msg.get("value", msg.get("text", ""))))
        if content:
            out.append({"role": role, "content": content})
    return out


def _to_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    if "messages" in example:
        msgs = _normalize_messages(example["messages"])
        if msgs:
            return msgs
    if "conversations" in example:
        msgs = _normalize_messages(example["conversations"])
        if msgs:
            return msgs
    if "question" in example and "answer" in example:
        return [
            {"role": "user", "content": str(example["question"]).strip()},
            {"role": "assistant", "content": str(example["answer"]).strip()},
        ]
    if "problem" in example and "solution" in example:
        return [
            {"role": "user", "content": str(example["problem"]).strip()},
            {"role": "assistant", "content": str(example["solution"]).strip()},
        ]
    if "instruction" in example and "output" in example:
        return [
            {"role": "user", "content": str(example["instruction"]).strip()},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]
    if "prompt" in example and "response" in example:
        return [
            {"role": "user", "content": str(example["prompt"]).strip()},
            {"role": "assistant", "content": str(example["response"]).strip()},
        ]
    if "input" in example and "output" in example:
        return [
            {"role": "user", "content": str(example["input"]).strip()},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]
    if "text" in example and str(example["text"]).strip():
        return [{"role": "user", "content": str(example["text"]).strip()}]
    return []


def _messages_to_text(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        ).strip()
    except Exception:
        return "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in messages]).strip()


def _load_split(dataset_name_or_path: str, dataset_config: str | None, split: str) -> Dataset:
    if os.path.isdir(dataset_name_or_path):
        ds = load_from_disk(dataset_name_or_path)
        if isinstance(ds, DatasetDict):
            if split in ds:
                return ds[split]
            first_key = next(iter(ds.keys()))
            return ds[first_key]
        if isinstance(ds, Dataset):
            return ds
        raise TypeError("Unsupported local dataset object.")

    if dataset_config:
        try:
            return load_dataset(dataset_name_or_path, dataset_config, split=split)
        except ValueError as e:
            msg = str(e)
            if "BuilderConfig" not in msg or "not found" not in msg:
                raise
    ds = load_dataset(dataset_name_or_path)
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        first_key = next(iter(ds.keys()))
        return ds[first_key]
    return ds


def _format_source(ds: Dataset, source: str, tokenizer: AutoTokenizer) -> Dataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, str]:
        messages = _to_messages(ex)
        if not messages:
            return {"text": "", "source": source}
        return {"text": _messages_to_text(messages, tokenizer), "source": source}

    ds = ds.map(_map, remove_columns=ds.column_names, desc=f"Formatting {source}")
    ds = ds.filter(lambda x: len(x["text"]) > 0, desc=f"Filtering empty rows ({source})")
    return ds


def _sample_exact(ds: Dataset, n: int, seed: int, source_name: str) -> Dataset:
    if n <= 0:
        return ds.select([])
    if len(ds) < n:
        raise ValueError(
            f"Not enough rows in {source_name}: requested {n}, available {len(ds)}. "
            "Adjust --*_samples or dataset filters."
        )
    return ds.shuffle(seed=seed).select(range(n))


def _print_trainable_params(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100.0 * trainable / total if total else 0.0
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def _update_max_model_len(model_dir: str, max_model_len: int) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["max_model_len"] = int(max_model_len)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    os.makedirs(args.adapter_output_dir, exist_ok=True)
    if os.path.isdir(args.merged_output_dir):
        shutil.rmtree(args.merged_output_dir)
    os.makedirs(args.merged_output_dir, exist_ok=True)

    print("[1/7] Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("[2/7] Loading and formatting Korean dataset...")
    ds_k = _load_split(args.korean_dataset, args.korean_config, args.korean_split)
    ds_k = _format_source(ds_k, "korean_quality", tokenizer)

    print("[3/7] Loading and formatting Code dataset...")
    ds_c = _load_split(args.code_dataset, args.code_config, args.code_split)
    ds_c = _format_source(ds_c, "code_feedback", tokenizer)

    print("[4/7] Sampling balanced 30k dataset (15k + 15k)...")
    part_k = _sample_exact(ds_k, args.korean_samples, args.seed, "korean dataset")
    part_c = _sample_exact(ds_c, args.code_samples, args.seed + 1, "code dataset")
    train_ds = concatenate_datasets([part_k, part_c]).shuffle(seed=args.seed)
    print(
        f"[INFO] Final train size: {len(train_ds)} "
        f"(korean={len(part_k)}, code={len(part_c)})"
    )

    print("[5/7] Applying LoRA and training...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    _print_trainable_params(model)

    train_args = TrainingArguments(
        output_dir=os.path.join(args.adapter_output_dir, "_trainer_ckpt"),
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
        seed=args.seed,
    )

    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = {"model": model, "args": train_args, "train_dataset": train_ds}
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

    print("[6/7] Saving adapter...")
    trainer.model.save_pretrained(args.adapter_output_dir)
    tokenizer.save_pretrained(args.adapter_output_dir)

    print("[7/7] Merging adapter into base and exporting single safetensors...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.config.use_cache = True
    merged_model.save_pretrained(
        args.merged_output_dir,
        safe_serialization=True,
        max_shard_size="20GB",
    )
    tokenizer.save_pretrained(args.merged_output_dir)
    _update_max_model_len(args.merged_output_dir, args.max_seq_length)

    weight_files = [f for f in os.listdir(args.merged_output_dir) if f.endswith(".safetensors")]
    if len(weight_files) != 1:
        raise RuntimeError(
            f"Merged model must be one safetensors file, found: {weight_files}"
        )

    meta = {
        "base_model_id": args.base_model_id,
        "korean_dataset": args.korean_dataset,
        "code_dataset": args.code_dataset,
        "korean_samples": args.korean_samples,
        "code_samples": args.code_samples,
        "total_samples": len(train_ds),
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": TARGET_MODULES,
        "merged_weight_file": weight_files[0],
    }
    with open(os.path.join(args.merged_output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Adapter: {args.adapter_output_dir}")
    print(f"[DONE] Merged model: {args.merged_output_dir}")


if __name__ == "__main__":
    main()

