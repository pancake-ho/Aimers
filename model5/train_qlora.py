#!/usr/bin/env python
"""
Model5 QLoRA training:
- trains adapter from full base model for quality
- supports local curated dataset and mixed schema datasets
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _sanitize_tokenizer_config(tokenizer: AutoTokenizer, output_dir: str | None = None) -> None:
    # Keep tokenizer metadata vLLM-safe.
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs.pop("fix_mistral_regex", None)
    if output_dir:
        cfg = os.path.join(output_dir, "tokenizer_config.json")
        if os.path.isfile(cfg):
            with open(cfg, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "fix_mistral_regex" in data:
                data.pop("fix_mistral_regex", None)
                with open(cfg, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model5 QLoRA training")
    p.add_argument("--base_model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--adapter_output_dir", type=str, default="./adapter_model5")

    p.add_argument("--dataset_path", type=str, default="./curated_train_dataset")
    p.add_argument("--dataset_name", type=str, default="MyeongHo0621/korean-quality-cleaned")
    p.add_argument("--dataset_config", type=str, default="default")
    p.add_argument("--split", type=str, default="train")

    p.add_argument("--max_samples", type=int, default=10000)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
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


def _to_qa(example: Dict[str, Any]) -> Tuple[str, str]:
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
        return str(example["text"]).strip(), ""
    raise ValueError(f"Unsupported dataset schema keys: {list(example.keys())}")


def _format_example(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, str]:
    if "text" in example and str(example["text"]).strip():
        return {"text": str(example["text"]).strip()}

    if "messages" in example:
        msgs = _normalize_messages(example["messages"])
        if msgs:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in msgs])
            return {"text": text.strip()}

    if "conversations" in example:
        msgs = _normalize_messages(example["conversations"])
        if msgs:
            try:
                text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in msgs])
            return {"text": text.strip()}

    user_text, assistant_text = _to_qa(example)
    if assistant_text:
        msgs = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]
    else:
        msgs = [{"role": "user", "content": user_text}]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    except Exception:
        if assistant_text:
            text = f"<|user|>\n{user_text}\n<|assistant|>\n{assistant_text}"
        else:
            text = user_text
    return {"text": text.strip()}


def _load_dataset_any(args: argparse.Namespace):
    if args.dataset_path and os.path.isdir(args.dataset_path):
        ds = load_from_disk(args.dataset_path)
        if isinstance(ds, DatasetDict):
            if args.split in ds:
                return ds[args.split]
            first = next(iter(ds.keys()))
            return ds[first]
        if isinstance(ds, Dataset):
            return ds
        raise TypeError("Unsupported local dataset object.")

    if args.dataset_config:
        try:
            return load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        except ValueError as e:
            msg = str(e)
            if "BuilderConfig" not in msg or "not found" not in msg:
                raise
    return load_dataset(args.dataset_name, split=args.split)


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


def main() -> None:
    args = parse_args()
    os.makedirs(args.adapter_output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    print("[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    _sanitize_tokenizer_config(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading full base model for QLoRA...")
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

    print("[3/5] Loading and preparing dataset...")
    dataset = _load_dataset_any(args)
    dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    if "text" in dataset.column_names:
        dataset = dataset.map(lambda ex: {"text": str(ex["text"]).strip()}, desc="Normalizing text column")
    else:
        dataset = dataset.map(
            lambda ex: _format_example(ex, tokenizer),
            remove_columns=dataset.column_names,
            desc="Formatting dataset with chat template",
        )
    dataset = dataset.filter(lambda x: len(x["text"]) > 0, desc="Filtering empty rows")
    print(f"[INFO] Train samples: {len(dataset)}")

    print("[4/5] Applying LoRA and training...")
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
    trainer_kwargs = {"model": model, "args": train_args, "train_dataset": dataset}
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

    print("[5/5] Saving adapter + tokenizer...")
    trainer.model.save_pretrained(args.adapter_output_dir)
    tokenizer.save_pretrained(args.adapter_output_dir)
    _sanitize_tokenizer_config(tokenizer, args.adapter_output_dir)

    meta = {
        "base_model_id": args.base_model_id,
        "dataset_path": args.dataset_path,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "max_samples": args.max_samples,
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": TARGET_MODULES,
    }
    with open(os.path.join(args.adapter_output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved adapter to: {args.adapter_output_dir}")


if __name__ == "__main__":
    main()
