#!/usr/bin/env python
"""
Model7 stage-1:
1) Build a 50k mixed training set (25k Korean + 25k Code) with short-length priority.
2) Train LoRA (r=32, alpha=64) on full-precision base with max_seq_length=512.
3) Merge adapters into base via merge_and_unload() and export a single safetensors file.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


TARGET_MODULES = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model7 train + merge")
    p.add_argument("--base_model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--adapter_output_dir", type=str, default="./adapter_model7")
    p.add_argument("--merged_output_dir", type=str, default="./merged_model7")
    p.add_argument("--curated_output_dir", type=str, default="./curated_train_dataset_model7")

    p.add_argument("--korean_dataset", type=str, default="MyeongHo0621/korean-quality-cleaned")
    p.add_argument("--korean_config", type=str, default="default")
    p.add_argument("--korean_split", type=str, default="train")
    p.add_argument("--korean_samples", type=int, default=25000)

    p.add_argument("--code_dataset", type=str, default="m-a-p/Code-Feedback")
    p.add_argument("--code_config", type=str, default=None)
    p.add_argument("--code_split", type=str, default="train")
    p.add_argument("--code_samples", type=int, default=25000)

    p.add_argument(
        "--candidate_pool_per_source",
        type=int,
        default=120000,
        help="Cap per-source candidate pool before token-length scoring (0 = full source).",
    )
    p.add_argument("--min_tokens", type=int, default=24)
    p.add_argument("--max_preferred_tokens", type=int, default=400)
    p.add_argument("--max_backfill_tokens", type=int, default=512)
    p.add_argument(
        "--strict_preferred_only",
        action="store_true",
        help="If set, only <=max_preferred_tokens samples are allowed (no backfill).",
    )

    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--num_train_epochs", type=int, default=2)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
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
            return ds[next(iter(ds.keys()))]
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
        return ds[next(iter(ds.keys()))]
    return ds


def _pre_sample_candidates(ds: Dataset, n: int, seed: int) -> Dataset:
    if n <= 0 or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def _format_and_score(ds: Dataset, source_name: str, tokenizer: AutoTokenizer) -> Dataset:
    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        msgs = _to_messages(ex)
        if not msgs:
            return {"text": "", "source": source_name, "tok_len": 0}
        text = _messages_to_text(msgs, tokenizer)
        tok_len = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
        return {"text": text, "source": source_name, "tok_len": tok_len}

    return ds.map(_map, remove_columns=ds.column_names, desc=f"Formatting+Scoring {source_name}")


def _sample_prioritized(
    preferred_ds: Dataset,
    backfill_ds: Dataset,
    target: int,
    strict_preferred_only: bool,
    seed: int,
    source_name: str,
) -> Tuple[Dataset, Dict[str, int]]:
    preferred_count = len(preferred_ds)
    backfill_count = len(backfill_ds)
    take_pref = min(preferred_count, target)
    selected_parts: List[Dataset] = []

    if take_pref > 0:
        selected_parts.append(preferred_ds.shuffle(seed=seed).select(range(take_pref)))

    need = target - take_pref
    if need > 0:
        if strict_preferred_only:
            raise ValueError(
                f"{source_name}: strict preferred mode requested {target}, "
                f"but only {preferred_count} samples <= preferred token limit."
            )
        if backfill_count < need:
            raise ValueError(
                f"{source_name}: requested {target}, available preferred={preferred_count}, "
                f"backfill={backfill_count}. Not enough samples."
            )
        selected_parts.append(backfill_ds.shuffle(seed=seed + 17).select(range(need)))

    selected = concatenate_datasets(selected_parts).shuffle(seed=seed) if selected_parts else preferred_ds.select([])
    stats = {
        "preferred_available": preferred_count,
        "backfill_available": backfill_count,
        "selected_preferred": take_pref,
        "selected_backfill": max(0, target - take_pref),
        "selected_total": len(selected),
    }
    return selected, stats


def _rebalance_targets(
    requested_korean: int,
    requested_code: int,
    available_korean: int,
    available_code: int,
) -> Tuple[int, int, Dict[str, int]]:
    target_korean = min(requested_korean, available_korean)
    target_code = min(requested_code, available_code)

    requested_total = requested_korean + requested_code
    selected_total = target_korean + target_code
    remaining = requested_total - selected_total

    if remaining > 0:
        korean_headroom = max(0, available_korean - target_korean)
        add_korean = min(korean_headroom, remaining)
        target_korean += add_korean
        remaining -= add_korean

    if remaining > 0:
        code_headroom = max(0, available_code - target_code)
        add_code = min(code_headroom, remaining)
        target_code += add_code
        remaining -= add_code

    info = {
        "requested_korean": requested_korean,
        "requested_code": requested_code,
        "requested_total": requested_total,
        "available_korean": available_korean,
        "available_code": available_code,
        "adjusted_korean": target_korean,
        "adjusted_code": target_code,
        "adjusted_total": target_korean + target_code,
        "unfilled_total": remaining,
    }
    return target_korean, target_code, info


def _sanitize_tokenizer_config(tokenizer: AutoTokenizer, output_dir: str) -> None:
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs.pop("fix_mistral_regex", None)
    cfg = os.path.join(output_dir, "tokenizer_config.json")
    if os.path.isfile(cfg):
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "fix_mistral_regex" in data:
            data.pop("fix_mistral_regex", None)
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)


def _update_model_config(model_dir: str, max_model_len: int) -> None:
    cfg = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg):
        return
    with open(cfg, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["use_cache"] = True
    data["max_model_len"] = int(max_model_len)
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    if args.max_backfill_tokens < args.max_preferred_tokens:
        raise ValueError("--max_backfill_tokens must be >= --max_preferred_tokens")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")
    bf16 = torch.cuda.is_bf16_supported()

    os.makedirs(args.adapter_output_dir, exist_ok=True)
    if os.path.isdir(args.merged_output_dir):
        shutil.rmtree(args.merged_output_dir)
    os.makedirs(args.merged_output_dir, exist_ok=True)

    print("[1/7] Loading tokenizer and full-precision base model...")
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

    print("[2/7] Loading source datasets...")
    ds_k = _load_split(args.korean_dataset, args.korean_config, args.korean_split)
    ds_c = _load_split(args.code_dataset, args.code_config, args.code_split)

    ds_k = _pre_sample_candidates(ds_k, args.candidate_pool_per_source, args.seed)
    ds_c = _pre_sample_candidates(ds_c, args.candidate_pool_per_source, args.seed + 1)

    print("[3/7] Formatting, token scoring, and length filtering...")
    ds_k = _format_and_score(ds_k, "korean_quality", tokenizer)
    ds_c = _format_and_score(ds_c, "code_feedback", tokenizer)

    ds_k = ds_k.filter(
        lambda x: len(x["text"]) > 0 and args.min_tokens <= x["tok_len"] <= args.max_backfill_tokens,
        desc="Filtering korean by token length",
    )
    ds_c = ds_c.filter(
        lambda x: len(x["text"]) > 0 and args.min_tokens <= x["tok_len"] <= args.max_backfill_tokens,
        desc="Filtering code by token length",
    )

    pref_k = ds_k.filter(lambda x: x["tok_len"] <= args.max_preferred_tokens, desc="Korean preferred <=400")
    back_k = ds_k.filter(lambda x: x["tok_len"] > args.max_preferred_tokens, desc="Korean backfill")
    pref_c = ds_c.filter(lambda x: x["tok_len"] <= args.max_preferred_tokens, desc="Code preferred <=400")
    back_c = ds_c.filter(lambda x: x["tok_len"] > args.max_preferred_tokens, desc="Code backfill")

    print("[4/7] Sampling with short-text priority...")
    if args.strict_preferred_only:
        avail_k = len(pref_k)
        avail_c = len(pref_c)
    else:
        avail_k = len(pref_k) + len(back_k)
        avail_c = len(pref_c) + len(back_c)

    target_k, target_c, sample_plan = _rebalance_targets(
        requested_korean=args.korean_samples,
        requested_code=args.code_samples,
        available_korean=avail_k,
        available_code=avail_c,
    )
    if sample_plan["adjusted_total"] < sample_plan["requested_total"]:
        print(
            "[WARN] Requested total cannot be fully met. "
            f"requested={sample_plan['requested_total']}, adjusted={sample_plan['adjusted_total']}"
        )
    if target_k != args.korean_samples or target_c != args.code_samples:
        print(
            "[WARN] Rebalanced sample targets due to source limits: "
            f"korean {args.korean_samples}->{target_k}, code {args.code_samples}->{target_c}"
        )

    part_k, stats_k = _sample_prioritized(
        pref_k,
        back_k,
        target=target_k,
        strict_preferred_only=args.strict_preferred_only,
        seed=args.seed,
        source_name="korean",
    )
    part_c, stats_c = _sample_prioritized(
        pref_c,
        back_c,
        target=target_c,
        strict_preferred_only=args.strict_preferred_only,
        seed=args.seed + 1,
        source_name="code",
    )

    train_ds = concatenate_datasets([part_k, part_c]).shuffle(seed=args.seed)
    print(
        f"[INFO] Final train size: {len(train_ds)} "
        f"(korean={len(part_k)}, code={len(part_c)})"
    )

    if args.curated_output_dir:
        if os.path.isdir(args.curated_output_dir):
            shutil.rmtree(args.curated_output_dir)
        train_ds.save_to_disk(args.curated_output_dir)
        print(f"[INFO] Saved curated dataset: {args.curated_output_dir}")

    print("[5/7] LoRA training...")
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
    trainer_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": train_ds,
        "dataset_text_field": "text",
    }
    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" not in sft_params:
        trainer_kwargs.pop("dataset_text_field", None)
    if "formatting_func" in sft_params and "dataset_text_field" not in sft_params:
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
    _sanitize_tokenizer_config(tokenizer, args.adapter_output_dir)

    print("[7/7] Merging LoRA into base and exporting merged model...")
    merged_model = trainer.model.merge_and_unload()  # mandatory: zero adapter overhead at inference
    merged_model.config.use_cache = True
    merged_model.save_pretrained(
        args.merged_output_dir,
        safe_serialization=True,
        max_shard_size="20GB",
    )
    tokenizer.save_pretrained(args.merged_output_dir)
    _sanitize_tokenizer_config(tokenizer, args.merged_output_dir)
    _update_model_config(args.merged_output_dir, args.max_seq_length)

    weight_files = [f for f in os.listdir(args.merged_output_dir) if f.endswith(".safetensors")]
    if len(weight_files) != 1:
        raise RuntimeError(
            f"Merged model must be a single safetensors file, found: {weight_files}"
        )

    meta = {
        "base_model_id": args.base_model_id,
        "max_seq_length": args.max_seq_length,
        "korean_samples_target_requested": args.korean_samples,
        "code_samples_target_requested": args.code_samples,
        "korean_samples_target_adjusted": target_k,
        "code_samples_target_adjusted": target_c,
        "sampling_plan": sample_plan,
        "korean_stats": stats_k,
        "code_stats": stats_c,
        "final_train_size": len(train_ds),
        "min_tokens": args.min_tokens,
        "max_preferred_tokens": args.max_preferred_tokens,
        "max_backfill_tokens": args.max_backfill_tokens,
        "strict_preferred_only": args.strict_preferred_only,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "target_modules": TARGET_MODULES,
        "merged_weight_file": weight_files[0],
    }
    with open(os.path.join(args.merged_output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Adapter dir: {args.adapter_output_dir}")
    print(f"[DONE] Merged model dir: {args.merged_output_dir}")


if __name__ == "__main__":
    main()
