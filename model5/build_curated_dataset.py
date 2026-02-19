#!/usr/bin/env python
"""
Build a stronger mixed SFT dataset for model5.

Sources:
- MyeongHo0621/korean-quality-cleaned
- m-a-p/Code-Feedback
- openai/gsm8k (main)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curated mixed dataset for model5")
    p.add_argument("--base_model_id", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    p.add_argument("--output_dir", type=str, default="./curated_train_dataset")
    p.add_argument("--target_size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_tokens", type=int, default=96)
    p.add_argument("--max_tokens", type=int, default=1200)

    p.add_argument("--korean_dataset", type=str, default="MyeongHo0621/korean-quality-cleaned")
    p.add_argument("--korean_config", type=str, default="default")
    p.add_argument("--code_dataset", type=str, default="m-a-p/Code-Feedback")
    p.add_argument("--code_config", type=str, default=None)
    p.add_argument("--math_dataset", type=str, default="openai/gsm8k")
    p.add_argument("--math_config", type=str, default="main")

    p.add_argument("--korean_ratio", type=float, default=0.55)
    p.add_argument("--code_ratio", type=float, default=0.25)
    p.add_argument("--math_ratio", type=float, default=0.20)
    return p.parse_args()


def _normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    role_map = {"human": "user", "gpt": "assistant", "bot": "assistant"}
    role = role_map.get(role, role)
    if role not in {"system", "user", "assistant"}:
        role = "user"
    return role


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
    out: List[Dict[str, str]] = []
    for msg in raw_messages:
        if not isinstance(msg, dict):
            continue
        role = _normalize_role(msg.get("role", msg.get("from", "user")))
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
    if "text" in example:
        text = str(example["text"]).strip()
        if text:
            return [{"role": "user", "content": text}]
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


def _load_split(dataset_name: str, dataset_config: str | None, split: str = "train") -> Dataset:
    if dataset_config:
        try:
            ds = load_dataset(dataset_name, dataset_config, split=split)
            return ds
        except ValueError as e:
            msg = str(e)
            if "BuilderConfig" not in msg or "not found" not in msg:
                raise
    ds = load_dataset(dataset_name)
    if isinstance(ds, DatasetDict):
        if split in ds:
            return ds[split]
        first_key = next(iter(ds.keys()))
        return ds[first_key]
    return ds


def _process_source(
    dataset_name: str,
    dataset_config: str | None,
    source_name: str,
    tokenizer: AutoTokenizer,
    min_tokens: int,
    max_tokens: int,
) -> Dataset:
    ds = _load_split(dataset_name, dataset_config, split="train")

    def _map(ex: Dict[str, Any]) -> Dict[str, Any]:
        messages = _to_messages(ex)
        if not messages:
            return {"text": "", "source": source_name, "tok_len": 0}
        text = _messages_to_text(messages, tokenizer)
        tok_len = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
        return {"text": text, "source": source_name, "tok_len": tok_len}

    ds = ds.map(_map, remove_columns=ds.column_names, desc=f"Formatting {source_name}")
    ds = ds.filter(
        lambda x: len(x["text"]) > 0 and min_tokens <= x["tok_len"] <= max_tokens,
        desc=f"Filtering {source_name}",
    )
    return ds


def _take(ds: Dataset, n: int, seed: int) -> Dataset:
    if len(ds) == 0 or n <= 0:
        return ds.select([])
    if n >= len(ds):
        return ds.shuffle(seed=seed)
    return ds.shuffle(seed=seed).select(range(n))


def main() -> None:
    args = parse_args()

    ratio_sum = args.korean_ratio + args.code_ratio + args.math_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[1/4] Loading and formatting source datasets...")
    ds_k = _process_source(
        args.korean_dataset,
        args.korean_config,
        "korean_quality",
        tokenizer,
        args.min_tokens,
        args.max_tokens,
    )
    ds_c = _process_source(
        args.code_dataset,
        args.code_config,
        "code_feedback",
        tokenizer,
        args.min_tokens,
        args.max_tokens,
    )
    ds_m = _process_source(
        args.math_dataset,
        args.math_config,
        "gsm8k",
        tokenizer,
        args.min_tokens,
        args.max_tokens,
    )
    print(f"[INFO] Available after filtering - korean: {len(ds_k)}, code: {len(ds_c)}, math: {len(ds_m)}")

    print("[2/4] Sampling by target ratios...")
    n_k = int(args.target_size * args.korean_ratio)
    n_c = int(args.target_size * args.code_ratio)
    n_m = args.target_size - n_k - n_c

    part_k = _take(ds_k, n_k, args.seed)
    part_c = _take(ds_c, n_c, args.seed + 1)
    part_m = _take(ds_m, n_m, args.seed + 2)

    parts = [p for p in [part_k, part_c, part_m] if len(p) > 0]
    if not parts:
        raise RuntimeError("No valid samples were produced from source datasets.")

    mixed = concatenate_datasets(parts).shuffle(seed=args.seed)

    if len(mixed) < args.target_size:
        pool = concatenate_datasets([ds_k, ds_c, ds_m]).shuffle(seed=args.seed + 99)
        need = min(args.target_size - len(mixed), len(pool))
        if need > 0:
            mixed = concatenate_datasets([mixed, pool.select(range(need))]).shuffle(seed=args.seed)

    if len(mixed) > args.target_size:
        mixed = mixed.select(range(args.target_size))

    print(f"[INFO] Final dataset size: {len(mixed)}")

    print("[3/4] Saving dataset...")
    if os.path.isdir(args.output_dir):
        import shutil

        shutil.rmtree(args.output_dir)
    mixed.save_to_disk(args.output_dir)

    print("[4/4] Writing metadata...")
    by_source = {}
    for s in mixed["source"]:
        by_source[s] = by_source.get(s, 0) + 1
    meta = {
        "target_size": args.target_size,
        "actual_size": len(mixed),
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "sources": by_source,
    }
    with open(os.path.join(args.output_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Saved curated dataset to {args.output_dir}")


if __name__ == "__main__":
    main()

