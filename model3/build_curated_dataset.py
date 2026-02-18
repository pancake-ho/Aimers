#!/usr/bin/env python
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import random
import re

# -----------------------------
# Config
# -----------------------------
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
TARGET_SIZE = 2048
SEED = 42

REASONING_KEYWORDS = [
    "step by step",
    "therefore",
    "calculate",
    "let's think",
]

MIN_TOKENS = 512
MAX_TOKENS = 1024

random.seed(SEED)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def contains_reasoning(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in REASONING_KEYWORDS)


def classify_bucket(text: str) -> str:
    t = text.lower()
    if re.search(r"\d+\s*[\+\-\*/=]\s*\d+|integral|derivative|equation|solve|calculate|proof", t):
        return "math"
    if any(k in t for k in ["therefore", "hence", "if", "then", "because", "step by step", "let's think"]):
        return "logic"
    return "general"


def token_len(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def manta_to_record(example):
    text = tokenizer.apply_chat_template(
        example["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "text": text,
        "source": "manta",
        "reasoning_hit": contains_reasoning(text),
        "bucket": classify_bucket(text),
        "tok_len": token_len(text),
    }


def gsm8k_to_record(example):
    messages = [
        {"role": "user", "content": example["question"].strip()},
        {"role": "assistant", "content": example["answer"].strip()},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {
        "text": text,
        "source": "gsm8k",
        "reasoning_hit": contains_reasoning(text),
        "bucket": classify_bucket(text),
        "tok_len": token_len(text),
    }


def main():
    manta_raw = load_dataset("LGAI-EXAONE/MANTA-1M", split="train")
    gsm_raw = load_dataset("openai/gsm8k", "main", split="train")

    manta = manta_raw.map(manta_to_record, remove_columns=manta_raw.column_names)
    gsm = gsm_raw.map(gsm8k_to_record, remove_columns=gsm_raw.column_names)
    all_ds = concatenate_datasets([manta, gsm])

    filtered = all_ds.filter(
        lambda x: x["reasoning_hit"] and (MIN_TOKENS <= x["tok_len"] <= MAX_TOKENS)
    )

    if len(filtered) < TARGET_SIZE:
        relaxed = all_ds.filter(lambda x: MIN_TOKENS <= x["tok_len"] <= MAX_TOKENS)
        filtered = relaxed

    by_bucket = {
        "math": filtered.filter(lambda x: x["bucket"] == "math"),
        "logic": filtered.filter(lambda x: x["bucket"] == "logic"),
        "general": filtered.filter(lambda x: x["bucket"] == "general"),
    }

    per_bucket = TARGET_SIZE // 3
    selected_parts = []
    for b in ["math", "logic", "general"]:
        ds_b = by_bucket[b].shuffle(seed=SEED)
        take_n = min(per_bucket, len(ds_b))
        if take_n > 0:
            selected_parts.append(ds_b.select(range(take_n)))

    selected = concatenate_datasets(selected_parts) if selected_parts else filtered.select([])

    remaining = TARGET_SIZE - len(selected)
    if remaining > 0:
        pool = filtered.shuffle(seed=SEED)
        take_n = min(remaining, len(pool))
        selected = concatenate_datasets([selected, pool.select(range(take_n))])

    train_dataset = selected.shuffle(seed=SEED)
    if len(train_dataset) < TARGET_SIZE:
        raise ValueError(
            f"Could only build {len(train_dataset)} samples. "
            "Try relaxing MIN/MAX_TOKENS or keyword strictness."
        )
    train_dataset = train_dataset.select(range(TARGET_SIZE))

    print(train_dataset)
    print(train_dataset[0].keys())

    out_dir = "./curated_train_dataset"
    train_dataset.save_to_disk(out_dir)
    print(f"Saved curated dataset to {out_dir}")


if __name__ == "__main__":
    main()
