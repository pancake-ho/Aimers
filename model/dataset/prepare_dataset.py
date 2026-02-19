import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import Dataset, concatenate_datasets, interleave_datasets, load_dataset


_USER_ROLE_ALIASES = {
    "human",
    "instruction",
    "prompt",
    "question",
    "input",
}
_ASSISTANT_ROLE_ALIASES = {
    "assistant",
    "gpt",
    "bot",
    "model",
    "response",
    "output",
}
_SYSTEM_ROLE_ALIASES = {
    "system",
    "developer",
    "context",
}
_NULL_CONFIG_TOKENS = {"", "none", "null", "default", "all"}


def _empty_conversation_dataset() -> Dataset:
    return Dataset.from_dict({"conversations": []})


def _parse_csv_values(raw: Any, keep_empty: bool = False) -> List[str]:
    text = "" if raw is None else str(raw)
    chunks = [part.strip() for part in text.split(",")]
    if keep_empty:
        return chunks
    return [part for part in chunks if part]


def _normalize_config_token(token: Any) -> Optional[str]:
    if token is None:
        return None
    text = str(token).strip()
    if text.lower() in _NULL_CONFIG_TOKENS:
        return None
    return text


def _normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    lowered = str(role).strip().lower()
    if lowered in _SYSTEM_ROLE_ALIASES:
        return "system"
    if lowered == "user" or lowered in _USER_ROLE_ALIASES:
        return "user"
    if lowered == "assistant" or lowered in _ASSISTANT_ROLE_ALIASES:
        return "assistant"
    return None


def _normalize_content(content: Any) -> Optional[str]:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    return stripped if stripped else None


def _is_valid_conversations(conversations: Any) -> bool:
    if not isinstance(conversations, list) or not conversations:
        return False
    if conversations[-1].get("role") != "assistant":
        return False
    for turn in conversations:
        role = turn.get("role")
        content = turn.get("content")
        if role not in {"system", "user", "assistant"}:
            return False
        if not isinstance(content, str) or not content.strip():
            return False
    return True


def normalize_to_conversations(example: Dict[str, Any], turn_policy: Optional[str] = None):
    source = example.get("conversations")
    if source is None:
        source = example.get("messages")
    if not isinstance(source, list):
        return None

    normalized: List[Dict[str, str]] = []
    for turn in source:
        if not isinstance(turn, dict):
            continue
        role = _normalize_role(turn.get("role"))
        if role is None:
            continue
        content = _normalize_content(turn.get("content"))
        if content is None:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        return None

    if turn_policy is not None:
        normalized = apply_turn_policy(normalized, turn_policy)
        if normalized is None:
            return None

    return {"conversations": normalized}


def apply_turn_policy(conversations: List[Dict[str, str]], policy: str):
    if not isinstance(conversations, list) or not conversations:
        return None

    pol = str(policy or "last_assistant").strip().lower()
    if pol == "keep_full":
        selected = list(conversations)
    else:
        last_assistant_idx = None
        for idx in range(len(conversations) - 1, -1, -1):
            if conversations[idx].get("role") == "assistant":
                last_assistant_idx = idx
                break
        if last_assistant_idx is None:
            return None

        if pol == "last_assistant":
            selected = conversations[: last_assistant_idx + 1]
        elif pol == "two_turn":
            first_user_idx = None
            for idx, turn in enumerate(conversations):
                if turn.get("role") == "user":
                    first_user_idx = idx
                    break
            if first_user_idx is None or first_user_idx >= last_assistant_idx:
                return None
            selected = [conversations[first_user_idx], conversations[last_assistant_idx]]
        else:
            raise ValueError(f"unsupported mix turn policy: {policy}")

    if not _is_valid_conversations(selected):
        return None
    return selected


def compute_mixed_quotas(target_count: int, weights: Sequence[float]) -> List[int]:
    if target_count <= 0:
        return [0 for _ in weights]
    if not weights:
        raise ValueError("mix weights must not be empty")

    parsed = [max(0.0, float(w)) for w in weights]
    total = sum(parsed)
    if total <= 0.0:
        raise ValueError("mix weights must contain at least one positive value")

    normalized = [w / total for w in parsed]
    raw = [target_count * w for w in normalized]
    quotas = [int(math.floor(v)) for v in raw]

    remainder = int(target_count - sum(quotas))
    if remainder > 0:
        order = sorted(
            range(len(raw)),
            key=lambda i: (-(raw[i] - quotas[i]), i),
        )
        for idx in order[:remainder]:
            quotas[idx] += 1
    return quotas


def _load_hf_split(
    dataset_id: str,
    split: str,
    config_name: Optional[str] = None,
    streaming: bool = False,
):
    if config_name is None:
        return load_dataset(dataset_id, split=split, streaming=streaming)
    return load_dataset(dataset_id, config_name, split=split, streaming=streaming)


def _normalize_dataset_map(row: Dict[str, Any], turn_policy: str):
    converted = normalize_to_conversations(row)
    if converted is None:
        return {"conversations": []}
    applied = apply_turn_policy(converted["conversations"], turn_policy)
    if applied is None:
        return {"conversations": []}
    return {"conversations": applied}


def _prepare_source_dataset(
    *,
    dataset_id: str,
    split: str,
    config_name: Optional[str],
    turn_policy: str,
    seed: int,
    streaming: bool,
):
    raw = _load_hf_split(dataset_id, split=split, config_name=config_name, streaming=streaming)

    if streaming:
        raw = raw.shuffle(seed=seed, buffer_size=10000)
        prepared = raw.map(lambda ex: _normalize_dataset_map(ex, turn_policy))
        prepared = prepared.filter(lambda ex: _is_valid_conversations(ex.get("conversations")))
        return prepared

    prepared = raw.map(
        lambda ex: _normalize_dataset_map(ex, turn_policy),
        remove_columns=raw.column_names,
    )
    prepared = prepared.filter(lambda ex: _is_valid_conversations(ex.get("conversations")))
    prepared = prepared.shuffle(seed=seed)
    return prepared


def _parse_mix_lists(args) -> Tuple[List[str], List[str], List[Optional[str]], List[float]]:
    ext_ids = _parse_csv_values(getattr(args, "mix_dataset_ids", ""), keep_empty=False)
    if len(ext_ids) != 2:
        raise ValueError(f"expected 2 external mix dataset ids, got {len(ext_ids)}: {ext_ids}")

    split_tokens = _parse_csv_values(getattr(args, "mix_dataset_splits", ""), keep_empty=True)
    while len(split_tokens) < 2:
        split_tokens.append("train")
    ext_splits = [(token or "train").strip() for token in split_tokens[:2]]

    config_tokens = _parse_csv_values(getattr(args, "mix_dataset_configs", ""), keep_empty=True)
    while len(config_tokens) < 2:
        config_tokens.append("")
    ext_configs = [_normalize_config_token(token) for token in config_tokens[:2]]

    mix_weights = [float(x) for x in _parse_csv_values(getattr(args, "mix_weights", ""), keep_empty=False)]
    if len(mix_weights) != 3:
        raise ValueError(f"expected 3 mix weights (base+2 external), got {len(mix_weights)}")
    total = sum(max(0.0, w) for w in mix_weights)
    if total <= 0.0:
        raise ValueError("mix weights must contain at least one positive value")
    normalized_weights = [max(0.0, w) / total for w in mix_weights]

    return ext_ids, ext_splits, ext_configs, normalized_weights


def build_mixed_train_dataset(args, target_count: int, stage: str, seed: int):
    ext_ids, ext_splits, ext_configs, weights = _parse_mix_lists(args)
    turn_policy = str(getattr(args, "mix_turn_policy", "last_assistant"))
    streaming = bool(getattr(args, "mix_streaming", False))

    base_id = str(args.dataset_id)
    source_defs = [
        {"name": "base", "dataset_id": base_id, "split": str(args.dataset_split), "config": None, "seed": int(seed)},
        {"name": "ext0", "dataset_id": ext_ids[0], "split": ext_splits[0], "config": ext_configs[0], "seed": int(seed) + 17},
        {"name": "ext1", "dataset_id": ext_ids[1], "split": ext_splits[1], "config": ext_configs[1], "seed": int(seed) + 29},
    ]

    mix_counts = {"base": 0, "ext0": 0, "ext1": 0}
    if target_count <= 0:
        return _empty_conversation_dataset(), {
            "mix_applied": True,
            "stage": stage,
            "target_count": int(target_count),
            "actual_count": 0,
            "shortfall": int(max(0, target_count)),
            "mix_sources": [base_id, ext_ids[0], ext_ids[1]],
            "mix_weights": weights,
            "mix_counts": mix_counts,
            "mix_turn_policy": turn_policy,
        }

    if streaming:
        iterable_sources = []
        for spec in source_defs:
            ds = _prepare_source_dataset(
                dataset_id=spec["dataset_id"],
                split=spec["split"],
                config_name=spec["config"],
                turn_policy=turn_policy,
                seed=spec["seed"],
                streaming=True,
            )
            ds = ds.map(
                lambda ex, source_name=spec["name"]: {
                    "conversations": ex["conversations"],
                    "__source": source_name,
                }
            )
            iterable_sources.append(ds)

        try:
            mixed = interleave_datasets(
                iterable_sources,
                probabilities=weights,
                seed=int(seed),
                stopping_strategy="all_exhausted",
            )
        except TypeError:
            mixed = interleave_datasets(
                iterable_sources,
                probabilities=weights,
                seed=int(seed),
            )

        rows: List[Dict[str, Any]] = []
        for ex in mixed.take(int(target_count)):
            conv = ex.get("conversations")
            source_name = ex.get("__source")
            if not _is_valid_conversations(conv):
                continue
            if source_name not in mix_counts:
                continue
            mix_counts[source_name] += 1
            rows.append({"conversations": conv})

        final_ds = Dataset.from_list(rows) if rows else _empty_conversation_dataset()
        actual_count = len(final_ds)
        shortfall = max(0, int(target_count) - int(actual_count))
        return final_ds, {
            "mix_applied": True,
            "stage": stage,
            "target_count": int(target_count),
            "actual_count": int(actual_count),
            "shortfall": int(shortfall),
            "mix_sources": [base_id, ext_ids[0], ext_ids[1]],
            "mix_weights": weights,
            "mix_counts": mix_counts,
            "mix_turn_policy": turn_policy,
        }

    quotas = compute_mixed_quotas(int(target_count), weights)

    prepared_sources = []
    capacities = []
    for idx, spec in enumerate(source_defs):
        ds = _prepare_source_dataset(
            dataset_id=spec["dataset_id"],
            split=spec["split"],
            config_name=spec["config"],
            turn_policy=turn_policy,
            seed=spec["seed"],
            streaming=False,
        )
        prepared_sources.append(ds)
        capacities.append(len(ds))

        take = min(int(quotas[idx]), len(ds))
        mix_counts[spec["name"]] = int(take)

    shortfall = int(target_count) - sum(mix_counts.values())
    if shortfall > 0:
        for spec, capacity in zip(source_defs, capacities):
            if shortfall <= 0:
                break
            name = spec["name"]
            available = int(capacity) - int(mix_counts[name])
            if available <= 0:
                continue
            extra = min(shortfall, available)
            mix_counts[name] += int(extra)
            shortfall -= int(extra)

    selected_parts: List[Dataset] = []
    for ds, spec in zip(prepared_sources, source_defs):
        n = int(mix_counts[spec["name"]])
        if n <= 0:
            continue
        selected_parts.append(ds.select(range(n)))

    if selected_parts:
        final_ds = concatenate_datasets(selected_parts).shuffle(seed=int(seed))
    else:
        final_ds = _empty_conversation_dataset()

    actual_count = len(final_ds)
    shortfall = max(0, int(target_count) - int(actual_count))

    return final_ds, {
        "mix_applied": True,
        "stage": stage,
        "target_count": int(target_count),
        "actual_count": int(actual_count),
        "shortfall": int(shortfall),
        "mix_sources": [base_id, ext_ids[0], ext_ids[1]],
        "mix_weights": weights,
        "mix_counts": mix_counts,
        "mix_turn_policy": turn_policy,
    }


def prepare_dataset(ds_id, ds_split, num_train, num_calib, seed=42):
    ds = load_dataset(ds_id, split=ds_split)
    ds = ds.shuffle(seed=int(seed))

    train_n = max(0, int(num_train))
    calib_n = max(0, int(num_calib))
    training_dataset = ds.select(range(min(train_n, len(ds))))
    calib_start = len(training_dataset)
    calib_end = min(calib_start + calib_n, len(ds))
    calib_dataset = ds.select(range(calib_start, calib_end))
    return training_dataset, calib_dataset


def make_calib_dataset(tokenizer, dataset_id, split, n, seed=42):
    try:
        it = load_dataset(dataset_id, split=split, streaming=True)
        it = it.shuffle(seed=seed, buffer_size=10000)
        samples = []
        for ex in it.take(n):
            conv = normalize_to_conversations(ex)
            if conv is None:
                continue
            turns = apply_turn_policy(conv["conversations"], "last_assistant")
            if turns is None:
                continue
            text = tokenizer.apply_chat_template(
                turns,
                add_generation_prompt=True,
                tokenize=False,
            )
            samples.append({"text": text})
        return Dataset.from_list(samples)
    except Exception as e:
        print("streaming calibration loading failed; fallback to standard loading")
        print(f"[INFO] fallback reason: {e}")

        ds = load_dataset(dataset_id, split=f"{split}[:{n}]")

        def _pp(ex):
            conv = normalize_to_conversations(ex)
            if conv is None:
                return {"text": ""}
            turns = apply_turn_policy(conv["conversations"], "last_assistant")
            if turns is None:
                return {"text": ""}
            return {
                "text": tokenizer.apply_chat_template(
                    turns,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            }

        mapped = ds.map(_pp)
        return mapped.filter(lambda row: bool(row.get("text")))
