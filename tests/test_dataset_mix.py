from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = _load_module(MODEL_ROOT / "dataset" / "prepare_dataset.py", "prepare_dataset_mod")


def test_normalize_messages_to_conversations_last_assistant() -> None:
    ex = {
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "human", "content": "u1"},
            {"role": "gpt", "content": "a1"},
            {"role": "user", "content": "u2"},
        ]
    }
    out = prep.normalize_to_conversations(ex, turn_policy="last_assistant")
    assert out is not None
    conv = out["conversations"]
    assert conv[-1]["role"] == "assistant"
    assert [t["content"] for t in conv] == ["s", "u1", "a1"]


def test_apply_turn_policy_two_turn() -> None:
    conv = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    out = prep.apply_turn_policy(conv, "two_turn")
    assert out is not None
    assert out == [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a2"},
    ]


def test_normalize_rejects_without_assistant() -> None:
    ex = {"messages": [{"role": "user", "content": "u"}]}
    out = prep.normalize_to_conversations(ex, turn_policy="last_assistant")
    assert out is None


def test_compute_mixed_quotas_is_deterministic() -> None:
    quotas = prep.compute_mixed_quotas(10, [0.60, 0.25, 0.15])
    assert quotas == [6, 3, 1]
    assert sum(quotas) == 10


def test_build_mixed_train_dataset_backfills_from_base_and_normalizes_all_config(monkeypatch) -> None:
    base_rows = [
        {"conversations": [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]}
        for i in range(6)
    ]
    ext1_rows = [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "r1"}]}
    ]
    # Invalid rows for last_assistant policy: no assistant.
    ext2_rows = [
        {"messages": [{"role": "user", "content": "only-user"}]}
    ]

    data_map = {
        ("base_ds", "", "train"): Dataset.from_list(base_rows),
        ("ext1_ds", "", "train"): Dataset.from_list(ext1_rows),
        ("ext2_ds", "", "train"): Dataset.from_list(ext2_rows),
    }

    def _fake_load_dataset(dataset_id, *args, split=None, streaming=False, **kwargs):
        _ = kwargs, streaming
        config = args[0] if args else ""
        key = (dataset_id, config, split)
        if key not in data_map:
            raise KeyError(f"unknown dataset key: {key}")
        return data_map[key]

    monkeypatch.setattr(prep, "load_dataset", _fake_load_dataset)

    args = SimpleNamespace(
        dataset_id="base_ds",
        dataset_split="train",
        mix_dataset_ids="ext1_ds,ext2_ds",
        mix_dataset_splits="train,train",
        mix_dataset_configs=",all",
        mix_weights="0.50,0.25,0.25",
        mix_turn_policy="last_assistant",
        mix_apply_stages="kd,lora",
        mix_streaming=False,
    )

    mixed, meta = prep.build_mixed_train_dataset(args=args, target_count=6, stage="kd", seed=42)

    assert len(mixed) == 6
    assert meta["mix_applied"] is True
    assert meta["shortfall"] == 0
    assert meta["mix_counts"]["base"] == 5
    assert meta["mix_counts"]["ext0"] == 1
    assert meta["mix_counts"]["ext1"] == 0
    assert meta["mix_sources"] == ["base_ds", "ext1_ds", "ext2_ds"]
