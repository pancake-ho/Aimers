from __future__ import annotations

import importlib.util
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gptq_grid_mod = _load_module(MODEL_ROOT / "quantizing" / "gptq_grid.py", "gptq_grid_mod")
build_deterministic_gptq_trials = gptq_grid_mod.build_deterministic_gptq_trials


def test_gptq_grid_has_six_deterministic_trials() -> None:
    trials = build_deterministic_gptq_trials(
        group_sizes_csv="64,128",
        dampening_csv="0.005,0.010,0.020",
        block_size=128,
        calib_samples=1024,
        calib_seq_len=1024,
        search_budget=6,
    )
    assert len(trials) == 6
    assert [(t["group_size"], t["dampening"]) for t in trials] == [
        (64, 0.005),
        (64, 0.01),
        (64, 0.02),
        (128, 0.005),
        (128, 0.01),
        (128, 0.02),
    ]
    assert all(t["block_size"] == 128 for t in trials)
    assert all(t["calib_samples"] == 1024 for t in trials)


def test_gptq_grid_respects_budget_prefix() -> None:
    trials = build_deterministic_gptq_trials(
        group_sizes_csv="64,128",
        dampening_csv="0.005,0.010,0.020",
        block_size=128,
        calib_samples=1024,
        calib_seq_len=2048,
        search_budget=4,
    )
    assert len(trials) == 4
    assert trials[-1]["group_size"] == 128
    assert trials[-1]["dampening"] == 0.005
    assert all(t["calib_seq_len"] == 2048 for t in trials)
