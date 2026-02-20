from __future__ import annotations

import importlib.util
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
CALIB_PATH = MODEL_ROOT / "utils" / "calib.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_downscale_policy_uses_min_without_replacement() -> None:
    mod = _load_module(CALIB_PATH, "calib_mod_downscale")
    indices, meta = mod.select_calib_indices(requested=10, available=6, policy="downscale", seed=7)
    assert indices == [0, 1, 2, 3, 4, 5]
    assert meta["calib_n_requested"] == 10
    assert meta["calib_n_available"] == 6
    assert meta["calib_n_used"] == 6
    assert meta["sampling_method"] == "without_replacement"


def test_replacement_policy_matches_requested_and_is_seeded() -> None:
    mod = _load_module(CALIB_PATH, "calib_mod_replacement")
    idx1, meta1 = mod.select_calib_indices(requested=8, available=3, policy="replacement", seed=42)
    idx2, meta2 = mod.select_calib_indices(requested=8, available=3, policy="replacement", seed=42)
    assert len(idx1) == 8
    assert idx1 == idx2
    assert meta1["calib_n_used"] == 8
    assert meta2["calib_n_used"] == 8
    assert meta1["sampling_method"] == "with_replacement"
