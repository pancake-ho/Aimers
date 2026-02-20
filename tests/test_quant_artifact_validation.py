from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
QA_PATH = MODEL_ROOT / "utils" / "quant_artifact.py"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gptq_quant_key_ok_detects_expected_markers() -> None:
    mod = _load_module(QA_PATH, "qa_mod_keys")
    assert mod.gptq_quant_key_ok(["layers.0.mlp.weight_packed"]) is True
    assert mod.gptq_quant_key_ok(["layers.0.mlp.weight"]) is False


def test_awq_quant_meta_ok_detects_recipe_or_config(tmp_path: Path) -> None:
    mod = _load_module(QA_PATH, "qa_mod_awq")
    assert mod.awq_quant_meta_ok("default_stage:\n  AWQModifier: {}", None) is True

    cfg = {"quantization_config": {"quant_method": "awq"}}
    assert mod.awq_quant_meta_ok("", cfg) is True
    assert mod.awq_quant_meta_ok("", {"quantization_config": {"quant_method": "gptq"}}) is False

    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "recipe.yaml").write_text("default_stage:\n  AWQModifier: {}", encoding="utf-8")
    (model_dir / "config.json").write_text(json.dumps({"quantization_config": {}}), encoding="utf-8")
    assert mod.awq_quant_meta_ok_from_dir(model_dir) is True
