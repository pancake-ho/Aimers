from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def gptq_quant_key_ok(keys: Iterable[str]) -> bool:
    hints = (
        "qweight",
        "qzeros",
        "scales",
        "weight_packed",
        "weight_scale",
        "weight_zero_point",
    )
    lowered = [str(key).lower() for key in keys]
    return any(any(hint in key for hint in hints) for key in lowered)


def awq_quant_meta_ok(recipe_text: str, config: Dict[str, Any] | None) -> bool:
    if recipe_text:
        lowered = recipe_text.lower()
        if "awqmodifier" in lowered or "awq" in lowered:
            return True
    if config:
        q_cfg = config.get("quantization_config")
        if q_cfg and "awq" in json.dumps(q_cfg).lower():
            return True
    return False


def awq_quant_meta_ok_from_dir(model_dir: Path) -> bool:
    recipe_text = ""
    recipe_path = model_dir / "recipe.yaml"
    if recipe_path.exists():
        recipe_text = recipe_path.read_text(encoding="utf-8", errors="ignore")

    config = None
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = None
    return awq_quant_meta_ok(recipe_text, config)
