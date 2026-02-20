#!/usr/bin/env python3
"""Shared guardrail helpers for Aimers Phase2 submission packaging."""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


WEIGHT_GLOBS = (
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "pytorch_model*.bin.index.json",
    "model*.pt",
)


OPTIONAL_CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "quantization_config.json",
    "compression_config.json",
    "compressed_tensors_config.json",
)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": self.details,
        }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_top_level_zip_entries(zf: zipfile.ZipFile) -> Set[str]:
    top: Set[str] = set()
    for member in zf.namelist():
        if not member or member.startswith("__MACOSX/"):
            continue
        head = member.split("/", 1)[0]
        if head:
            top.add(head)
    return top


def _collect_weight_files(model_dir: Path) -> List[str]:
    found: Set[str] = set()
    for pattern in WEIGHT_GLOBS:
        for path in model_dir.glob(pattern):
            if path.is_file():
                found.add(path.name)
    return sorted(found)


def _tokenizer_assets_ok(model_dir: Path) -> Tuple[bool, Dict[str, Any], Optional[Dict[str, Any]]]:
    details: Dict[str, Any] = {}
    tokenizer_cfg_path = model_dir / "tokenizer_config.json"
    if not tokenizer_cfg_path.exists():
        details["tokenizer_config"] = False
        return False, details, None

    tokenizer_cfg = _load_json(tokenizer_cfg_path)
    details["tokenizer_config"] = True
    details["tokenizer_json"] = (model_dir / "tokenizer.json").exists()
    details["tokenizer_model"] = (model_dir / "tokenizer.model").exists()
    details["vocab_json"] = (model_dir / "vocab.json").exists()
    details["merges_txt"] = (model_dir / "merges.txt").exists()

    has_vocab_pair = bool(details["vocab_json"] and details["merges_txt"])
    has_tokenizer_format = bool(details["tokenizer_json"] or details["tokenizer_model"] or has_vocab_pair)
    details["has_supported_tokenizer_format"] = has_tokenizer_format
    return has_tokenizer_format, details, tokenizer_cfg


def _resolve_auto_map_python_files(config: Dict[str, Any]) -> Set[str]:
    auto_map = config.get("auto_map")
    if not isinstance(auto_map, dict):
        return set()

    modules: Set[str] = set()

    def _extract(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, str):
                    yield item

    for value in auto_map.values():
        for candidate in _extract(value):
            # Example: "exaone.modeling_exaone.EXAONEForCausalLM" -> "modeling_exaone.py"
            parts = candidate.split(".")
            if len(parts) < 2:
                continue
            module_name = parts[-2]
            if module_name:
                modules.add(f"{module_name}.py")
    return modules


def validate_model_dir(model_dir: Path) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {"model_dir": str(model_dir)}

    if not model_dir.exists() or not model_dir.is_dir():
        return ValidationResult(
            ok=False,
            errors=[f"model directory does not exist: {model_dir}"],
            warnings=[],
            details=details,
        )

    config_path = model_dir / "config.json"
    if not config_path.exists():
        errors.append("missing required file: config.json")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, details=details)

    try:
        config = _load_json(config_path)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"invalid config.json: {exc}")
        return ValidationResult(ok=False, errors=errors, warnings=warnings, details=details)

    weight_files = _collect_weight_files(model_dir)
    details["weight_files"] = weight_files
    if not weight_files:
        errors.append("missing model weight files (expected *.safetensors or pytorch_model*.bin)")

    tok_ok, tok_details, tokenizer_cfg = _tokenizer_assets_ok(model_dir)
    details["tokenizer_assets"] = tok_details
    if not tok_ok:
        errors.append(
            "missing tokenizer assets: require tokenizer_config.json and one format "
            "(tokenizer.json OR tokenizer.model OR vocab.json+merges.txt)"
        )

    has_chat_template = False
    if tokenizer_cfg is not None:
        chat_template = tokenizer_cfg.get("chat_template")
        if isinstance(chat_template, str) and chat_template.strip():
            has_chat_template = True
    if not has_chat_template and (model_dir / "chat_template.jinja").exists():
        has_chat_template = True
    details["chat_template_ready"] = has_chat_template
    if not has_chat_template:
        errors.append(
            "missing chat template for apply_chat_template "
            "(tokenizer_config.json:chat_template or chat_template.jinja)"
        )

    required_py = sorted(_resolve_auto_map_python_files(config))
    details["auto_map_python_files"] = required_py
    missing_py = [name for name in required_py if not (model_dir / name).exists()]
    if missing_py:
        errors.append(
            "missing trust_remote_code assets referenced by config.auto_map: "
            + ", ".join(missing_py)
        )

    if not required_py:
        warnings.append("config.auto_map not found; ensure remote-code assets are not required")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, details=details)


def validate_submit_zip(zip_path: Path) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {"zip_path": str(zip_path)}

    if not zip_path.exists() or not zip_path.is_file():
        return ValidationResult(
            ok=False,
            errors=[f"zip file does not exist: {zip_path}"],
            warnings=[],
            details=details,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            top = sorted(_iter_top_level_zip_entries(zf))
            details["zip_top_level"] = top
            if top != ["model"]:
                errors.append(
                    "submit.zip root must contain exactly one top-level directory named 'model/'"
                )
            with tempfile.TemporaryDirectory(prefix="aimers_submit_check_") as tmp:
                tmp_path = Path(tmp)
                zf.extractall(tmp_path)
                model_dir = tmp_path / "model"
                model_result = validate_model_dir(model_dir)
                details["model_validation"] = model_result.to_dict()
                errors.extend(model_result.errors)
                warnings.extend(model_result.warnings)
    except zipfile.BadZipFile as exc:
        errors.append(f"invalid zip file: {exc}")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"unexpected zip validation failure: {exc}")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings, details=details)
