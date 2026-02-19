#!/usr/bin/env python
"""
Model7 stage-3:
Create a clean submit.zip with only runtime-required files in top-level model/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List


ALLOWED_STATIC_FILES = {
    "config.json",
    "generation_config.json",
    "quantization_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.model",
    "chat_template.jinja",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package a clean model/ submit.zip for vLLM")
    p.add_argument("--source_dir", type=str, default="./model")
    p.add_argument("--zip_name", type=str, default="submit")
    p.add_argument("--staging_dir", type=str, default="./_submit_staging")
    p.add_argument("--max_model_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--repetition_penalty", type=float, default=1.08)
    return p.parse_args()


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _sanitize_runtime_configs(model_dir: Path, max_model_len: int, max_new_tokens: int, repetition_penalty: float) -> None:
    cfg_path = model_dir / "config.json"
    cfg = _load_json(cfg_path)
    if cfg:
        cfg["max_model_len"] = int(max_model_len)
        cfg["use_cache"] = True
        _save_json(cfg_path, cfg)

    gen_path = model_dir / "generation_config.json"
    gen = _load_json(gen_path)
    gen["max_new_tokens"] = int(max_new_tokens)
    gen["repetition_penalty"] = float(repetition_penalty)
    gen["do_sample"] = False
    gen["temperature"] = 0.0
    gen["top_p"] = 1.0
    _save_json(gen_path, gen)

    tok_cfg_path = model_dir / "tokenizer_config.json"
    tok_cfg = _load_json(tok_cfg_path)
    if "fix_mistral_regex" in tok_cfg:
        tok_cfg.pop("fix_mistral_regex", None)
        _save_json(tok_cfg_path, tok_cfg)


def _copy_clean_model(source_dir: Path, staging_model_dir: Path) -> List[str]:
    copied: List[str] = []
    ignored: List[str] = []
    for item in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
        if item.is_dir():
            ignored.append(item.name)
            continue
        name = item.name
        if name.endswith(".safetensors") or name in ALLOWED_STATIC_FILES:
            shutil.copy2(item, staging_model_dir / name)
            copied.append(name)
        else:
            ignored.append(name)

    print("[INFO] Ignored files/directories:", ", ".join(ignored) if ignored else "(none)")
    print("[INFO] Included files:", ", ".join(copied) if copied else "(none)")
    return copied


def _validate_packaged_model(model_dir: Path, copied: List[str]) -> None:
    if not (model_dir / "config.json").is_file():
        raise RuntimeError("Missing required file: config.json")
    if not (model_dir / "tokenizer_config.json").is_file():
        raise RuntimeError("Missing required file: tokenizer_config.json")

    has_tokenizer = (
        (model_dir / "tokenizer.json").is_file()
        or ((model_dir / "vocab.json").is_file() and (model_dir / "merges.txt").is_file())
        or (model_dir / "tokenizer.model").is_file()
    )
    if not has_tokenizer:
        raise RuntimeError("Missing tokenizer assets (tokenizer.json OR vocab+merges OR tokenizer.model).")

    weights = [f for f in copied if f.endswith(".safetensors")]
    if len(weights) != 1:
        raise RuntimeError(f"Expected exactly one safetensors weight file, found: {weights}")

    for bad_key in ("adapter", "_trainer_ckpt", "optimizer", "scheduler"):
        if any(bad_key in name.lower() for name in copied):
            raise RuntimeError(f"Redundant artifact detected in package: {bad_key}")


def _make_zip(staging_dir: Path, zip_name: str) -> None:
    zip_path = Path(f"{zip_name}.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=str(staging_dir),
        base_dir="model",
    )


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.is_dir():
        raise RuntimeError(f"Source model directory not found: {source_dir}")

    staging_dir = Path(args.staging_dir).resolve()
    staging_model_dir = staging_dir / "model"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_model_dir.mkdir(parents=True, exist_ok=True)

    copied = _copy_clean_model(source_dir, staging_model_dir)
    _sanitize_runtime_configs(
        staging_model_dir,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    _validate_packaged_model(staging_model_dir, copied)
    _make_zip(staging_dir, args.zip_name)

    print(f"[DONE] Created {args.zip_name}.zip with clean top-level model/ directory.")


if __name__ == "__main__":
    main()
