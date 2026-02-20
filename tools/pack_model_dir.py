#!/usr/bin/env python3
"""Pack final model directory by preserving base assets and replacing weights/config."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

try:
    from .submit_guardrails import OPTIONAL_CONFIG_FILES, WEIGHT_GLOBS, validate_model_dir
except ImportError:  # pragma: no cover
    from submit_guardrails import OPTIONAL_CONFIG_FILES, WEIGHT_GLOBS, validate_model_dir


def _collect_files(source_dir: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(p for p in source_dir.glob(pattern) if p.is_file()))
    return files


def _remove_files(model_dir: Path, patterns: List[str]) -> int:
    removed = 0
    for pattern in patterns:
        for path in model_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def _copy_flat(files: List[Path], destination: Path) -> int:
    copied = 0
    for file_path in files:
        shutil.copy2(file_path, destination / file_path.name)
        copied += 1
    return copied


def create_submit_zip(model_dir: Path, zip_out: Path) -> Path:
    zip_out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aimers_pack_") as tmp:
        tmp_path = Path(tmp)
        staging = tmp_path / "model"
        shutil.copytree(model_dir, staging)
        base = zip_out.with_suffix("")
        created = Path(
            shutil.make_archive(
                base_name=str(base),
                format="zip",
                root_dir=tmp_path,
                base_dir="model",
            )
        )
        return created


def pack_model_dir(base_model_dir: Path, weights_dir: Path, out_model_dir: Path) -> Dict[str, object]:
    if not base_model_dir.exists() or not base_model_dir.is_dir():
        raise FileNotFoundError(f"base model dir not found: {base_model_dir}")
    if not weights_dir.exists() or not weights_dir.is_dir():
        raise FileNotFoundError(f"weights dir not found: {weights_dir}")

    if out_model_dir.exists():
        shutil.rmtree(out_model_dir)
    out_model_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base_model_dir, out_model_dir)

    removed = _remove_files(out_model_dir, list(WEIGHT_GLOBS))
    copied_weights = _copy_flat(_collect_files(weights_dir, list(WEIGHT_GLOBS)), out_model_dir)

    copied_configs = 0
    for name in OPTIONAL_CONFIG_FILES:
        src = weights_dir / name
        if src.exists() and src.is_file():
            shutil.copy2(src, out_model_dir / name)
            copied_configs += 1

    result = validate_model_dir(out_model_dir)
    if not result.ok:
        raise RuntimeError(
            "packed model failed validation: " + "; ".join(result.errors)
        )

    return {
        "out_model_dir": str(out_model_dir),
        "removed_old_weight_files": removed,
        "copied_weight_files": copied_weights,
        "copied_config_files": copied_configs,
        "warnings": result.warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack model dir for Aimers Phase2 submission")
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--weights-dir", type=str, required=True)
    parser.add_argument("--out-model-dir", type=str, required=True)
    parser.add_argument(
        "--zip-out",
        type=str,
        default="",
        help="Optional submit.zip output path (zip root will be exactly model/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_model_dir = Path(args.base_model_dir).resolve()
    weights_dir = Path(args.weights_dir).resolve()
    out_model_dir = Path(args.out_model_dir).resolve()

    summary = pack_model_dir(base_model_dir, weights_dir, out_model_dir)
    print("[PASS] model directory packed")
    print(f"  out_model_dir: {summary['out_model_dir']}")
    print(f"  removed_old_weight_files: {summary['removed_old_weight_files']}")
    print(f"  copied_weight_files: {summary['copied_weight_files']}")
    print(f"  copied_config_files: {summary['copied_config_files']}")
    warnings = summary.get("warnings") or []
    for warning in warnings:
        print(f"  [WARN] {warning}")

    if args.zip_out:
        created = create_submit_zip(out_model_dir, Path(args.zip_out).resolve())
        print(f"  submit_zip: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
