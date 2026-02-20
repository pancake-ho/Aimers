#!/usr/bin/env python3
"""Validate submit.zip structure and model compatibility guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .submit_guardrails import ValidationResult, validate_submit_zip
except ImportError:  # pragma: no cover
    from submit_guardrails import ValidationResult, validate_submit_zip


def _print_human(result: ValidationResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] submit.zip validation")
    if result.details.get("zip_top_level") is not None:
        print(f"  zip top-level: {result.details['zip_top_level']}")
    if result.warnings:
        print("  warnings:")
        for item in result.warnings:
            print(f"    - {item}")
    if result.errors:
        print("  errors:")
        for item in result.errors:
            print(f"    - {item}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Aimers Phase2 submit.zip")
    parser.add_argument("--zip", type=str, required=True, help="Path to submit.zip")
    parser.add_argument("--json", action="store_true", help="Print JSON summary only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_submit_zip(Path(args.zip).resolve())
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print_human(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
