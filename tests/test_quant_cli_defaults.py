from __future__ import annotations

import sys

from model.quantize import ensure_ignore_patterns, parse_args, parse_csv


def test_quant_cli_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["quantize.py"])
    args = parse_args()
    assert args.quant_method == "gptq"
    assert args.num_calibration_samples == 1024
    assert args.max_seq_length == 1024
    assert args.scheme == "W4A16"
    ignore = ensure_ignore_patterns(parse_csv(args.ignore))
    assert "embed_tokens" in ignore
    assert "lm_head" in ignore

