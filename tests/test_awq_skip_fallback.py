from __future__ import annotations

import argparse

import model.quantize as quantize


def test_awq_skip_falls_back_to_gptq(monkeypatch) -> None:
    calls = {"gptq": 0, "awq": 0}

    def fake_awq(*args, **kwargs):
        _ = args, kwargs
        calls["awq"] += 1
        raise RuntimeError("AWQ unavailable")

    def fake_gptq(*args, **kwargs):
        _ = args, kwargs
        calls["gptq"] += 1
        return {"quant_method": "gptq"}

    monkeypatch.setattr(quantize, "run_awq_quantization", fake_awq)
    monkeypatch.setattr(quantize, "run_gptq_quantization", fake_gptq)

    args = argparse.Namespace(
        quant_method="awq",
        awq_allow_skip=True,
        awq_fallback_to_gptq=True,
        scheme="W4A16",
        max_seq_length=1024,
        num_calibration_samples=1024,
    )
    out = quantize.quantize_with_fallback(
        model=object(),
        calib_dataset=object(),
        args=args,
        targets=["Linear"],
        ignore=["embed_tokens", "lm_head"],
    )
    assert calls["awq"] == 1
    assert calls["gptq"] == 1
    assert out["quant_method"] == "gptq"
    assert out["fallback_from"] == "awq"

