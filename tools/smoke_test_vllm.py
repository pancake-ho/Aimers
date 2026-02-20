#!/usr/bin/env python3
"""Offline HF + vLLM smoke test for Aimers Phase2 submission artifacts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from utils import load_tokenizer


def force_offline() -> None:
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _safe_generated_tokens(outputs: Any) -> int:
    total = 0
    try:
        for out in outputs:
            total += len(out.outputs[0].token_ids)
    except Exception:
        total = 0
    return int(total)


def run_smoke_test(
    model_dir: Path,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_new_tokens: int = 32,
) -> Dict[str, Any]:
    force_offline()

    import torch
    from transformers import AutoModelForCausalLM
    from vllm import LLM, SamplingParams

    result: Dict[str, Any] = {
        "model_dir": str(model_dir),
        "hf_load_success": False,
        "vllm_load_success": False,
        "generated_tokens": 0,
        "elapsed_sec": None,
        "tokens_per_sec": None,
        "peak_memory_bytes": None,
    }

    tokenizer = load_tokenizer(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    _ = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    result["hf_load_success"] = True

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    llm = LLM(
        model=str(model_dir),
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        trust_remote_code=True,
        dtype="auto",
    )
    result["vllm_load_success"] = True

    prompts = [[{"role": "user", "content": "간단히 한 문장으로 답해줘: 모델 경량화의 핵심은?"}]]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=int(max_new_tokens),
    )
    start = time.perf_counter()
    outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
    elapsed = max(1e-9, time.perf_counter() - start)

    gen_tokens = _safe_generated_tokens(outputs)
    result["generated_tokens"] = gen_tokens
    result["elapsed_sec"] = elapsed
    result["tokens_per_sec"] = (gen_tokens / elapsed) if gen_tokens > 0 else 0.0

    if torch.cuda.is_available():
        result["peak_memory_bytes"] = int(torch.cuda.max_memory_reserved())

    # Touch apply_chat_template path explicitly.
    _ = tokenizer.apply_chat_template(prompts[0], tokenize=False, add_generation_prompt=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline vLLM smoke test")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model-dir not found: {model_dir}")

    result = run_smoke_test(
        model_dir=model_dir,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        max_new_tokens=args.max_new_tokens,
    )
    print("[PASS] smoke test")
    print(f"  hf_load_success: {result['hf_load_success']}")
    print(f"  vllm_load_success: {result['vllm_load_success']}")
    print(f"  generated_tokens: {result['generated_tokens']}")
    print(f"  elapsed_sec: {result['elapsed_sec']:.4f}")
    print(f"  tokens_per_sec: {result['tokens_per_sec']:.4f}")
    peak = result["peak_memory_bytes"]
    print(f"  peak_memory_bytes: {peak if peak is not None else 'N/A'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
