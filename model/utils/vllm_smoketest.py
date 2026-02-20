from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict


def run_vllm_smoke(
    model_dir: str | Path,
    max_new_tokens: int = 32,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
) -> Dict[str, Any]:
    target = Path(model_dir).resolve()
    result: Dict[str, Any] = {
        "ok": False,
        "model_dir": str(target),
        "generated_tokens": 0,
        "elapsed_sec": None,
        "tokens_per_sec": None,
        "error": "",
    }

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except Exception as exc:
        result["error"] = f"dependency missing: {exc}"
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(target), trust_remote_code=True)
        messages = [{"role": "user", "content": "한 문장으로 답변해줘: KD와 양자화의 핵심은?"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        llm = LLM(
            model=str(target),
            tensor_parallel_size=int(tensor_parallel_size),
            gpu_memory_utilization=float(gpu_memory_utilization),
            trust_remote_code=True,
            dtype="auto",
        )
        sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_new_tokens))
        started = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params=sampling)
        elapsed = max(1e-9, time.perf_counter() - started)

        generated = 0
        for out in outputs:
            if out.outputs:
                generated += len(out.outputs[0].token_ids)

        result.update(
            {
                "ok": True,
                "generated_tokens": int(generated),
                "elapsed_sec": float(elapsed),
                "tokens_per_sec": float(generated / elapsed) if generated > 0 else 0.0,
            }
        )
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result

