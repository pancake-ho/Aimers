from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import zipfile

import torch

DEFAULT_BASE_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"
DEFAULT_TARGET_MODEL = str(
    Path(__file__).resolve().parent /
    "submit_lm_w8a8_spanish_newdata.zip"
)

# 이중 언어 및 추론 부하를 실제 평가에 맞게 재구성한 프롬프트
SPEED_PROMPTS = [
    "양자화(Quantization) 기법 중 PTQ와 QAT의 차이점을 상세히 설명해 주세요. 특히 4-bit 양자화 시 발생하는 오차(Quantization Error)를 어떻게 보정하는지 수식적 원리를 포함하여 서술하시오.",
    "Write a comprehensive Python script that implements an asynchronous web scraper. Ensure that the code handles rate limiting, timeouts, and concurrent connections elegantly.",
    "최근 대형 언어 모델(LLM)을 On-Device 환경인 스마트폰에 배포하려 합니다. 제한된 메모리 대역폭(Memory Bandwidth)과 VRAM 용량 하에서 추론 지연 시간(Latency)을 최소화하기 위한 구체적인 아키텍처 및 소프트웨어 최적화 기법을 5가지 이상 제시하시오.",
    "Explain the tradeoff between Group Size and Model Accuracy in GPTQ quantization. Why does a smaller group size preserve perplexity better, and what is the cost regarding VRAM usage?",
    "Resuelve este problema paso a paso: si una función cuadrática f(x) = 2x^2 - 8x + 6 tiene raíces r1 y r2, calcula r1 + r2, r1 * r2 y el valor mínimo de f(x).",
]

@dataclass
class SpeedResult:
    model_path: str
    tokens_generated: int
    elapsed_seconds: float
    tokens_per_second: float

@dataclass
class PerfResult:
    model_path: str
    task: str
    accuracy: float

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DACON Speed + Accuracy benchmark.")
    parser.add_argument("--mode", choices=["speed", "perf", "both"], default="speed")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--target_model", type=str, default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--eval_task", type=str, default="gsm8k", help="lm_eval task (e.g., gsm8k, kmmlu_direct)")
    parser.add_argument("--eval_limit", type=int, default=256, help="Number of samples to evaluate for speedup testing")
    parser.add_argument("--tp", type=int, default=1) # 대회 고정: 1
    parser.add_argument("--gpu_mem", type=float, default=0.85) # 대회 고정: 0.85
    parser.add_argument("--max_gen_toks", type=int, default=16384) # 대회 고정: 16384
    parser.add_argument("--report_json", type=str, default="benchmark_report.json")
    return parser.parse_args()

def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir(): return False
    if not (path / "config.json").exists(): return False
    weight_patterns = ("*.safetensors", "pytorch_model*.bin", "model*.pt")
    for pattern in weight_patterns:
        if any(path.glob(pattern)): return True
    return False

def resolve_model_ref(model_ref: str, temp_roots: List[Path]) -> str:
    # 기존 zip 해제 로직 (동일)
    candidate = Path(model_ref).expanduser()
    if not candidate.exists():
        if candidate.is_absolute() or model_ref.startswith((".", "~")):
            raise RuntimeError(f"Local model path not found: {candidate}")
        return model_ref

    if candidate.is_dir():
        if not _looks_like_model_dir(candidate):
            raise RuntimeError(f"Directory exists but is not a valid model directory: {candidate}")
        return str(candidate.resolve())

    if candidate.is_file() and candidate.suffix.lower() == ".zip":
        stem_dir = candidate.parent / candidate.stem
        nested_model = stem_dir / "model"
        if _looks_like_model_dir(nested_model): return str(nested_model.resolve())
        
        extract_base = Path(os.environ.get("TEST_SPEED_EXTRACT_DIR", str(candidate.parent))).expanduser().resolve()
        extract_base.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="test_speed_model_", dir=str(extract_base)))
        temp_roots.append(temp_root)
        
        with zipfile.ZipFile(candidate, "r") as zf:
            zf.extractall(temp_root)
            
        direct_model = temp_root / "model"
        if direct_model.is_dir(): return str(direct_model.resolve())
        raise RuntimeError(f"Zip does not contain a readable model directory: {candidate}")
    return str(candidate.resolve())

def benchmark_speed_vllm(
    model_path: str,
    model_name: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_tokens: int,
) -> SpeedResult:
    from vllm import LLM, SamplingParams
    print(f"\n{'=' * 70}\n[SPEED] Benchmarking: {model_name}\n[SPEED] Path: {model_path}\n{'=' * 70}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    # 대회 환경과 완벽히 동일한 샘플링 파라미터 적용
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens, 
        ignore_eos=True,
    )

    print("[INFO] Speed warm-up run...")
    llm.generate(SPEED_PROMPTS[:1], sampling_params=sampling_params, use_tqdm=False)

    print("[INFO] Speed benchmark start...")
    started = time.perf_counter()
    outputs = llm.generate(SPEED_PROMPTS, sampling_params=sampling_params, use_tqdm=False)
    elapsed = max(1e-9, time.perf_counter() - started)

    generated_tokens = sum(int(len(out.outputs[0].token_ids)) for out in outputs if out.outputs)
    tokens_per_second = float(generated_tokens / elapsed) if generated_tokens > 0 else 0.0

    print(f"[RESULT][SPEED]\n- generated_tokens: {generated_tokens}\n- elapsed_seconds : {elapsed:.4f}\n- tokens_per_sec  : {tokens_per_second:.2f}")

    from vllm.distributed.parallel_state import destroy_model_parallel
    destroy_model_parallel()
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()

    return SpeedResult(model_path, int(generated_tokens), float(elapsed), float(tokens_per_second))


def benchmark_perf_lmeval(
    model_path: str,
    model_name: str,
    task: str,
    limit: int,
    gpu_mem: float,
    max_gen_toks: int,
) -> PerfResult:
    """
    강의 자료의 bash run_lmeval.sh 방식을 파이썬 subprocess로 구현하여 vLLM 기반 성능 평가
    """
    print(f"\n{'=' * 70}\n[PERF] Benchmarking Accuracy: {model_name} (Task: {task})\n[PERF] Path: {model_path}\n{'=' * 70}")
    
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = Path(f"results_{model_name.lower()}_{task}_{run_tag}")
    model_args = f"pretrained={model_path},gpu_memory_utilization={gpu_mem},max_gen_toks={max_gen_toks},tensor_parallel_size=1"
    
    cmd = [
        "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", task,
        "--limit", str(limit),
        "--output_path", str(output_dir),
        "--apply_chat_template",
        "--batch_size", "auto"
    ]
    
    print(f"[INFO] Executing lm_eval command:\n{' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] lm_eval failed: {e}", file=sys.stderr)
        raise

    # JSON 결과 파싱
    result_files = sorted(output_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not result_files:
        raise RuntimeError(f"lm_eval output JSON not found in: {output_dir}")

    selected_result: Optional[Path] = None
    accuracy: Optional[float] = None
    for result_path in result_files:
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        task_results = data.get("results", {}).get(task, {})
        if not isinstance(task_results, dict):
            continue

        # exact_match(strict) 우선, 없으면 acc 대체
        if "exact_match,strict-match" in task_results or "acc,none" in task_results:
            accuracy = float(task_results.get("exact_match,strict-match", task_results.get("acc,none", 0.0)))
            selected_result = result_path
            break

    if accuracy is None or selected_result is None:
        raise RuntimeError(
            f"lm_eval JSON parsing failed for task='{task}'. scanned={len(result_files)} files under {output_dir}"
        )

    print(f"[INFO] Parsed lm_eval JSON: {selected_result}")
    
    print(f"[RESULT][PERF]\n- Task: {task}\n- Accuracy: {accuracy:.4f}")
    return PerfResult(model_path, task, accuracy)


def compute_official_score(speed: Optional[Dict[str, Any]], perf: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    delta: Dict[str, Optional[float]] = {"SpeedNorm": None, "PerfNorm": None, "TotalScore": None}
    
    if speed is not None:
        base_tps = speed["base"]["tokens_per_second"]
        target_tps = speed["target"]["tokens_per_second"]
        if target_tps > 0:
            # 공식 문서 산식 완벽 구현: 1 - ( (Target Time/Token) / (Base Time/Token) )
            # = 1 - ( (1 / Target TPS) / (1 / Base TPS) ) = 1 - (Base TPS / Target TPS)
            delta["SpeedNorm"] = float(1.0 - (base_tps / target_tps))
            
    if perf is not None:
        base_acc = perf["base"]["accuracy"]
        target_acc = perf["target"]["accuracy"]
        if base_acc > 0:
            # 공식 문서 산식 완벽 구현: Perf_model / Perf_base_model
            delta["PerfNorm"] = float(target_acc / base_acc)

    if delta["SpeedNorm"] is not None and delta["PerfNorm"] is not None:
        # 공식 리더보드 점수 산식
        score = max(0.5 * delta["PerfNorm"] + 0.5 * delta["SpeedNorm"], 0.0)
        delta["TotalScore"] = float(score)

    return delta

def main() -> int:
    args = parse_args()
    speed_payload, perf_payload = None, None
    temp_roots: List[Path] = []


    try:
        print(f"모델 {args.target_model} 사용")
        print(f"모드 {args.mode} 사용")
        resolved_base = resolve_model_ref(args.base_model, temp_roots)
        resolved_target = resolve_model_ref(args.target_model, temp_roots)

        if args.mode in ("speed", "both"):
            base_speed = benchmark_speed_vllm(resolved_base, "Base", tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem, max_tokens=args.max_gen_toks)
            target_speed = benchmark_speed_vllm(resolved_target, "Target", tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem, max_tokens=args.max_gen_toks)
            speed_payload = {"base": asdict(base_speed), "target": asdict(target_speed)}

        if args.mode in ("perf", "both"):
            base_perf = benchmark_perf_lmeval(resolved_base, "Base", args.eval_task, args.eval_limit, args.gpu_mem, args.max_gen_toks)
            target_perf = benchmark_perf_lmeval(resolved_target, "Target", args.eval_task, args.eval_limit, args.gpu_mem, args.max_gen_toks)
            perf_payload = {"base": asdict(base_perf), "target": asdict(target_perf)}

        delta = compute_official_score(speed=speed_payload, perf=perf_payload)
        
        print(f"\n{'*' * 70}\n[OFFICIAL DACON SUMMARY]")
        if speed_payload: print(f"- SpeedNorm : {delta.get('SpeedNorm', 0.0):.4f} (Base TPS: {speed_payload['base']['tokens_per_second']:.2f} -> Target TPS: {speed_payload['target']['tokens_per_second']:.2f})")
        if perf_payload:  print(f"- PerfNorm  : {delta.get('PerfNorm', 0.0):.4f} (Base Acc: {perf_payload['base']['accuracy']:.4f} -> Target Acc: {perf_payload['target']['accuracy']:.4f})")
        if "TotalScore" in delta: print(f"- Total Score: {delta['TotalScore']:.4f}")
        print(f"{'*' * 70}")

        # JSON 리포트 저장 로직 생략 (기존과 동일)
        return 0
    finally:
        for root in temp_roots: shutil.rmtree(root, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
