#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aimers EXAONE 경량화 해커톤 제출물(submit.zip) 로컬 검증 스크립트

목표:
1) submit.zip 구조가 평가 스크립트가 기대하는 형태인지 검증
2) 오프라인(인터넷 없이) tokenizer/config/model 로딩 가능한지 검증
3) apply_chat_template 기반 chat 추론이 가능한지 검증
4) vLLM로 실제 로딩/추론 스모크 테스트 + 대략적인 throughput(토큰/초) 측정
5) "제출하면 터질만한" 대표 원인을 빠르게 에러 메시지로 노출

사용 예:
  python test.py --zip ./submit.zip --mode full
  python test.py --zip ./submit.zip --mode package
  python test.py --model-dir ./model --mode smoke   # 이미 model 폴더가 있을 때

권장:
- 평가 환경과 vLLM/torch/transformers 버전을 맞추면 가장 좋음.
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from utils import load_tokenizer


# ---------------------------
# Pretty printing helpers
# ---------------------------
def h1(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def ok(msg: str):
    print(f"[OK] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def fail(msg: str):
    print(f"[FAIL] {msg}")

def info(msg: str):
    print(f"[INFO] {msg}")


# ---------------------------
# Offline / env helpers
# ---------------------------
def force_offline():
    # 평가 환경은 보통 인터넷이 막혀있다고 가정하는 게 안전함
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def print_versions():
    h1("환경 버전 점검")
    try:
        import torch
        info(f"python: {sys.version.split()[0]}")
        info(f"torch: {torch.__version__}")
        info(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            info(f"cuda device: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        warn(f"torch 버전 확인 실패: {e}")

    try:
        import transformers
        info(f"transformers: {transformers.__version__}")
    except Exception as e:
        warn(f"transformers 버전 확인 실패: {e}")

    try:
        import vllm
        info(f"vllm: {vllm.__version__}")
    except Exception as e:
        warn(f"vllm 버전 확인 실패/미설치: {e}")
# ---------------------------
# Zip / folder validation
# ---------------------------
@dataclass
class ExtractResult:
    work_dir: Path
    model_dir: Path

def _zip_top_level_items(zipf: zipfile.ZipFile) -> List[str]:
    """zip 내부 최상위 항목(폴더/파일) 리스트"""
    top = set()
    for name in zipf.namelist():
        if not name or name.startswith("__MACOSX/"):
            continue
        parts = name.split("/")
        if parts:
            top.add(parts[0])
    return sorted(top)

def extract_submit_zip(zip_path: Path, out_dir: Path) -> ExtractResult:
    """
    submit.zip을 out_dir에 풀고, model_dir 경로를 찾아 반환.
    기대 형태(권장): zip 최상위에 "model/" 폴더가 존재.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"zip 파일이 없습니다: {zip_path}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        top_items = _zip_top_level_items(z)
        info(f"zip 최상위 항목: {top_items}")

        # 압축 해제
        z.extractall(out_dir)

    # 1순위: out_dir/model
    model_dir = out_dir / "model"
    if model_dir.exists() and model_dir.is_dir():
        ok(f"zip 최상위에 model/ 폴더 확인: {model_dir}")
        return ExtractResult(work_dir=out_dir, model_dir=model_dir)

    # 2순위: 단일 상위폴더 밑에 model이 있는 경우 (예: submit/model)
    subdirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(subdirs) == 1:
        candidate = subdirs[0] / "model"
        if candidate.exists() and candidate.is_dir():
            warn(f"중첩 폴더 감지: {subdirs[0].name}/model 구조")
            warn("평가에서 zip 최상위에 model/을 기대할 수 있으니, 최상위에 model/이 오도록 다시 zip 권장")
            return ExtractResult(work_dir=out_dir, model_dir=candidate)

    # 3순위: config.json이 있는 폴더를 model_dir로 간주
    candidates = list(out_dir.rglob("config.json"))
    if candidates:
        guess = candidates[0].parent
        warn(f"model/ 폴더를 못 찾았지만 config.json 발견 -> model_dir 추정: {guess}")
        warn("이 구조는 평가에서 '설치 에러' 또는 '제출 에러'를 유발할 가능성이 큼 (권장 구조로 재압축 필요)")
        return ExtractResult(work_dir=out_dir, model_dir=guess)

    raise RuntimeError("압축 해제는 됐지만 model_dir를 찾지 못했습니다. zip 구조를 다시 확인하세요.")


def required_files_check(model_dir: Path) -> None:
    h1("필수 파일/구조 점검")

    must_have_any = [
        # 모델 설정
        ("config.json",),
    ]
    for group in must_have_any:
        found = any((model_dir / f).exists() for f in group)
        if not found:
            raise FileNotFoundError(f"필수 파일이 없습니다: {group} (model_dir={model_dir})")
    ok("config.json 존재")

    # 가중치 파일: safetensors 또는 bin 계열 중 하나는 있어야 함
    weight_patterns = ["*.safetensors", "pytorch_model*.bin", "model*.pt"]
    weight_files = []
    for pat in weight_patterns:
        weight_files += list(model_dir.glob(pat))
    if not weight_files:
        raise FileNotFoundError(
            "가중치 파일을 찾지 못했습니다. "
            f"({', '.join(weight_patterns)}) 중 최소 1개 필요. model_dir={model_dir}"
        )
    ok(f"가중치 파일 {len(weight_files)}개 발견 (예: {weight_files[0].name})")

    # 토크나이저 파일은 형태가 다양하므로, 최소한 하나라도 있으면 OK로 보되 미존재 시 경고
    tokenizer_candidates = [
        "tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt",
        "tokenizer_config.json", "special_tokens_map.json"
    ]
    tok_found = [f for f in tokenizer_candidates if (model_dir / f).exists()]
    if not tok_found:
        warn("토크나이저 관련 파일을 거의 못 찾았습니다. transformers/vLLM 로딩이 실패할 수 있습니다.")
    else:
        ok(f"토크나이저 관련 파일 일부 확인: {tok_found}")

    # 용량(경량화 관점)
    total = 0
    for p in model_dir.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    info(f"model_dir 총 파일 크기(대략): {total/1024/1024:.2f} MiB")


def load_and_sanity_check_config(model_dir: Path) -> Dict:
    h1("config.json 로드/기본 sanity check")
    cfg_path = model_dir / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"config.json 파싱 실패(UTF-8/JSON 형태 확인 필요): {e}")

    # 자주 터지는 케이스: 숫자가 문자열, NaN/Infinity 등
    def _scan(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _scan(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]")
        else:
            if isinstance(obj, float):
                if obj != obj:
                    raise ValueError(f"config.json에 NaN 존재: {path}")
                if obj == float("inf") or obj == float("-inf"):
                    raise ValueError(f"config.json에 Infinity 존재: {path}")

    _scan(cfg)
    ok("config.json JSON 형태/NaN/Inf 검사 통과")

    # 최소 키 점검(너무 빡세게 강제하진 않되, 없으면 위험한 것들만 경고)
    for k in ["model_type", "architectures"]:
        if k not in cfg:
            warn(f"config.json에 '{k}' 키가 없습니다. (평가/로딩 스크립트에서 ValidationError 원인이 될 수 있음)")
        else:
            ok(f"config.json 키 확인: {k}")

    # auto_map 기반 trust_remote_code 사용 시: 참조하는 .py 파일이 로컬에 존재하는지 체크
    # (인터넷 막힌 환경에서 매우 중요)
    auto_map = cfg.get("auto_map")
    if isinstance(auto_map, dict):
        missing_py = []
        for _, v in auto_map.items():
            # 예: "exaone.modeling_exaone.EXAONEForCausalLM" -> modeling_exaone.py 필요 가능
            if isinstance(v, str):
                # module path 추정: 마지막 클래스 제외
                module = ".".join(v.split(".")[:-1])
                py_name = module.split(".")[-1] + ".py" if module else None
                if py_name and not (model_dir / py_name).exists():
                    missing_py.append(py_name)
        if missing_py:
            warn("trust_remote_code(auto_map) 참조 파일이 model_dir에 없을 수 있습니다:")
            warn("  - " + ", ".join(sorted(set(missing_py))))
            warn("이 경우 평가(오프라인)에서 모델 로딩이 실패할 가능성이 큽니다. 관련 .py를 model 폴더에 포함시키세요.")
        else:
            ok("auto_map 참조 .py 파일(추정) 존재 여부: 큰 문제 없음")
    else:
        info("auto_map 없음/비정상 -> 표준 아키텍처면 괜찮지만, 커스텀 모델이면 오프라인 로딩 실패 위험")

    # 컨텍스트 길이 관련 힌트(평가에서 긴 생성이 있을 수 있음)
    ctx_keys = ["max_position_embeddings", "model_max_length"]
    for ck in ctx_keys:
        if ck in cfg:
            info(f"{ck}: {cfg.get(ck)}")

    return cfg


def tokenizer_chat_template_check(model_dir: Path) -> None:
    h1("토크나이저 로드 + apply_chat_template 체크(중요)")
    try:
        tok = load_tokenizer(
            str(model_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        ok("AutoTokenizer 로드 성공 (오프라인)")
    except Exception as e:
        raise RuntimeError(
            "AutoTokenizer 로드 실패. (제출 환경에서 흔히 그대로 실패)\n"
            f"- 원인: tokenizer 파일 누락/깨짐, trust_remote_code 필요 파일 누락, 잘못된 config 등\n"
            f"- 상세: {e}"
        )

    messages = [{"role": "user", "content": "짧게: 파이썬 리스트 정렬 예시 코드 보여줘."}]
    try:
        # apply_chat_template = true 환경을 직접 재현
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        ok("tokenizer.apply_chat_template 성공 (apply_chat_template=true 대응 가능)")
        info(f"prompt preview: {repr(prompt[:200])} ...")
    except Exception as e:
        raise RuntimeError(
            "apply_chat_template 실패 -> 평가에서 chat 추론이 바로 터질 가능성이 큼\n"
            "- 해결: tokenizer에 chat_template를 제공해야 합니다.\n"
            "  1) tokenizer_config.json에 'chat_template' 필드 추가 또는\n"
            "  2) 해당 모델이 요구하는 chat_template 파일/설정 포함\n"
            f"- 상세: {e}"
        )


# ---------------------------
# vLLM smoke / benchmark
# ---------------------------
def vllm_smoke_test(
    model_dir: Path,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> None:
    h1("vLLM 로딩/스모크 테스트 (평가 핵심 경로)")
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vllm import 실패. 로컬에서 vLLM로 평가 경로를 재현하기 어렵습니다.\n"
            f"상세: {e}"
        )

    # 평가 환경처럼 최대한 인터넷 없이
    force_offline()

    # (중요) vLLM 로딩 자체가 제출에서 가장 자주 터지는 포인트
    try:
        t0 = time.perf_counter()
        llm = LLM(
            model=str(model_dir),
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
            trust_remote_code=True,
            # enforce_eager는 환경에 따라 필요/불필요. 기본은 False 권장.
        )
        t1 = time.perf_counter()
        ok(f"vLLM LLM 로드 성공 (load time: {t1 - t0:.2f}s)")
    except Exception as e:
        raise RuntimeError(
            "vLLM 모델 로딩 실패(제출에서 그대로 실패 가능)\n"
            "- 흔한 원인: config.json 불일치, 커스텀 코드(.py) 누락, 가중치/토크나이저 파일 누락, OOM\n"
            f"- 상세: {e}"
        )

    prompts = [
        [{"role": "user", "content": "인공지능 모델 경량화에서 양자화가 왜 중요한지 5문장으로 설명해줘."}],
        [{"role": "user", "content": "다음 문제를 파이썬으로 풀어줘: 정수 배열에서 최대 부분합(카데인 알고리즘) 구현."}],
        [{"role": "user", "content": "설명만: LoRA를 베이스 모델에 merge한다는 게 무슨 뜻인지 쉽게 알려줘."}],
    ]

    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=None,
    )

    # 실제 추론 + 토큰/초 측정(대략)
    try:
        t0 = time.perf_counter()
        outputs = llm.chat(messages=prompts, sampling_params=sp)
        t1 = time.perf_counter()
    except Exception as e:
        raise RuntimeError(
            "vLLM chat 추론 실패(제출에서 평가 데이터 돌리다 터질 가능성 큼)\n"
            f"- 상세: {e}"
        )

    # 토큰 수 추정: vLLM 결과 구조가 버전마다 조금 달라서 방어적으로 처리
    total_gen_tokens = 0
    for out in outputs:
        try:
            total_gen_tokens += len(out.outputs[0].token_ids)
        except Exception:
            # token_ids 없으면 텍스트 길이 기반으로 대충만 표시
            pass

    dt = t1 - t0
    ok(f"추론 성공 (batch={len(prompts)}, elapsed={dt:.2f}s)")
    if total_gen_tokens > 0 and dt > 0:
        info(f"대략 생성 throughput: {total_gen_tokens/dt:.2f} tokens/s (참고용)")

    # 결과 미리보기
    print("\n" + "-" * 80)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text if out.outputs else ""
        print(f"[{i}] preview: {text.strip()[:200]} ...")
    print("-" * 80)


# ---------------------------
# Main
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, default=None, help="submit.zip 경로")
    p.add_argument("--model-dir", type=str, default=None, help="이미 풀린 model/ 경로")
    p.add_argument("--work-dir", type=str, default="./_submit_extract", help="zip 해제 작업 폴더")
    p.add_argument("--mode", type=str, default="full",
                   choices=["package", "config", "tokenizer", "smoke", "full"],
                   help="검증 모드")

    # vLLM 옵션(평가 고정값이 있을 수 있으므로 기본값을 평가에 맞춰 둠)
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    p.add_argument("--gpu-mem", type=float, default=0.85, help="gpu_memory_utilization")
    p.add_argument("--max-tokens", type=int, default=256, help="스모크 테스트 생성 토큰 수(로컬 테스트용)")
    p.add_argument("--temperature", type=float, default=0.0, help="스모크 테스트 temperature")

    return p.parse_args()


def main():
    args = parse_args()
    force_offline()
    print_versions()

    if args.model_dir is None and args.zip is None:
        fail("둘 중 하나는 필요합니다: --zip submit.zip 또는 --model-dir ./model")
        sys.exit(2)

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        if not model_dir.exists():
            fail(f"--model-dir 경로가 없습니다: {model_dir}")
            sys.exit(2)
        work_dir = model_dir.parent
    else:
        zip_path = Path(args.zip).resolve()
        work_dir = Path(args.work_dir).resolve()
        res = extract_submit_zip(zip_path, work_dir)
        model_dir = res.model_dir

    info(f"model_dir = {model_dir}")

    # 1) package
    if args.mode in ["package", "full"]:
        required_files_check(model_dir)

    # 2) config
    if args.mode in ["config", "full"]:
        load_and_sanity_check_config(model_dir)

    # 3) tokenizer + chat template
    if args.mode in ["tokenizer", "full"]:
        tokenizer_chat_template_check(model_dir)

    # 4) vLLM smoke
    if args.mode in ["smoke", "full"]:
        vllm_smoke_test(
            model_dir=model_dir,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    ok("모든 선택된 검증 단계를 통과했습니다.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        fail(str(e))
        sys.exit(1)
