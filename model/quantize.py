from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from .utils.size_report import gather_safetensor_size, print_size_report, write_size_report
    from .utils.vllm_smoketest import run_vllm_smoke
except ImportError:
    from utils.chat_preprocess import normalize_conversations, render_chat_prompt
    from utils.size_report import gather_safetensor_size, print_size_report, write_size_report
    from utils.vllm_smoketest import run_vllm_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize distilled model with GPTQ/AWQ and package output.")
    parser.add_argument("--input_model_dir", type=str, default="./distilled_merged")
    parser.add_argument("--out_dir", type=str, default="./model")
    parser.add_argument("--quant_method", type=str, choices=["gptq", "awq"], default="gptq")
    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--num_calibration_samples", type=int, default=1024)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--scheme", type=str, default="W4A16")
    parser.add_argument("--targets", type=str, default="Linear")
    parser.add_argument("--ignore", type=str, default="embed_tokens,lm_head")
    parser.add_argument("--zip_name", type=str, default="submit")
    parser.add_argument("--run_vllm_smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_vllm_smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_allow_skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--awq_fallback_to_gptq", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_csv(text: str) -> List[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def ensure_ignore_patterns(ignore_patterns: List[str]) -> List[str]:
    required = ["embed_tokens", "lm_head"]
    output = list(ignore_patterns)
    for token in required:
        if token not in output:
            output.append(token)
    return output


def runtime_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_calibration_dataset(
    tokenizer,
    dataset_id: str,
    dataset_split: str,
    num_calibration_samples: int,
):
    ds = load_dataset(dataset_id, split=f"{dataset_split}[:{int(num_calibration_samples)}]")

    def _pp(example: Dict[str, Any]) -> Dict[str, str]:
        conversations = normalize_conversations(example)
        if conversations:
            text = render_chat_prompt(tokenizer, conversations, add_generation_prompt=True)
        else:
            text = str(example.get("text", "")).strip()
        return {"text": text}

    mapped = ds.map(_pp, remove_columns=ds.column_names)
    mapped = mapped.filter(lambda row: bool(row.get("text")))
    return mapped


def run_gptq_quantization(
    model,
    calib_dataset,
    *,
    scheme: str,
    targets: List[str],
    ignore: List[str],
    max_seq_length: int,
    num_calibration_samples: int,
) -> Dict[str, Any]:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import GPTQModifier

    recipe = [
        GPTQModifier(
            scheme=scheme,
            targets=targets,
            ignore=ignore,
        )
    ]

    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=recipe,
        max_seq_length=int(max_seq_length),
        num_calibration_samples=int(num_calibration_samples),
    )
    return {
        "quant_method": "gptq",
        "scheme": scheme,
        "targets": targets,
        "ignore": ignore,
    }


def resolve_awq_modifier():
    candidates = [
        ("llmcompressor.modifiers.quantization", "AWQModifier"),
        ("llmcompressor.modifiers.awq", "AWQModifier"),
    ]
    last_error = None
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except Exception as exc:
            last_error = exc
    raise ImportError(f"AWQModifier is not available: {last_error}")


def build_awq_kwargs(modifier_cls, targets: List[str], ignore: List[str]) -> Dict[str, Any]:
    names = set(inspect.signature(modifier_cls).parameters.keys())
    kwargs: Dict[str, Any] = {}
    if "targets" in names:
        kwargs["targets"] = targets
    if "ignore" in names:
        kwargs["ignore"] = ignore
    if "scheme" in names:
        kwargs["scheme"] = "W4A16"
    if "group_size" in names:
        kwargs["group_size"] = 128
    if "symmetric" in names:
        kwargs["symmetric"] = False
    if "config_groups" in names:
        kwargs["config_groups"] = {
            "group_0": {
                "targets": targets,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,
                    "strategy": "group",
                    "group_size": 128,
                },
            }
        }
    return kwargs


def run_awq_quantization(
    model,
    calib_dataset,
    *,
    targets: List[str],
    ignore: List[str],
    max_seq_length: int,
    num_calibration_samples: int,
) -> Dict[str, Any]:
    from llmcompressor import oneshot

    modifier_cls = resolve_awq_modifier()
    kwargs = build_awq_kwargs(modifier_cls, targets=targets, ignore=ignore)
    modifier = modifier_cls(**kwargs)
    oneshot(
        model=model,
        dataset=calib_dataset,
        recipe=[modifier],
        max_seq_length=int(max_seq_length),
        num_calibration_samples=int(num_calibration_samples),
    )
    return {
        "quant_method": "awq",
        "targets": targets,
        "ignore": ignore,
        "awq_modifier_kwargs": sorted(kwargs.keys()),
    }


def make_submit_archive(zip_name: str, out_dir: Path) -> Path:
    out_dir = out_dir.resolve()
    try:
        rel = os.path.relpath(str(out_dir), ".")
        zip_path = shutil.make_archive(
            base_name=zip_name,
            format="zip",
            root_dir=".",
            base_dir=rel,
        )
    except Exception:
        zip_path = shutil.make_archive(
            base_name=zip_name,
            format="zip",
            root_dir=str(out_dir.parent),
            base_dir=out_dir.name,
        )
    return Path(zip_path).resolve()


def resolve_input_model_dir(input_model_dir: str) -> Path:
    requested = Path(input_model_dir).resolve()
    if requested.exists():
        return requested
    fallback = Path("./base_model").resolve()
    if fallback.exists():
        print(f"[WARN] input_model_dir missing, fallback to {fallback}")
        return fallback
    raise FileNotFoundError(
        f"input_model_dir not found: {requested}; fallback ./base_model also not found."
    )


def quantize_with_fallback(
    model,
    calib_dataset,
    args: argparse.Namespace,
    targets: List[str],
    ignore: List[str],
) -> Dict[str, Any]:
    if args.quant_method == "gptq":
        return run_gptq_quantization(
            model=model,
            calib_dataset=calib_dataset,
            scheme=args.scheme,
            targets=targets,
            ignore=ignore,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
        )

    try:
        return run_awq_quantization(
            model=model,
            calib_dataset=calib_dataset,
            targets=targets,
            ignore=ignore,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
        )
    except Exception as exc:
        awq_error = str(exc)
        if not args.awq_allow_skip:
            raise
        if args.awq_fallback_to_gptq:
            print(f"[WARN] AWQ not available, fallback to GPTQ: {awq_error}")
            summary = run_gptq_quantization(
                model=model,
                calib_dataset=calib_dataset,
                scheme=args.scheme,
                targets=targets,
                ignore=ignore,
                max_seq_length=args.max_seq_length,
                num_calibration_samples=args.num_calibration_samples,
            )
            summary["fallback_from"] = "awq"
            summary["awq_error"] = awq_error
            return summary
        print(f"[WARN] AWQ skipped without fallback: {awq_error}")
        return {
            "quant_method": "awq_skipped",
            "targets": targets,
            "ignore": ignore,
            "awq_error": awq_error,
        }


def main() -> None:
    args = parse_args()

    input_model_dir = resolve_input_model_dir(args.input_model_dir)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = parse_csv(args.targets)
    ignore = ensure_ignore_patterns(parse_csv(args.ignore))

    dtype = runtime_dtype()
    model_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    tokenizer = AutoTokenizer.from_pretrained(str(input_model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(input_model_dir), **model_kwargs)

    calib_dataset = load_calibration_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        num_calibration_samples=args.num_calibration_samples,
    )

    quant_summary = quantize_with_fallback(
        model=model,
        calib_dataset=calib_dataset,
        args=args,
        targets=targets,
        ignore=ignore,
    )

    model.save_pretrained(str(out_dir), save_compressed=True)
    tokenizer.save_pretrained(str(out_dir))

    zip_path = make_submit_archive(args.zip_name, out_dir)
    size_stats = gather_safetensor_size(out_dir)
    print_size_report(size_stats)
    size_report_path = write_size_report(size_stats, out_dir / "size_report.json")

    smoke_result = {"ok": False, "error": "disabled"}
    if args.run_vllm_smoke:
        smoke_result = run_vllm_smoke(out_dir)
        if not smoke_result.get("ok", False):
            message = f"vLLM smoke failed: {smoke_result.get('error', 'unknown')}"
            if args.strict_vllm_smoke:
                raise RuntimeError(message)
            print(f"[WARN] {message}")

    summary = {
        "input_model_dir": str(input_model_dir),
        "out_dir": str(out_dir),
        "zip_path": str(zip_path),
        "num_calibration_samples": int(args.num_calibration_samples),
        "max_seq_length": int(args.max_seq_length),
        "targets": targets,
        "ignore": ignore,
        "quant_summary": quant_summary,
        "size_report_path": str(size_report_path),
        "size_total_bytes": int(size_stats["total_bytes"]),
        "size_total_mib": float(size_stats["total_mib"]),
        "smoke_result": smoke_result,
    }
    summary_path = out_dir / "quantize_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] quantized model saved: {out_dir}")
    print(f"[INFO] zip created: {zip_path}")
    print(f"[INFO] summary written: {summary_path}")


if __name__ == "__main__":
    main()
