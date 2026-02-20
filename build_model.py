#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import inspect
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "model"
for _path in (REPO_ROOT, MODEL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from dataset import make_calib_dataset, prepare_dataset
from quantizing.gptq_grid import (
    build_deterministic_gptq_trials,
    parse_float_csv,
    parse_int_csv,
)
from tuning import (
    KDTrainer,
    DataCollatorForCausalLM,
    Fine_tuning,
    build_kd_features,
    tokenizers_compatible,
)
from tuning.lora import LoRALinear
from tools.pack_model_dir import create_submit_zip, pack_model_dir
from tools.smoke_test_vllm import run_smoke_test
from tools.submit_guardrails import validate_model_dir, validate_submit_zip


@dataclass
class StageMetric:
    stage: str
    trial: int
    perplexity: Optional[float]
    hf_tokens_per_sec: Optional[float]
    vllm_tokens_per_sec: Optional[float]
    model_size_mb: Optional[float]
    notes: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantTrialRecord:
    trial: int
    group_size: int
    dampening: float
    block_size: int
    calib_samples: int
    calib_seq_len: int
    ignore_patterns: List[str]
    perplexity: Optional[float]
    hf_tokens_per_sec: Optional[float]
    vllm_tokens_per_sec: Optional[float]
    weighted_proxy: Optional[float]
    selected: bool
    weights_dir: str
    model_dir: str
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aimers Phase2 build pipeline: KD -> LoRA(merge) -> GPTQ -> pack -> guardrails -> smoke"
    )
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--base-model-dir", dest="base_model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")

    parser.add_argument("--do_kd", action="store_true")
    parser.add_argument("--kd_out", type=str, default="kd")
    parser.add_argument("--kd_samples", type=int, default=50_000)
    parser.add_argument("--kd_max_len", type=int, default=1024)
    parser.add_argument("--kd_lr", type=float, default=5e-5)
    parser.add_argument("--kd_steps", type=int, default=1200)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--disable_kd_grid", action="store_true")

    parser.add_argument("--do_lora", action="store_true")
    parser.add_argument("--lora_out", type=str, default="lora_merged")
    parser.add_argument("--lora_samples", type=int, default=2000)
    parser.add_argument("--lora_r", type=int, choices=[8, 16], default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_epochs", type=float, default=1.0)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    parser.add_argument("--quant_method", type=str, choices=["gptq"], default="gptq")
    parser.add_argument("--calib_samples", type=int, default=1024)
    parser.add_argument("--gptq_block_size", type=int, default=128)
    parser.add_argument("--gptq_group_sizes", type=str, default="64,128")
    parser.add_argument("--gptq_dampening_values", type=str, default="0.005,0.010,0.020")
    parser.add_argument("--gptq_calib_seq_len", type=int, choices=[1024, 2048], default=1024)
    parser.add_argument("--gptq_ignore_first_n_blocks", type=int, default=0)
    parser.add_argument("--gptq_ignore_last_n_blocks", type=int, default=0)
    parser.add_argument("--disable_quant_grid", action="store_true")
    parser.add_argument("--search_budget", type=int, default=6)

    parser.add_argument("--eval_dataset_id", type=str, default="")
    parser.add_argument("--eval_dataset_split", type=str, default="train")
    parser.add_argument("--eval_start", type=int, default=200_000)
    parser.add_argument("--eval_count", type=int, default=128)
    parser.add_argument("--eval_max_len", type=int, default=1024)
    parser.add_argument("--skip_eval", action="store_true")

    parser.add_argument("--selection_metric", choices=["weighted_proxy", "lb_proxy", "ppl"], default="weighted_proxy")
    parser.add_argument("--score_perf_weight", type=float, default=0.5)
    parser.add_argument("--score_speed_weight", type=float, default=0.5)

    parser.add_argument("--report_path", type=str, default="metrics.csv")
    parser.add_argument("--strict_guardrails", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--smoke_max_new_tokens", type=int, default=32)
    return parser.parse_args()


def _from_pretrained_common(args: argparse.Namespace) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return kwargs


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def runtime_torch_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def use_bf16_training() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def cleanup_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_first_param_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


def estimate_model_size_mb(model) -> Optional[float]:
    try:
        total = 0
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                total += int(tensor.numel()) * int(tensor.element_size())
        return float(total / (1024 * 1024))
    except Exception:
        return None


def dir_size_mb(path: Path) -> float:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return float(total / (1024 * 1024))


def append_metric(
    metrics: List[StageMetric],
    stage: str,
    trial: int,
    perplexity: Optional[float],
    hf_tokens_per_sec: Optional[float],
    vllm_tokens_per_sec: Optional[float],
    model_size_mb: Optional[float],
    notes: str,
    params: Optional[Dict[str, Any]] = None,
) -> None:
    metrics.append(
        StageMetric(
            stage=stage,
            trial=trial,
            perplexity=perplexity,
            hf_tokens_per_sec=hf_tokens_per_sec,
            vllm_tokens_per_sec=vllm_tokens_per_sec,
            model_size_mb=model_size_mb,
            notes=notes,
            params=params or {},
        )
    )


def _to_chat_text(tokenizer, example: Dict[str, Any], add_generation_prompt: bool) -> Optional[str]:
    conversations = example.get("conversations")
    if not conversations:
        return None
    try:
        return tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return None


def build_eval_texts(args: argparse.Namespace, tokenizer) -> List[str]:
    if args.skip_eval or args.eval_count <= 0:
        return []
    dataset_id = args.eval_dataset_id or args.dataset_id
    ds = load_dataset(dataset_id, split=args.eval_dataset_split).shuffle(seed=args.seed + 17)
    total = len(ds)
    if total == 0:
        return []

    start = max(0, int(args.eval_start))
    if start >= total:
        start = max(0, total - args.eval_count)
    end = min(total, start + int(args.eval_count))

    selected = ds.select(range(start, end))
    texts: List[str] = []
    for ex in selected:
        text = _to_chat_text(tokenizer, ex, add_generation_prompt=False)
        if text:
            texts.append(text)
    print(
        f"[INFO] eval set dataset={dataset_id} split={args.eval_dataset_split} "
        f"slice=[{start}:{end}) usable={len(texts)}"
    )
    return texts


@torch.no_grad()
def evaluate_model(model, tokenizer, texts: List[str], max_len: int) -> Tuple[Optional[float], Optional[float]]:
    if not texts:
        return None, None
    model.eval()
    device = get_first_param_device(model)
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()

    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        valid_tokens = int(attention_mask.sum().item())
        if valid_tokens < 2:
            continue

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        token_count = valid_tokens - 1
        total_nll += float(out.loss.detach().float().item()) * token_count
        total_tokens += token_count

    elapsed = max(1e-9, time.perf_counter() - start)
    if total_tokens == 0:
        return None, None
    ppl = math.exp(total_nll / total_tokens)
    tps = total_tokens / elapsed
    return float(ppl), float(tps)


def save_model_checkpoint(model, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)

def parse_int_list(csv_text: str) -> List[int]:
    return parse_int_csv(csv_text)


def parse_float_list(csv_text: str) -> List[float]:
    return parse_float_csv(csv_text)


def compute_weighted_proxy(
    base_ppl: Optional[float],
    base_tps: Optional[float],
    ppl: Optional[float],
    tps: Optional[float],
    perf_weight: float,
    speed_weight: float,
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    perf_ratio = None
    if base_ppl is not None and ppl is not None and base_ppl > 0 and ppl > 0:
        perf_ratio = base_ppl / ppl

    tpt_reduction_ratio = None
    if base_tps is not None and tps is not None and base_tps > 0 and tps > 0:
        tpt_reduction_ratio = 1.0 - (base_tps / tps)

    terms: List[Tuple[float, float]] = []
    if perf_ratio is not None:
        terms.append((max(0.0, perf_weight), perf_ratio - 1.0))
    if tpt_reduction_ratio is not None:
        terms.append((max(0.0, speed_weight), tpt_reduction_ratio))

    score = None
    if terms:
        denom = sum(weight for weight, _ in terms)
        if denom <= 0:
            score = sum(value for _, value in terms) / float(len(terms))
        else:
            score = sum(weight * value for weight, value in terms) / denom

    details = {
        "perf_ratio": perf_ratio,
        "perf_gain_ratio": None if perf_ratio is None else (perf_ratio - 1.0),
        "tpt_reduction_ratio": tpt_reduction_ratio,
    }
    return score, details


def build_selection_key(
    selection_metric: str,
    ppl: Optional[float],
    weighted_proxy: Optional[float],
) -> Tuple[Tuple[int, float], str]:
    metric = "weighted_proxy" if selection_metric == "lb_proxy" else selection_metric
    if metric == "ppl":
        if ppl is not None:
            return (2, -ppl), "ppl"
        if weighted_proxy is not None:
            return (1, weighted_proxy), "weighted_proxy_fallback"
        return (0, float("-inf")), "none"

    if weighted_proxy is not None:
        return (2, weighted_proxy), "weighted_proxy"
    if ppl is not None:
        return (1, -ppl), "ppl_fallback"
    return (0, float("-inf")), "none"


def run_vllm_tps(model_dir: Path, max_new_tokens: int) -> Tuple[Optional[float], str]:
    try:
        result = run_smoke_test(
            model_dir=model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_new_tokens=max_new_tokens,
        )
        return float(result["tokens_per_sec"]), ""
    except Exception as exc:
        return None, str(exc)


def build_kd_train_dataset(args: argparse.Namespace, tokenizer) -> Dataset:
    raw = load_dataset(args.dataset_id, split=args.dataset_split)
    raw = raw.shuffle(seed=args.seed).select(range(min(args.kd_samples, len(raw))))

    def _map(ex):
        feat = build_kd_features(tokenizer, ex, args.kd_max_len)
        if feat is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return feat

    mapped = raw.map(_map, remove_columns=raw.column_names)
    mapped = mapped.filter(lambda x: len(x["input_ids"]) > 0)
    return mapped


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def build_kd_trials(args: argparse.Namespace) -> List[Tuple[float, float, int]]:
    if args.disable_kd_grid:
        return [(args.kd_temp, args.kd_alpha, args.kd_steps)]

    temps = sorted({max(1.0, args.kd_temp - 0.5), args.kd_temp, args.kd_temp + 0.5})
    alphas = sorted(
        {
            clamp(args.kd_alpha - 0.1, 0.1, 0.9),
            clamp(args.kd_alpha, 0.1, 0.9),
            clamp(args.kd_alpha + 0.1, 0.1, 0.9),
        }
    )
    steps = sorted({max(300, int(args.kd_steps * 0.7)), int(args.kd_steps)})

    candidates = list(itertools.product(temps, alphas, steps))

    def _dist(candidate: Tuple[float, float, int]) -> float:
        temp, alpha, step = candidate
        return abs(temp - args.kd_temp) + abs(alpha - args.kd_alpha) + abs(step - args.kd_steps) / max(1, args.kd_steps)

    candidates.sort(key=_dist)
    return candidates[: max(1, args.search_budget)]


def run_kd_trial(
    args: argparse.Namespace,
    tokenizer,
    train_ds: Dataset,
    eval_texts: List[str],
    trial_idx: int,
    temperature: float,
    alpha: float,
    steps: int,
    out_dir: Path,
) -> Tuple[str, Optional[float], Optional[float], Optional[float], bool, str]:
    trial_dir = out_dir / f"trial_{trial_idx:02d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    common = _from_pretrained_common(args)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
        **common,
    ).eval()
    student = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
        **common,
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable()

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.base_model, **common)
    compatible, reason = tokenizers_compatible(tokenizer, teacher_tokenizer)
    use_logit_kd = bool(compatible)
    if not use_logit_kd:
        print(f"[WARN] tokenizer mismatch -> disable logit KD and fallback to CE-only ({reason})")

    train_args = TrainingArguments(
        output_dir=str(trial_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=int(steps),
        learning_rate=args.kd_lr,
        warmup_ratio=0.03,
        bf16=use_bf16_training(),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = KDTrainer(
        model=student,
        teacher_model=teacher,
        temperature=float(temperature),
        alpha=float(alpha),
        use_logit_kd=use_logit_kd,
        args=train_args,
        train_dataset=train_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()

    ppl, hf_tps = (None, None)
    if not args.skip_eval:
        ppl, hf_tps = evaluate_model(trainer.model, tokenizer, eval_texts, args.eval_max_len)
    model_size = estimate_model_size_mb(trainer.model)

    save_model_checkpoint(trainer.model, tokenizer, trial_dir)

    del trainer
    del teacher
    cleanup_cuda()
    return str(trial_dir), ppl, hf_tps, model_size, use_logit_kd, reason

def run_kd_if_enabled(
    args: argparse.Namespace,
    tokenizer,
    metrics: List[StageMetric],
    eval_texts: List[str],
    base_ppl: Optional[float],
    base_hf_tps: Optional[float],
    out_dir: Path,
) -> str:
    if not args.do_kd:
        return args.base_model

    kd_root = out_dir / args.kd_out
    train_ds = build_kd_train_dataset(args, tokenizer)
    trials = build_kd_trials(args)

    best_path: Optional[str] = None
    best_key: Optional[Tuple[int, float]] = None

    for idx, (temp, alpha, steps) in enumerate(trials, start=1):
        ckpt, ppl, hf_tps, model_size, use_logit_kd, reason = run_kd_trial(
            args=args,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_texts=eval_texts,
            trial_idx=idx,
            temperature=temp,
            alpha=alpha,
            steps=steps,
            out_dir=kd_root,
        )

        weighted_proxy, details = compute_weighted_proxy(
            base_ppl=base_ppl,
            base_tps=base_hf_tps,
            ppl=ppl,
            tps=hf_tps,
            perf_weight=args.score_perf_weight,
            speed_weight=args.score_speed_weight,
        )
        key, rule = build_selection_key(args.selection_metric, ppl, weighted_proxy)
        note = (
            f"temp={temp}, alpha={alpha}, steps={steps}, logit_kd={use_logit_kd}, "
            f"select={rule}, reason={reason}"
        )
        append_metric(
            metrics,
            stage="kd",
            trial=idx,
            perplexity=ppl,
            hf_tokens_per_sec=hf_tps,
            vllm_tokens_per_sec=None,
            model_size_mb=model_size,
            notes=note,
            params={
                "temp": temp,
                "alpha": alpha,
                "steps": steps,
                "use_logit_kd": use_logit_kd,
                "tokenizer_compatibility": reason,
                "selection_rule": rule,
                "weighted_proxy": weighted_proxy,
                **details,
            },
        )
        if best_path is None or best_key is None or key > best_key:
            best_path = ckpt
            best_key = key

    if best_path is None:
        raise RuntimeError("KD stage produced no valid checkpoint")
    print(f"[INFO] best KD checkpoint: {best_path}")
    return best_path


def parse_target_modules(csv_text: str) -> List[str]:
    modules = [item.strip() for item in csv_text.split(",") if item.strip()]
    if not modules:
        raise ValueError("lora_target_modules cannot be empty")
    return modules


def run_lora_if_enabled(
    args: argparse.Namespace,
    tokenizer,
    model_path: str,
    metrics: List[StageMetric],
    eval_texts: List[str],
    out_dir: Path,
) -> str:
    if not args.do_lora:
        return model_path

    print(f"[INFO] LoRA stage start from: {model_path}")
    common = _from_pretrained_common(args)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
        **common,
    )

    train_ds, _ = prepare_dataset(args.dataset_id, args.dataset_split, args.lora_samples, 0, seed=args.seed)
    target_modules = parse_target_modules(args.lora_target_modules)
    tuner = Fine_tuning(
        model=model,
        tokenizer=tokenizer,
        seq_length=args.eval_max_len,
        train_ds=train_ds,
    )
    tuned_model = tuner.setup_lora(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        epochs=args.lora_epochs,
        lr=args.lora_lr,
        seed=args.seed,
        target_modules=target_modules,
    )

    unmerged = [name for name, module in tuned_model.named_modules() if isinstance(module, LoRALinear)]
    if unmerged:
        raise RuntimeError(f"LoRA merge failed; unresolved LoRALinear modules: {unmerged[:5]}")

    lora_out = out_dir / args.lora_out
    save_model_checkpoint(tuned_model, tokenizer, lora_out)

    ppl, hf_tps = (None, None)
    if not args.skip_eval:
        ppl, hf_tps = evaluate_model(tuned_model, tokenizer, eval_texts, args.eval_max_len)

    append_metric(
        metrics,
        stage="lora",
        trial=1,
        perplexity=ppl,
        hf_tokens_per_sec=hf_tps,
        vllm_tokens_per_sec=None,
        model_size_mb=estimate_model_size_mb(tuned_model),
        notes=f"samples={args.lora_samples}, r={args.lora_r}, merged=True",
        params={
            "samples": args.lora_samples,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "merged": True,
        },
    )

    del tuned_model
    cleanup_cuda()
    print(f"[INFO] LoRA stage done -> {lora_out}")
    return str(lora_out)


def build_gptq_trials(args: argparse.Namespace) -> List[Dict[str, Any]]:
    return build_deterministic_gptq_trials(
        group_sizes_csv=args.gptq_group_sizes,
        dampening_csv=args.gptq_dampening_values,
        block_size=args.gptq_block_size,
        calib_samples=args.calib_samples,
        calib_seq_len=args.gptq_calib_seq_len,
        search_budget=args.search_budget,
        disable_grid=args.disable_quant_grid,
    )


def build_gptq_ignore_patterns(model_config, args: argparse.Namespace) -> List[str]:
    ignore = ["re:.*embed_tokens.*", "re:.*lm_head.*"]
    num_layers = int(getattr(model_config, "num_hidden_layers", 0) or 0)
    first_n = max(0, int(args.gptq_ignore_first_n_blocks))
    last_n = max(0, int(args.gptq_ignore_last_n_blocks))

    for layer_idx in range(min(first_n, num_layers)):
        ignore.append(fr"re:.*layers\.{layer_idx}\..*")
    for layer_idx in range(max(0, num_layers - last_n), num_layers):
        ignore.append(fr"re:.*layers\.{layer_idx}\..*")

    deduped: List[str] = []
    seen = set()
    for item in ignore:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def create_gptq_modifier_adapter(
    block_size: int,
    dampening: float,
    group_size: int,
    ignore_patterns: List[str],
):
    signature = inspect.signature(GPTQModifier.__init__)
    params = set(signature.parameters.keys())
    supports_config_groups = "config_groups" in params
    supports_group_size = "group_size" in params
    if not supports_config_groups and not supports_group_size:
        raise RuntimeError(
            "Installed GPTQModifier API does not expose config_groups or group_size. "
            "Cannot enforce group_size safely."
        )

    base_kwargs = {
        "scheme": "W4A16",
        "targets": ["Linear"],
        "ignore": ignore_patterns,
        "block_size": int(block_size),
        "dampening_frac": float(dampening),
        "actorder": "weight",
        "offload_hessians": False,
    }

    attempts: List[Tuple[str, Dict[str, Any]]] = []
    if supports_config_groups:
        config_groups = {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "strategy": "group",
                    "group_size": int(group_size),
                },
                "input_activations": None,
                "output_activations": None,
            }
        }
        attempts.append(("config_groups", {**base_kwargs, "config_groups": config_groups}))
    if supports_group_size:
        attempts.append(("group_size", {**base_kwargs, "group_size": int(group_size)}))

    errors: List[str] = []
    for mode, kwargs in attempts:
        try:
            modifier = GPTQModifier(**kwargs)
            return modifier, {"mode": mode, "group_size": int(group_size), "kwargs": kwargs}
        except TypeError as exc:
            errors.append(f"{mode}: {exc}")

    raise RuntimeError(
        "GPTQModifier group-size adapter failed. "
        f"Unable to configure group_size={group_size}. Errors: {' | '.join(errors)}"
    )

def run_gptq_quantization(
    args: argparse.Namespace,
    tokenizer,
    model_path: str,
    trial_cfg: Dict[str, Any],
    ignore_patterns: List[str],
):
    print(
        "[INFO] GPTQ start "
        f"(group={trial_cfg['group_size']}, damp={trial_cfg['dampening']}, block={trial_cfg['block_size']})"
    )
    common = _from_pretrained_common(args)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
        **common,
    )

    calib_ds = make_calib_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        split=args.dataset_split,
        n=int(trial_cfg["calib_samples"]),
        seed=args.seed,
    )
    if len(calib_ds) == 0:
        raise RuntimeError("calibration dataset is empty after KD prompt_text filtering")

    modifier, adapter_meta = create_gptq_modifier_adapter(
        block_size=int(trial_cfg["block_size"]),
        dampening=float(trial_cfg["dampening"]),
        group_size=int(trial_cfg["group_size"]),
        ignore_patterns=ignore_patterns,
    )

    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=[modifier],
        max_seq_length=int(trial_cfg["calib_seq_len"]),
        num_calibration_samples=min(int(trial_cfg["calib_samples"]), len(calib_ds)),
        concatenate_data=True,
        pad_to_max_length=False,
        shuffle_calibration_samples=True,
    )
    return model, adapter_meta, len(calib_ds)


def write_gptq_sweep_csv(path: Path, records: List[QuantTrialRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial",
        "group_size",
        "dampening",
        "block_size",
        "calib_samples",
        "calib_seq_len",
        "ignore_patterns",
        "perplexity",
        "hf_tokens_per_sec",
        "vllm_tokens_per_sec",
        "weighted_proxy",
        "selected",
        "weights_dir",
        "model_dir",
        "status",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            data = asdict(row)
            data["ignore_patterns"] = "|".join(row.ignore_patterns)
            writer.writerow(data)


def run_quantization_with_grid(
    args: argparse.Namespace,
    tokenizer,
    model_path: str,
    metrics: List[StageMetric],
    eval_texts: List[str],
    base_ppl: Optional[float],
    base_vllm_tps: Optional[float],
    out_dir: Path,
) -> Tuple[Path, Path, List[QuantTrialRecord], str]:
    trials = build_gptq_trials(args)
    quant_root = out_dir / "quant_trials"
    quant_root.mkdir(parents=True, exist_ok=True)

    records: List[QuantTrialRecord] = []
    best_key: Optional[Tuple[int, float]] = None
    best_weights_dir: Optional[Path] = None
    best_model_dir: Optional[Path] = None
    best_note = ""

    for idx, trial_cfg in enumerate(trials, start=1):
        trial_weights_dir = quant_root / f"trial_{idx:02d}" / "weights"
        trial_model_dir = quant_root / f"trial_{idx:02d}" / "model"

        ppl: Optional[float] = None
        hf_tps: Optional[float] = None
        vllm_tps: Optional[float] = None
        weighted_proxy: Optional[float] = None
        status = "ok"
        error_text = ""
        ignore_patterns: List[str] = []
        adapter_meta: Dict[str, Any] = {}

        try:
            common = _from_pretrained_common(args)
            cfg_probe = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=runtime_torch_dtype(),
                device_map="cpu",
                **common,
            )
            ignore_patterns = build_gptq_ignore_patterns(cfg_probe.config, args)
            del cfg_probe
            cleanup_cuda()

            q_model, adapter_meta, used_calib = run_gptq_quantization(
                args=args,
                tokenizer=tokenizer,
                model_path=model_path,
                trial_cfg=trial_cfg,
                ignore_patterns=ignore_patterns,
            )
            trial_cfg["calib_samples"] = int(used_calib)

            if not args.skip_eval:
                ppl, hf_tps = evaluate_model(q_model, tokenizer, eval_texts, args.eval_max_len)

            save_model_checkpoint(q_model, tokenizer, trial_weights_dir)
            pack_model_dir(
                base_model_dir=Path(args.base_model_dir).resolve(),
                weights_dir=trial_weights_dir,
                out_model_dir=trial_model_dir,
            )

            if not args.skip_smoke:
                vllm_tps, smoke_error = run_vllm_tps(trial_model_dir, args.smoke_max_new_tokens)
                if smoke_error:
                    error_text = smoke_error
                    if args.strict_guardrails:
                        raise RuntimeError(smoke_error)

            weighted_proxy, proxy_details = compute_weighted_proxy(
                base_ppl=base_ppl,
                base_tps=base_vllm_tps,
                ppl=ppl,
                tps=vllm_tps,
                perf_weight=args.score_perf_weight,
                speed_weight=args.score_speed_weight,
            )
            key, rule = build_selection_key(args.selection_metric, ppl, weighted_proxy)

            note = (
                f"group={trial_cfg['group_size']}, damp={trial_cfg['dampening']}, "
                f"block={trial_cfg['block_size']}, select={rule}, adapter={adapter_meta.get('mode')}"
            )
            append_metric(
                metrics,
                stage="quant",
                trial=idx,
                perplexity=ppl,
                hf_tokens_per_sec=hf_tps,
                vllm_tokens_per_sec=vllm_tps,
                model_size_mb=dir_size_mb(trial_weights_dir),
                notes=note,
                params={
                    **trial_cfg,
                    "ignore_patterns": ignore_patterns,
                    "adapter_meta": adapter_meta,
                    "weighted_proxy": weighted_proxy,
                    **proxy_details,
                },
            )

            if best_key is None or key > best_key:
                best_key = key
                best_weights_dir = trial_weights_dir
                best_model_dir = trial_model_dir
                best_note = note

        except Exception as exc:
            status = "failed"
            error_text = str(exc)
            append_metric(
                metrics,
                stage="quant",
                trial=idx,
                perplexity=None,
                hf_tokens_per_sec=None,
                vllm_tokens_per_sec=None,
                model_size_mb=None,
                notes=f"failed: {error_text}",
                params={**trial_cfg, "ignore_patterns": ignore_patterns},
            )
            if args.strict_guardrails:
                raise

        finally:
            try:
                del q_model
            except Exception:
                pass
            cleanup_cuda()

        records.append(
            QuantTrialRecord(
                trial=idx,
                group_size=int(trial_cfg["group_size"]),
                dampening=float(trial_cfg["dampening"]),
                block_size=int(trial_cfg["block_size"]),
                calib_samples=int(trial_cfg["calib_samples"]),
                calib_seq_len=int(trial_cfg["calib_seq_len"]),
                ignore_patterns=list(ignore_patterns),
                perplexity=ppl,
                hf_tokens_per_sec=hf_tps,
                vllm_tokens_per_sec=vllm_tps,
                weighted_proxy=weighted_proxy,
                selected=False,
                weights_dir=str(trial_weights_dir),
                model_dir=str(trial_model_dir),
                status=status,
                error=error_text,
            )
        )

    if best_weights_dir is None or best_model_dir is None:
        raise RuntimeError("all GPTQ trials failed")

    for item in records:
        if Path(item.weights_dir).resolve() == best_weights_dir.resolve():
            item.selected = True

    print(f"[INFO] best quant config: {best_note}")
    return best_weights_dir, best_model_dir, records, best_note


def resolve_report_paths(out_dir: Path, report_path: str) -> Tuple[Path, Path]:
    base = Path(report_path)
    if not base.is_absolute():
        base = out_dir / base
    if base.suffix.lower() == ".jsonl":
        return base.with_suffix(".csv"), base
    csv_path = base if base.suffix.lower() == ".csv" else base.with_suffix(".csv")
    return csv_path, csv_path.with_suffix(".jsonl")


def write_metric_reports(out_dir: Path, report_path: str, metrics: List[StageMetric], summary: Dict[str, Any]) -> Tuple[Path, Path, Path]:
    csv_path, jsonl_path = resolve_report_paths(out_dir, report_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "stage",
                "trial",
                "perplexity",
                "hf_tokens_per_sec",
                "vllm_tokens_per_sec",
                "model_size_mb",
                "notes",
                "params",
            ],
        )
        writer.writeheader()
        for item in metrics:
            row = asdict(item)
            row["params"] = json.dumps(row["params"], ensure_ascii=False)
            writer.writerow(row)

    with jsonl_path.open("w", encoding="utf-8") as fp:
        for item in metrics:
            fp.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary_path = out_dir / "final_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, jsonl_path, summary_path


def print_metric_table(metrics: List[StageMetric]) -> None:
    print("\n=== Metric Summary ===")
    print(
        f"{'Stage':14} | {'Trial':>5} | {'PPL':>10} | {'HF Tok/s':>10} | "
        f"{'vLLM Tok/s':>11} | {'Model(MB)':>10} | Notes"
    )
    print("-" * 120)
    for item in metrics:
        ppl = "N/A" if item.perplexity is None else f"{item.perplexity:.4f}"
        hf_tps = "N/A" if item.hf_tokens_per_sec is None else f"{item.hf_tokens_per_sec:.2f}"
        vl_tps = "N/A" if item.vllm_tokens_per_sec is None else f"{item.vllm_tokens_per_sec:.2f}"
        model_size = "N/A" if item.model_size_mb is None else f"{item.model_size_mb:.2f}"
        print(
            f"{item.stage:14} | {item.trial:5d} | {ppl:>10} | {hf_tps:>10} | "
            f"{vl_tps:>11} | {model_size:>10} | {item.notes}"
        )

def gather_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0], "torch": getattr(torch, "__version__", "unknown")}
    for module_name in ("transformers", "datasets", "vllm", "llmcompressor"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[module_name] = "not_installed"
    return versions


def write_run_configs(args: argparse.Namespace, out_dir: Path) -> Dict[str, str]:
    cfg_dir = out_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    kd_cfg = {
        "enabled": args.do_kd,
        "samples": args.kd_samples,
        "max_len": args.kd_max_len,
        "lr": args.kd_lr,
        "steps": args.kd_steps,
        "temperature": args.kd_temp,
        "alpha": args.kd_alpha,
    }
    lora_cfg = {
        "enabled": args.do_lora,
        "samples": args.lora_samples,
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        "epochs": args.lora_epochs,
        "lr": args.lora_lr,
        "target_modules": parse_target_modules(args.lora_target_modules),
    }
    quant_cfg = {
        "method": args.quant_method,
        "calib_samples": args.calib_samples,
        "block_size": args.gptq_block_size,
        "group_sizes": parse_int_list(args.gptq_group_sizes),
        "dampening_values": parse_float_list(args.gptq_dampening_values),
        "calib_seq_len": args.gptq_calib_seq_len,
        "ignore_first_n_blocks": args.gptq_ignore_first_n_blocks,
        "ignore_last_n_blocks": args.gptq_ignore_last_n_blocks,
    }
    dataset_manifest = {
        "train_dataset_id": args.dataset_id,
        "train_dataset_split": args.dataset_split,
        "eval_dataset_id": args.eval_dataset_id or args.dataset_id,
        "eval_dataset_split": args.eval_dataset_split,
        "eval_start": args.eval_start,
        "eval_count": args.eval_count,
    }

    paths = {
        "kd_config": cfg_dir / "kd_config.json",
        "lora_config": cfg_dir / "lora_config.json",
        "quant_config": cfg_dir / "quant_config.json",
        "dataset_manifest": cfg_dir / "dataset_manifest.json",
        "run_manifest": cfg_dir / "run_manifest.json",
    }

    paths["kd_config"].write_text(json.dumps(kd_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["lora_config"].write_text(json.dumps(lora_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["quant_config"].write_text(json.dumps(quant_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["dataset_manifest"].write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["run_manifest"].write_text(
        json.dumps(
            {
                "seed": args.seed,
                "versions": gather_versions(),
                "base_model": args.base_model,
                "base_model_dir": args.base_model_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def enforce_validation(result, strict_warnings: bool, label: str) -> None:
    if not result.ok:
        raise RuntimeError(f"{label} validation failed: {'; '.join(result.errors)}")
    if strict_warnings and result.warnings:
        raise RuntimeError(f"{label} warnings in strict mode: {'; '.join(result.warnings)}")


def main() -> None:
    args = parse_args()
    args.base_model_dir = str(Path(args.base_model_dir).resolve())
    if not args.base_model:
        args.base_model = args.base_model_dir

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    config_paths = write_run_configs(args, out_dir)

    common = _from_pretrained_common(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **common)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    metrics: List[StageMetric] = []
    eval_texts = build_eval_texts(args, tokenizer)

    base_ppl: Optional[float] = None
    base_hf_tps: Optional[float] = None
    base_vllm_tps: Optional[float] = None

    if not args.skip_eval:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=runtime_torch_dtype(),
            device_map="auto",
            **common,
        )
        base_ppl, base_hf_tps = evaluate_model(base_model, tokenizer, eval_texts, args.eval_max_len)
        append_metric(
            metrics,
            stage="base",
            trial=0,
            perplexity=base_ppl,
            hf_tokens_per_sec=base_hf_tps,
            vllm_tokens_per_sec=None,
            model_size_mb=estimate_model_size_mb(base_model),
            notes=f"eval_count={len(eval_texts)}",
            params={"eval_dataset": args.eval_dataset_id or args.dataset_id},
        )
        del base_model
        cleanup_cuda()

    if not args.skip_smoke:
        base_vllm_tps, base_smoke_error = run_vllm_tps(Path(args.base_model_dir), args.smoke_max_new_tokens)
        if base_smoke_error:
            print(f"[WARN] base model vLLM smoke failed: {base_smoke_error}")
            if args.strict_guardrails:
                raise RuntimeError(base_smoke_error)
        append_metric(
            metrics,
            stage="base_smoke",
            trial=0,
            perplexity=None,
            hf_tokens_per_sec=None,
            vllm_tokens_per_sec=base_vllm_tps,
            model_size_mb=None,
            notes="base model vLLM throughput",
            params={"error": base_smoke_error},
        )

    print("[INFO] Pipeline: KD(optional) -> LoRA(optional, merged) -> GPTQ(6-run) -> pack -> guardrails -> smoke")
    model_path = run_kd_if_enabled(args, tokenizer, metrics, eval_texts, base_ppl, base_hf_tps, out_dir)
    model_path = run_lora_if_enabled(args, tokenizer, model_path, metrics, eval_texts, out_dir)

    best_weights_dir, _, quant_records, best_note = run_quantization_with_grid(
        args=args,
        tokenizer=tokenizer,
        model_path=model_path,
        metrics=metrics,
        eval_texts=eval_texts,
        base_ppl=base_ppl,
        base_vllm_tps=base_vllm_tps,
        out_dir=out_dir,
    )

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    gptq_sweep_csv = reports_dir / "gptq_sweep.csv"
    write_gptq_sweep_csv(gptq_sweep_csv, quant_records)

    final_model_dir = out_dir / "model"
    pack_model_dir(
        base_model_dir=Path(args.base_model_dir),
        weights_dir=best_weights_dir,
        out_model_dir=final_model_dir,
    )
    zip_path = create_submit_zip(final_model_dir, out_dir / "submit.zip")

    model_validation = validate_model_dir(final_model_dir)
    zip_validation = validate_submit_zip(zip_path)
    enforce_validation(model_validation, args.strict_guardrails, "model/")
    enforce_validation(zip_validation, args.strict_guardrails, "submit.zip")

    final_smoke: Dict[str, Any] = {"skipped": args.skip_smoke}
    final_vllm_tps = None
    if not args.skip_smoke:
        smoke_result = run_smoke_test(
            model_dir=final_model_dir,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.85,
            max_new_tokens=args.smoke_max_new_tokens,
        )
        final_smoke = smoke_result
        final_vllm_tps = smoke_result.get("tokens_per_sec")

    selected_trial = next((row for row in quant_records if row.selected), None)
    final_ppl = selected_trial.perplexity if selected_trial else None
    final_hf_tps = selected_trial.hf_tokens_per_sec if selected_trial else None

    final_proxy, final_proxy_details = compute_weighted_proxy(
        base_ppl=base_ppl,
        base_tps=base_vllm_tps,
        ppl=final_ppl,
        tps=final_vllm_tps,
        perf_weight=args.score_perf_weight,
        speed_weight=args.score_speed_weight,
    )
    append_metric(
        metrics,
        stage="final",
        trial=0,
        perplexity=final_ppl,
        hf_tokens_per_sec=final_hf_tps,
        vllm_tokens_per_sec=final_vllm_tps,
        model_size_mb=dir_size_mb(final_model_dir),
        notes=best_note,
        params={"weighted_proxy": final_proxy, **final_proxy_details},
    )

    summary = {
        "base_model": args.base_model,
        "base_model_dir": args.base_model_dir,
        "selected_quant_trial": None if selected_trial is None else asdict(selected_trial),
        "selection_metric": args.selection_metric,
        "score_perf_weight": args.score_perf_weight,
        "score_speed_weight": args.score_speed_weight,
        "base_perplexity": base_ppl,
        "base_hf_tokens_per_sec": base_hf_tps,
        "base_vllm_tokens_per_sec": base_vllm_tps,
        "final_perplexity": final_ppl,
        "final_hf_tokens_per_sec": final_hf_tps,
        "final_vllm_tokens_per_sec": final_vllm_tps,
        "final_weighted_proxy": final_proxy,
        "model_dir": str(final_model_dir),
        "submit_zip": str(zip_path),
        "model_validation": model_validation.to_dict(),
        "zip_validation": zip_validation.to_dict(),
        "smoke_test": final_smoke,
        "config_paths": config_paths,
        "gptq_sweep_csv": str(gptq_sweep_csv),
    }

    csv_path, jsonl_path, summary_path = write_metric_reports(
        out_dir=out_dir,
        report_path=args.report_path,
        metrics=metrics,
        summary=summary,
    )

    print_metric_table(metrics)
    print(f"[INFO] model_dir: {final_model_dir}")
    print(f"[INFO] submit.zip: {zip_path}")
    print(f"[INFO] metrics.csv: {csv_path}")
    print(f"[INFO] metrics.jsonl: {jsonl_path}")
    print(f"[INFO] final_summary.json: {summary_path}")


if __name__ == "__main__":
    main()
