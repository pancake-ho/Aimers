import argparse
import csv
import gc
import itertools
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from dataset.prepare_dataset import build_mixed_train_dataset, make_calib_dataset, prepare_dataset
# from quantizing import AutoRoundQuantizer
from tuning import KDTrainer, DataCollatorForCausalLM, Fine_tuning, build_kd_features
from utils import save


def parse_args():
    parser = argparse.ArgumentParser(description="Aimers quantization pipeline")

    parser.add_argument("--base_model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--out_dir", type=str, default="./artifacts")

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_external_mix", action="store_true")
    parser.add_argument(
        "--mix_dataset_ids",
        type=str,
        default="MyeongHo0621/korean-quality-cleaned,m-a-p/Code-Feedback",
    )
    parser.add_argument("--mix_dataset_splits", type=str, default="train,train")
    parser.add_argument("--mix_dataset_configs", type=str, default=",all")
    parser.add_argument("--mix_weights", type=str, default="0.60,0.25,0.15")
    parser.add_argument(
        "--mix_turn_policy",
        type=str,
        choices=["last_assistant", "keep_full", "two_turn"],
        default="last_assistant",
    )
    parser.add_argument("--mix_apply_stages", type=str, default="kd,lora")

    parser.add_argument("--do_kd", action="store_true")
    parser.add_argument("--kd_out", type=str, default="./artifacts/kd")
    parser.add_argument("--kd_samples", type=int, default=50_000)
    parser.add_argument("--kd_max_len", type=int, default=1024)
    parser.add_argument("--kd_lr", type=float, default=5e-5)
    parser.add_argument("--kd_steps", type=int, default=1200)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--disable_kd_grid", action="store_true")

    parser.add_argument("--do_lora", action="store_true")
    parser.add_argument("--lora_out", type=str, default="./artifacts/lora")
    parser.add_argument("--lora_samples", type=int, default=2000)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_epochs", type=float, default=1.0)
    parser.add_argument("--lora_lr", type=float, default=1e-4)

    parser.add_argument("--quant_method", type=str, choices=["gptq", "autoround"], default="gptq")
    parser.add_argument("--calib_samples", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--disable_quant_grid", action="store_true")

    parser.add_argument("--gptq_block_size", type=int, default=128)
    parser.add_argument("--gptq_dampening", type=float, default=0.01)

    parser.add_argument("--autoround_bits", type=int, default=4)
    parser.add_argument("--autoround_group_size", type=int, default=128)
    parser.add_argument("--autoround_sym", action="store_true")
    parser.add_argument("--autoround_iters", type=int, default=500)
    parser.add_argument("--autoround_lr", type=float, default=1e-2)

    # Evaluation split is separated from training by default.
    parser.add_argument("--eval_dataset_id", type=str, default="")
    parser.add_argument("--eval_dataset_split", type=str, default="train")
    parser.add_argument("--eval_start", type=int, default=200_000)
    parser.add_argument("--eval_count", type=int, default=128)
    # Backward compatibility alias
    parser.add_argument("--eval_samples", type=int, default=-1)
    parser.add_argument("--eval_max_len", type=int, default=1024)
    parser.add_argument("--skip_eval", action="store_true")

    parser.add_argument("--search_budget", type=int, default=6, help="Max trial count per stage")
    parser.add_argument("--report_path", type=str, default="metrics.csv")
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["lb_proxy", "ppl"],
        default="lb_proxy",
        help="Trial selection metric. lb_proxy aligns with leaderboard-style perf+speed objective.",
    )
    parser.add_argument("--score_perf_weight", type=float, default=0.5)
    parser.add_argument("--score_speed_weight", type=float, default=0.5)

    parser.add_argument("--skip_rehearsal", action="store_true")
    parser.add_argument("--rehearsal_mode", type=str, default="full", choices=["package", "config", "tokenizer", "smoke", "full"])
    parser.add_argument("--strict_rehearsal", action="store_true")

    return parser.parse_args()


@dataclass
class StageMetric:
    stage: str
    trial: int
    perplexity: Optional[float]
    tokens_per_sec: Optional[float]
    model_size_mb: Optional[float]
    notes: str
    params: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Utility helpers
# ----------------------------
def _to_text(tokenizer, example: Dict[str, Any], add_generation_prompt: bool) -> Optional[str]:
    conv = example.get("conversations")
    if not conv:
        return None
    try:
        return tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception:
        return None


def _get_first_param_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def _cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def runtime_torch_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def use_bf16_training() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def _parse_stage_set(raw: str) -> set:
    return {s.strip().lower() for s in str(raw).split(",") if s.strip()}


def should_apply_external_mix(args, stage: str) -> bool:
    if not bool(args.use_external_mix):
        return False
    stages = _parse_stage_set(args.mix_apply_stages)
    return stage.strip().lower() in stages


def _extract_mix_metric_params(mix_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not mix_meta:
        return {}
    return {
        "mix_applied": bool(mix_meta.get("mix_applied", False)),
        "mix_stage": mix_meta.get("stage"),
        "mix_target_count": mix_meta.get("target_count"),
        "mix_actual_count": mix_meta.get("actual_count"),
        "mix_shortfall": mix_meta.get("shortfall"),
        "mix_counts": mix_meta.get("mix_counts"),
        "mix_weights": mix_meta.get("mix_weights"),
        "mix_sources": mix_meta.get("mix_sources"),
        "mix_turn_policy": mix_meta.get("mix_turn_policy"),
    }


def _score_lower_is_better(value: Optional[float]) -> float:
    return float("inf") if value is None else value


def compute_lb_proxy_score(
    base_ppl: Optional[float],
    base_tps: Optional[float],
    ppl: Optional[float],
    tps: Optional[float],
    perf_weight: float,
    speed_weight: float,
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    perf_ratio = None
    if base_ppl is not None and ppl is not None and base_ppl > 0.0 and ppl > 0.0:
        perf_ratio = base_ppl / ppl

    tpt_reduction_ratio = None
    tps_gain_ratio = None
    if base_tps is not None and tps is not None and base_tps > 0.0 and tps > 0.0:
        tpt_reduction_ratio = 1.0 - (base_tps / tps)
        tps_gain_ratio = (tps - base_tps) / base_tps

    score_terms: List[Tuple[float, float]] = []
    if perf_ratio is not None:
        score_terms.append((max(0.0, float(perf_weight)), perf_ratio - 1.0))
    if tpt_reduction_ratio is not None:
        score_terms.append((max(0.0, float(speed_weight)), tpt_reduction_ratio))

    score = None
    if score_terms:
        denom = sum(w for w, _ in score_terms)
        if denom <= 0.0:
            denom = float(len(score_terms))
            score = sum(v for _, v in score_terms) / denom
        else:
            score = sum(w * v for w, v in score_terms) / denom

    details = {
        "perf_ratio": perf_ratio,
        "perf_gain_ratio": None if perf_ratio is None else (perf_ratio - 1.0),
        "tpt_reduction_ratio": tpt_reduction_ratio,
        "tps_gain_ratio": tps_gain_ratio,
    }
    return score, details


def build_trial_selection_key(
    selection_metric: str,
    ppl: Optional[float],
    lb_proxy_score: Optional[float],
) -> Tuple[Tuple[int, float], str]:
    if selection_metric == "ppl":
        if ppl is not None:
            return (2, -ppl), "ppl"
        if lb_proxy_score is not None:
            return (1, lb_proxy_score), "lb_proxy_fallback"
        return (0, float("-inf")), "none"

    # lb_proxy is default
    if lb_proxy_score is not None:
        return (2, lb_proxy_score), "lb_proxy"
    if ppl is not None:
        return (1, -ppl), "ppl_fallback"
    return (0, float("-inf")), "none"


def estimate_model_size_mb(model) -> Optional[float]:
    try:
        total = 0
        for tensor in model.state_dict().values():
            if torch.is_tensor(tensor):
                total += int(tensor.numel()) * int(tensor.element_size())
        return total / (1024 * 1024)
    except Exception:
        return None


def dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def resolve_report_paths(out_dir: str, report_path: str) -> Tuple[Path, Path]:
    base = Path(report_path)
    if not base.is_absolute():
        base = Path(out_dir) / base

    if base.suffix.lower() == ".jsonl":
        jsonl_path = base
        csv_path = base.with_suffix(".csv")
    else:
        csv_path = base if base.suffix.lower() == ".csv" else base.with_suffix(".csv")
        jsonl_path = csv_path.with_suffix(".jsonl")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, jsonl_path


def append_metric(
    metrics: List[StageMetric],
    stage: str,
    trial: int,
    perplexity: Optional[float],
    tokens_per_sec: Optional[float],
    model_size_mb: Optional[float],
    notes: str,
    params: Optional[Dict[str, Any]] = None,
):
    metrics.append(
        StageMetric(
            stage=stage,
            trial=trial,
            perplexity=perplexity,
            tokens_per_sec=tokens_per_sec,
            model_size_mb=model_size_mb,
            notes=notes,
            params=params or {},
        )
    )


def print_metric_table(metrics: List[StageMetric]):
    print("\n=== Metric Summary ===")
    print(
        f"{'Stage':18} | {'Trial':>5} | {'PPL':>10} | {'Tok/s':>10} | {'Model(MB)':>10} | Notes"
    )
    print("-" * 96)
    for m in metrics:
        ppl = "N/A" if m.perplexity is None else f"{m.perplexity:.4f}"
        tps = "N/A" if m.tokens_per_sec is None else f"{m.tokens_per_sec:.2f}"
        ms = "N/A" if m.model_size_mb is None else f"{m.model_size_mb:.2f}"
        print(f"{m.stage:18} | {m.trial:5d} | {ppl:>10} | {tps:>10} | {ms:>10} | {m.notes}")


def write_metric_reports(
    args,
    metrics: List[StageMetric],
    model_dir: Path,
    zip_path: Path,
    rehearsal: Dict[str, Any],
):
    csv_path, jsonl_path = resolve_report_paths(args.out_dir, args.report_path)

    rows = []
    for m in metrics:
        row = asdict(m)
        row["params"] = json.dumps(row["params"], ensure_ascii=False)
        rows.append(row)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "trial",
                "perplexity",
                "tokens_per_sec",
                "model_size_mb",
                "notes",
                "params",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")

    base = next((m for m in metrics if m.stage == "base"), None)
    final = next((m for m in reversed(metrics) if m.stage == "final"), None)

    ppl_gain_pct = None
    if base and final and base.perplexity and final.perplexity:
        ppl_gain_pct = (base.perplexity - final.perplexity) / base.perplexity * 100.0

    tps_gain_pct = None
    if base and final and base.tokens_per_sec and final.tokens_per_sec:
        tps_gain_pct = (final.tokens_per_sec - base.tokens_per_sec) / base.tokens_per_sec * 100.0

    final_lb_proxy_score, final_lb_proxy_details = compute_lb_proxy_score(
        None if base is None else base.perplexity,
        None if base is None else base.tokens_per_sec,
        None if final is None else final.perplexity,
        None if final is None else final.tokens_per_sec,
        args.score_perf_weight,
        args.score_speed_weight,
    )

    leaderboard = {
        "base_model": args.base_model,
        "quant_method": args.quant_method,
        "search_budget": args.search_budget,
        "selection_metric": args.selection_metric,
        "score_perf_weight": args.score_perf_weight,
        "score_speed_weight": args.score_speed_weight,
        "use_external_mix": bool(args.use_external_mix),
        "mix_dataset_ids": args.mix_dataset_ids,
        "mix_dataset_splits": args.mix_dataset_splits,
        "mix_dataset_configs": args.mix_dataset_configs,
        "mix_weights": args.mix_weights,
        "mix_turn_policy": args.mix_turn_policy,
        "mix_apply_stages": args.mix_apply_stages,
        "metrics_csv": str(csv_path),
        "metrics_jsonl": str(jsonl_path),
        "model_dir": str(model_dir),
        "submit_zip": str(zip_path),
        "model_dir_size_mb": dir_size_mb(model_dir),
        "base_perplexity": None if base is None else base.perplexity,
        "final_perplexity": None if final is None else final.perplexity,
        "perplexity_gain_pct": ppl_gain_pct,
        "base_tokens_per_sec": None if base is None else base.tokens_per_sec,
        "final_tokens_per_sec": None if final is None else final.tokens_per_sec,
        "tokens_per_sec_gain_pct": tps_gain_pct,
        "lb_proxy_score": final_lb_proxy_score,
        "perf_ratio": final_lb_proxy_details["perf_ratio"],
        "tpt_reduction_ratio": final_lb_proxy_details["tpt_reduction_ratio"],
        "tps_gain_ratio": final_lb_proxy_details["tps_gain_ratio"],
        "rehearsal": rehearsal,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_json_path = out_dir / "metrics.json"
    metric_json_path.write_text(
        json.dumps(
            {
                "metrics": [asdict(m) for m in metrics],
                "leaderboard": leaderboard,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leaderboard_path = out_dir / "leaderboard_ready_report.json"
    leaderboard_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] metrics.csv: {csv_path}")
    print(f"[INFO] metrics.jsonl: {jsonl_path}")
    print(f"[INFO] metrics.json: {metric_json_path}")
    print(f"[INFO] leaderboard report: {leaderboard_path}")


# ----------------------------
# Evaluation
# ----------------------------
def build_eval_texts(args, tokenizer) -> List[str]:
    if args.eval_samples > 0:
        args.eval_count = args.eval_samples

    if args.eval_count <= 0:
        return []

    eval_dataset_id = args.eval_dataset_id or args.dataset_id
    ds = load_dataset(eval_dataset_id, split=args.eval_dataset_split)
    ds = ds.shuffle(seed=args.seed + 17)

    total = len(ds)
    if total <= 0:
        return []

    start = max(0, int(args.eval_start))
    if start >= total:
        start = max(0, total - args.eval_count)

    end = min(total, start + args.eval_count)
    sub = ds.select(range(start, end))

    texts = []
    for ex in sub:
        text = _to_text(tokenizer, ex, add_generation_prompt=False)
        if text:
            texts.append(text)

    print(
        f"[INFO] eval set: dataset={eval_dataset_id}, split={args.eval_dataset_split}, "
        f"slice=[{start}:{end}), usable={len(texts)}"
    )
    return texts


@torch.no_grad()
def evaluate_model(model, tokenizer, texts: List[str], max_len: int) -> Tuple[Optional[float], Optional[float]]:
    if not texts:
        return None, None

    model.eval()
    device = _get_first_param_device(model)

    total_nll = 0.0
    total_tokens = 0
    t0 = time.perf_counter()

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        valid_tokens = int(attention_mask.sum().item())
        if valid_tokens < 2:
            continue

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        n_tokens = valid_tokens - 1
        total_nll += float(out.loss.detach().float().item()) * n_tokens
        total_tokens += n_tokens

    elapsed = max(1e-9, time.perf_counter() - t0)
    if total_tokens == 0:
        return None, None

    ppl = math.exp(total_nll / total_tokens)
    tps = total_tokens / elapsed
    return float(ppl), float(tps)


def save_model_checkpoint(model, tokenizer, out_dir: str):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


# ----------------------------
# KD stage
# ----------------------------
def build_kd_train_dataset(args, tokenizer) -> Tuple[Dataset, Dict[str, Any]]:
    mix_meta: Dict[str, Any] = {}
    if should_apply_external_mix(args, "kd"):
        raw, mix_meta = build_mixed_train_dataset(
            args=args,
            target_count=args.kd_samples,
            stage="kd",
            seed=args.seed,
        )
    else:
        raw = load_dataset(args.dataset_id, split=args.dataset_split)
        raw = raw.shuffle(seed=args.seed).select(range(min(args.kd_samples, len(raw))))

    def _map(ex):
        feat = build_kd_features(tokenizer, ex, args.kd_max_len)
        if feat is None:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        return feat

    train_ds = raw.map(_map, remove_columns=raw.column_names)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)
    if mix_meta:
        mix_meta = dict(mix_meta)
        mix_meta["raw_count"] = int(len(raw))
        mix_meta["usable_count"] = int(len(train_ds))
    return train_ds, mix_meta


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def build_kd_trials(args) -> List[Tuple[float, float, int]]:
    if args.disable_kd_grid:
        return [(args.kd_temp, args.kd_alpha, args.kd_steps)]

    temps = sorted({max(1.0, args.kd_temp - 0.5), args.kd_temp, args.kd_temp + 0.5})
    alphas = sorted(
        {
            _clamp(args.kd_alpha - 0.1, 0.1, 0.9),
            _clamp(args.kd_alpha, 0.1, 0.9),
            _clamp(args.kd_alpha + 0.1, 0.1, 0.9),
        }
    )
    steps = sorted({max(300, int(args.kd_steps * 0.7)), int(args.kd_steps)})

    candidates = list(itertools.product(temps, alphas, steps))

    def _dist(c):
        t, a, s = c
        return abs(t - args.kd_temp) + abs(a - args.kd_alpha) + abs(s - args.kd_steps) / max(1, args.kd_steps)

    candidates.sort(key=_dist)
    return candidates[: max(1, args.search_budget)]


def run_kd_trial(
    args,
    tokenizer,
    train_ds,
    eval_texts: List[str],
    trial_idx: int,
    temperature: float,
    alpha: float,
    steps: int,
) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
    out_dir = Path(args.kd_out) / f"trial_{trial_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] KD trial#{trial_idx} start temp={temperature}, "
        f"alpha={alpha}, steps={steps}"
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
    ).eval()

    student = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
    )

    student.config.use_cache = False
    student.gradient_checkpointing_enable()

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=int(steps),
        learning_rate=args.kd_lr,
        warmup_ratio=0.03,
        bf16=use_bf16_training(),
        logging_steps=20,
        save_strategy="no",
        report_to=[],
    )

    trainer = KDTrainer(
        model=student,
        teacher_model=teacher,
        temperature=float(temperature),
        alpha=float(alpha),
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()

    ppl, tps = (None, None)
    if not args.skip_eval:
        ppl, tps = evaluate_model(trainer.model, tokenizer, eval_texts, args.eval_max_len)

    model_size = estimate_model_size_mb(trainer.model)
    save_model_checkpoint(trainer.model, tokenizer, str(out_dir))

    del trainer
    del teacher
    _cleanup_cuda()

    print(f"[INFO] KD trial#{trial_idx} done -> {out_dir} (ppl={ppl}, tok/s={tps})")
    return str(out_dir), ppl, tps, model_size


def run_kd_if_enabled(
    args,
    tokenizer,
    metrics: List[StageMetric],
    eval_texts: List[str],
    base_ppl: Optional[float],
    base_tps: Optional[float],
) -> str:
    if not args.do_kd:
        return args.base_model

    train_ds, kd_mix_meta = build_kd_train_dataset(args, tokenizer)
    trials = build_kd_trials(args)

    best_path = None
    best_key = None
    best_rule = "none"
    best_proxy = None

    for idx, (temp, alpha, steps) in enumerate(trials, start=1):
        ckpt, ppl, tps, msz = run_kd_trial(args, tokenizer, train_ds, eval_texts, idx, temp, alpha, steps)
        lb_proxy_score, lb_proxy_details = compute_lb_proxy_score(
            base_ppl,
            base_tps,
            ppl,
            tps,
            args.score_perf_weight,
            args.score_speed_weight,
        )
        trial_key, selection_rule = build_trial_selection_key(args.selection_metric, ppl, lb_proxy_score)
        note = f"temp={temp}, alpha={alpha}, steps={steps}, select={selection_rule}"
        if lb_proxy_score is not None:
            note += f", proxy={lb_proxy_score:.6f}"
        if kd_mix_meta.get("mix_applied"):
            note += ", mix=on"

        append_metric(
            metrics,
            stage="kd",
            trial=idx,
            perplexity=ppl,
            tokens_per_sec=tps,
            model_size_mb=msz,
            notes=note,
            params={
                "temp": temp,
                "alpha": alpha,
                "steps": steps,
                "selection_rule": selection_rule,
                "lb_proxy_score": lb_proxy_score,
                **_extract_mix_metric_params(kd_mix_meta),
                **lb_proxy_details,
            },
        )

        if best_path is None or best_key is None or trial_key > best_key:
            best_path = ckpt
            best_key = trial_key
            best_rule = selection_rule
            best_proxy = lb_proxy_score

    print(f"[INFO] best KD checkpoint: {best_path} (rule={best_rule}, proxy={best_proxy})")
    return best_path


# ----------------------------
# LoRA stage
# ----------------------------
def run_lora_if_enabled(args, tokenizer, model_path: str, metrics: List[StageMetric], eval_texts: List[str]) -> str:
    if not args.do_lora:
        return model_path

    print(f"[INFO] LoRA stage start from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
    )

    lora_mix_meta: Dict[str, Any] = {}
    if should_apply_external_mix(args, "lora"):
        train_ds, lora_mix_meta = build_mixed_train_dataset(
            args=args,
            target_count=args.lora_samples,
            stage="lora",
            seed=args.seed + 101,
        )
    else:
        train_ds, _ = prepare_dataset(
            args.dataset_id,
            args.dataset_split,
            args.lora_samples,
            0,
            seed=args.seed,
        )
    tuner = Fine_tuning(
        model=model,
        tokenizer=tokenizer,
        seq_length=args.max_seq_len,
        train_ds=train_ds,
    )

    tuned_model = tuner.setup_lora(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        epochs=args.lora_epochs,
        lr=args.lora_lr,
    )

    lora_out = Path(args.lora_out)
    save_model_checkpoint(tuned_model, tokenizer, str(lora_out))

    ppl, tps = (None, None)
    if not args.skip_eval:
        ppl, tps = evaluate_model(tuned_model, tokenizer, eval_texts, args.eval_max_len)

    note = f"samples={args.lora_samples}, r={args.lora_r}"
    if lora_mix_meta.get("mix_applied"):
        note += ", mix=on"

    append_metric(
        metrics,
        stage="lora",
        trial=1,
        perplexity=ppl,
        tokens_per_sec=tps,
        model_size_mb=estimate_model_size_mb(tuned_model),
        notes=note,
        params={
            "samples": args.lora_samples,
            "r": args.lora_r,
            **_extract_mix_metric_params(lora_mix_meta),
        },
    )

    del tuned_model
    _cleanup_cuda()
    print(f"[INFO] LoRA stage done -> {lora_out}")
    return str(lora_out)


# ----------------------------
# Quantization stage
# ----------------------------
def run_gptq_quantization(args, tokenizer, model_path: str, calib_samples: int, block_size: int, dampening: float):
    print(
        f"[INFO] GPTQ start (calib={calib_samples}, block={block_size}, damp={dampening})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
    )

    calib_ds = make_calib_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        split=args.dataset_split,
        n=calib_samples,
        seed=args.seed,
    )

    recipe = [
        GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["re:.*embed_tokens.*", "re:.*lm_head.*"],
            block_size=block_size,
            dampening_frac=dampening,
            actorder="weight",
            offload_hessians=False,
        )
    ]

    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=calib_samples,
        concatenate_data=True,
        pad_to_max_length=False,
        shuffle_calibration_samples=True,
    )
    return model


def run_autoround_quantization(
    args,
    tokenizer,
    model_path: str,
    calib_samples: int,
    group_size: int,
    sym: bool,
):
    print(
        f"[INFO] AutoRound start (calib={calib_samples}, group={group_size}, sym={sym})"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=runtime_torch_dtype(),
        device_map="auto",
    )

    calib_ds = make_calib_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        split=args.dataset_split,
        n=calib_samples,
        seed=args.seed,
    )

    quantizer = AutoRoundQuantizer(
        model=model,
        tokenizer=tokenizer,
        calib_dataset=calib_ds,
        seq_length=args.max_seq_len,
        bits=args.autoround_bits,
        group_size=group_size,
        sym=sym,
        iters=args.autoround_iters,
        lr=args.autoround_lr,
    )
    return quantizer.execute()


def build_quant_trials(args) -> List[Dict[str, Any]]:
    if args.disable_quant_grid:
        if args.quant_method == "gptq":
            return [
                {
                    "method": "gptq",
                    "calib_samples": args.calib_samples,
                    "block_size": args.gptq_block_size,
                    "dampening": args.gptq_dampening,
                }
            ]
        return [
            {
                "method": "autoround",
                "calib_samples": args.calib_samples,
                "group_size": args.autoround_group_size,
                "sym": args.autoround_sym,
            }
        ]

    calib_set = sorted(
        {
            max(256, int(args.calib_samples * 0.5)),
            int(args.calib_samples),
            max(256, int(args.calib_samples * 1.5)),
        }
    )

    if args.quant_method == "gptq":
        trials = []
        for calib, block, damp in itertools.product(calib_set, [64, 128, 256], [0.005, args.gptq_dampening]):
            trials.append(
                {
                    "method": "gptq",
                    "calib_samples": int(calib),
                    "block_size": int(block),
                    "dampening": float(damp),
                }
            )
    else:
        sym_values = sorted({False, bool(args.autoround_sym)})
        trials = []
        for calib, group, sym in itertools.product(calib_set, [64, 128, 256], sym_values):
            trials.append(
                {
                    "method": "autoround",
                    "calib_samples": int(calib),
                    "group_size": int(group),
                    "sym": bool(sym),
                }
            )

    # Prefer center configs first.
    def _dist(cfg):
        d = abs(cfg["calib_samples"] - args.calib_samples) / max(1, args.calib_samples)
        if cfg["method"] == "gptq":
            d += abs(cfg["block_size"] - args.gptq_block_size) / max(1, args.gptq_block_size)
        else:
            d += abs(cfg["group_size"] - args.autoround_group_size) / max(1, args.autoround_group_size)
        return d

    trials.sort(key=_dist)
    return trials[: max(1, args.search_budget)]


def run_quantization_with_grid(
    args,
    tokenizer,
    model_path: str,
    metrics: List[StageMetric],
    eval_texts: List[str],
    base_ppl: Optional[float],
    base_tps: Optional[float],
):
    trials = build_quant_trials(args)
    best_model = None
    best_note = ""
    best_key = None

    for idx, trial in enumerate(trials, start=1):
        if trial["method"] == "gptq":
            q_model = run_gptq_quantization(
                args,
                tokenizer,
                model_path,
                trial["calib_samples"],
                trial["block_size"],
                trial["dampening"],
            )
            note = (
                f"method=gptq, calib={trial['calib_samples']}, "
                f"block={trial['block_size']}, damp={trial['dampening']}"
            )
        else:
            q_model = run_autoround_quantization(
                args,
                tokenizer,
                model_path,
                trial["calib_samples"],
                trial["group_size"],
                trial["sym"],
            )
            note = (
                f"method=autoround, calib={trial['calib_samples']}, "
                f"group={trial['group_size']}, sym={trial['sym']}"
            )

        ppl, tps = (None, None)
        if not args.skip_eval:
            ppl, tps = evaluate_model(q_model, tokenizer, eval_texts, args.eval_max_len)

        lb_proxy_score, lb_proxy_details = compute_lb_proxy_score(
            base_ppl,
            base_tps,
            ppl,
            tps,
            args.score_perf_weight,
            args.score_speed_weight,
        )
        trial_key, selection_rule = build_trial_selection_key(args.selection_metric, ppl, lb_proxy_score)

        append_metric(
            metrics,
            stage="quant",
            trial=idx,
            perplexity=ppl,
            tokens_per_sec=tps,
            model_size_mb=estimate_model_size_mb(q_model),
            notes=f"{note}, select={selection_rule}"
            + ("" if lb_proxy_score is None else f", proxy={lb_proxy_score:.6f}"),
            params={
                **trial,
                "selection_rule": selection_rule,
                "lb_proxy_score": lb_proxy_score,
                **lb_proxy_details,
            },
        )

        if best_model is None or best_key is None or trial_key > best_key:
            if best_model is not None:
                del best_model
                _cleanup_cuda()
            best_model = q_model
            best_key = trial_key
            best_note = f"{note}, select={selection_rule}"
            if lb_proxy_score is not None:
                best_note += f", proxy={lb_proxy_score:.6f}"
        else:
            del q_model
            _cleanup_cuda()

    print(f"[INFO] best quant config: {best_note}")
    return best_model, best_note


# ----------------------------
# Submission rehearsal
# ----------------------------
def run_submission_rehearsal(args, zip_path: Path) -> Dict[str, Any]:
    if args.skip_rehearsal:
        return {"status": "skipped", "mode": args.rehearsal_mode}

    repo_root = Path(__file__).resolve().parent.parent
    test_script = repo_root / "test.py"

    cmd = [
        sys.executable,
        str(test_script),
        "--zip",
        str(zip_path),
        "--mode",
        args.rehearsal_mode,
    ]
    print(f"[INFO] Rehearsal start: {' '.join(cmd)}")

    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    if proc.returncode == 0:
        print("[INFO] Rehearsal passed")
        return {"status": "passed", "mode": args.rehearsal_mode, "returncode": 0}

    print("[ERROR] Rehearsal failed")
    if proc.stdout:
        print("[ERROR] rehearsal stdout (tail):")
        print("\n".join(proc.stdout.splitlines()[-40:]))
    if proc.stderr:
        print("[ERROR] rehearsal stderr (tail):")
        print("\n".join(proc.stderr.splitlines()[-40:]))

    result = {
        "status": "failed",
        "mode": args.rehearsal_mode,
        "returncode": proc.returncode,
    }

    if args.strict_rehearsal:
        raise RuntimeError("submission rehearsal failed")

    return result


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    metrics: List[StageMetric] = []
    base_ppl: Optional[float] = None
    base_tps: Optional[float] = None

    eval_texts = []
    if not args.skip_eval:
        eval_texts = build_eval_texts(args, tokenizer)

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=runtime_torch_dtype(),
            device_map="auto",
        )
        base_ppl, base_tps = evaluate_model(base_model, tokenizer, eval_texts, args.eval_max_len)
        append_metric(
            metrics,
            stage="base",
            trial=0,
            perplexity=base_ppl,
            tokens_per_sec=base_tps,
            model_size_mb=estimate_model_size_mb(base_model),
            notes=f"eval_count={len(eval_texts)}",
            params={
                "eval_dataset_id": args.eval_dataset_id or args.dataset_id,
                "eval_dataset_split": args.eval_dataset_split,
                "eval_start": args.eval_start,
                "eval_count": len(eval_texts),
            },
        )
        del base_model
        _cleanup_cuda()

    print("[INFO] Pipeline: load -> KD(optional) -> LoRA(optional) -> quantize -> eval -> package")

    model_path = run_kd_if_enabled(args, tokenizer, metrics, eval_texts, base_ppl, base_tps)
    model_path = run_lora_if_enabled(args, tokenizer, model_path, metrics, eval_texts)

    quant_model, quant_note = run_quantization_with_grid(
        args,
        tokenizer,
        model_path,
        metrics,
        eval_texts,
        base_ppl,
        base_tps,
    )

    final_ppl, final_tps = (None, None)
    if not args.skip_eval:
        final_ppl, final_tps = evaluate_model(quant_model, tokenizer, eval_texts, args.eval_max_len)
    final_lb_proxy_score, final_lb_proxy_details = compute_lb_proxy_score(
        base_ppl,
        base_tps,
        final_ppl,
        final_tps,
        args.score_perf_weight,
        args.score_speed_weight,
    )

    append_metric(
        metrics,
        stage="final",
        trial=0,
        perplexity=final_ppl,
        tokens_per_sec=final_tps,
        model_size_mb=estimate_model_size_mb(quant_model),
        notes=quant_note,
        params={
            "quant_method": args.quant_method,
            "selection_metric": args.selection_metric,
            "lb_proxy_score": final_lb_proxy_score,
            **final_lb_proxy_details,
        },
    )

    model_dir, zip_path = save(args.out_dir, quant_model, tokenizer)
    print(f"[INFO] saved model_dir: {model_dir}")
    print(f"[INFO] saved zip: {zip_path}")

    rehearsal = run_submission_rehearsal(args, zip_path)

    print_metric_table(metrics)
    write_metric_reports(args, metrics, model_dir, zip_path, rehearsal)


if __name__ == "__main__":
    main()
