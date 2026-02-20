from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KD -> LoRA -> Quant pipeline wrapper.")

    parser.add_argument("--student_model", type=str, default="./base_model")
    parser.add_argument("--base_model", type=str, default="./base_model")
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--teacher_preset_32b", type=str, default="LGAI-EXAONE/EXAONE-4.0-32B")
    parser.add_argument("--teacher_preset_24b", type=str, default="LGAI-EXAONE/EXAONE-4.0-2.4B")
    parser.add_argument("--teacher_preset_12b", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--kd_num_samples", type=int, default=5000)
    parser.add_argument("--kd_max_new_tokens", type=int, default=256)
    parser.add_argument("--kd_temperature", type=float, default=0.0)
    parser.add_argument("--kd_top_p", type=float, default=1.0)
    parser.add_argument(
        "--kd_format",
        type=str,
        choices=["prompt_completion", "conversations", "both"],
        default="prompt_completion",
    )
    parser.add_argument("--kd_output_path", type=str, default="./kd_data/train.jsonl")

    parser.add_argument("--train_data_format", type=str, choices=["prompt_completion", "conversations"], default="prompt_completion")
    parser.add_argument("--output_lora_dir", type=str, default="./distilled_lora")
    parser.add_argument("--output_merged_dir", type=str, default="./distilled_merged")
    parser.add_argument("--train_max_seq_len", type=int, default=1024)
    parser.add_argument("--train_epochs", type=float, default=1.0)
    parser.add_argument("--train_lr", type=float, default=2e-4)
    parser.add_argument("--train_per_device_batch_size", type=int, default=1)
    parser.add_argument("--train_grad_accum_steps", type=int, default=32)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="")

    parser.add_argument("--input_model_dir", type=str, default="./distilled_merged")
    parser.add_argument("--out_dir", type=str, default="./model")
    parser.add_argument("--quant_method", type=str, choices=["gptq", "awq"], default="gptq")
    parser.add_argument("--num_calibration_samples", type=int, default=1024)
    parser.add_argument("--quant_max_seq_length", type=int, default=1024)
    parser.add_argument("--scheme", type=str, default="W4A16")
    parser.add_argument("--targets", type=str, default="Linear")
    parser.add_argument("--ignore", type=str, default="embed_tokens,lm_head")
    parser.add_argument("--zip_name", type=str, default="submit")
    parser.add_argument("--run_vllm_smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict_vllm_smoke", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--awq_allow_skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--awq_fallback_to_gptq", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--run_lm_eval", action="store_true", default=False)
    parser.add_argument("--lm_eval_tasks", type=str, default="")
    parser.add_argument("--lm_eval_model_backend", type=str, default="vllm")

    parser.add_argument("--stop_after", type=str, choices=["kd", "train", "quant", "none"], default="none")
    return parser.parse_args()


def validate_lm_eval_args(run_lm_eval: bool, lm_eval_tasks: str) -> None:
    if run_lm_eval and not lm_eval_tasks.strip():
        raise ValueError("--lm_eval_tasks is required when --run_lm_eval is enabled.")


def run_command(command: List[str]) -> None:
    print("[INFO] run:", " ".join(shlex.quote(token) for token in command))
    subprocess.run(command, check=True)


def run_lm_eval(backend: str, tasks: str, model_dir: str) -> None:
    model_args = f"pretrained={model_dir}"
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        backend,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "auto",
    ]
    run_command(command)


def main() -> None:
    args = parse_args()
    validate_lm_eval_args(args.run_lm_eval, args.lm_eval_tasks)

    kd_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "generate_kd_data.py"),
        "--student_model",
        args.student_model,
        "--teacher_model",
        args.teacher_model,
        "--teacher_preset_32b",
        args.teacher_preset_32b,
        "--teacher_preset_24b",
        args.teacher_preset_24b,
        "--teacher_preset_12b",
        args.teacher_preset_12b,
        "--dataset_id",
        args.dataset_id,
        "--dataset_split",
        args.dataset_split,
        "--num_samples",
        str(args.kd_num_samples),
        "--max_new_tokens",
        str(args.kd_max_new_tokens),
        "--temperature",
        str(args.kd_temperature),
        "--top_p",
        str(args.kd_top_p),
        "--kd_format",
        args.kd_format,
        "--output_path",
        args.kd_output_path,
        "--seed",
        str(args.seed),
    ]
    run_command(kd_cmd)
    if args.stop_after == "kd":
        return

    train_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_kd_lora.py"),
        "--base_model",
        args.base_model,
        "--kd_data_path",
        args.kd_output_path,
        "--data_format",
        args.train_data_format,
        "--output_lora_dir",
        args.output_lora_dir,
        "--output_merged_dir",
        args.output_merged_dir,
        "--max_seq_len",
        str(args.train_max_seq_len),
        "--epochs",
        str(args.train_epochs),
        "--lr",
        str(args.train_lr),
        "--per_device_batch_size",
        str(args.train_per_device_batch_size),
        "--grad_accum_steps",
        str(args.train_grad_accum_steps),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        str(args.lora_dropout),
        "--target_modules",
        args.target_modules,
        "--seed",
        str(args.seed),
    ]
    run_command(train_cmd)
    if args.stop_after == "train":
        return

    quant_input = args.input_model_dir
    if quant_input.strip() == "./distilled_merged":
        quant_input = args.output_merged_dir

    quant_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "quantize.py"),
        "--input_model_dir",
        quant_input,
        "--out_dir",
        args.out_dir,
        "--quant_method",
        args.quant_method,
        "--dataset_id",
        args.dataset_id,
        "--dataset_split",
        args.dataset_split,
        "--num_calibration_samples",
        str(args.num_calibration_samples),
        "--max_seq_length",
        str(args.quant_max_seq_length),
        "--scheme",
        args.scheme,
        "--targets",
        args.targets,
        "--ignore",
        args.ignore,
        "--zip_name",
        args.zip_name,
    ]
    quant_cmd.append("--run_vllm_smoke" if args.run_vllm_smoke else "--no-run_vllm_smoke")
    quant_cmd.append("--strict_vllm_smoke" if args.strict_vllm_smoke else "--no-strict_vllm_smoke")
    quant_cmd.append("--awq_allow_skip" if args.awq_allow_skip else "--no-awq_allow_skip")
    quant_cmd.append("--awq_fallback_to_gptq" if args.awq_fallback_to_gptq else "--no-awq_fallback_to_gptq")
    run_command(quant_cmd)

    if args.stop_after == "quant":
        return

    if args.run_lm_eval:
        run_lm_eval(
            backend=args.lm_eval_model_backend,
            tasks=args.lm_eval_tasks,
            model_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()

