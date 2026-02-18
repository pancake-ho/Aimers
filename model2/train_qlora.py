#!/usr/bin/env python
"""
QLoRA fine-tuning script for a pre-quantized EXAONE-4.0-1.2B model.

Key design constraints:
- Load quantized base model from local ./model (AutoRound auto_gptq output).
- LoRA targets: ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] only.
- Exclude q_proj, k_proj, embed_tokens, lm_head by design.
- Use math/reasoning dataset (openai/gsm8k train) with EXAONE chat template.
- Keep training safe for single L4 GPU (22.4GB): gradient checkpointing, tiny batch,
  paged_adamw_8bit optimizer.
- Merge LoRA into base model and export standalone model/tokenizer to ./model_final
  in auto_gptq format.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
from typing import Dict, Tuple

import torch
from datasets import concatenate_datasets, load_dataset
from auto_round import AutoRound
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


BASE_MODEL_DIR = "./model"
OUTPUT_DIR = "./model_final"
TARGET_MODULES = ["v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for quantized EXAONE model")
    parser.add_argument(
        "--preset",
        type=str,
        choices=["A", "B", "C"],
        default=None,
        help="Experiment preset: A(balance), B(quality), C(speed-protected).",
    )
    parser.add_argument("--model_dir", type=str, default=BASE_MODEL_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--mix_dataset_name",
        type=str,
        default="hendrycks/competition_math",
        help="Optional secondary reasoning dataset. Set '' to disable mixing.",
    )
    parser.add_argument("--mix_dataset_config", type=str, default=None)
    parser.add_argument("--mix_split", type=str, default="train")
    parser.add_argument(
        "--mix_ratio",
        type=float,
        default=0.3,
        help="Approximate fraction from secondary dataset in final train set.",
    )
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--merge_base_model_id",
        type=str,
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="Full-precision base model id/path used only for LoRA merge.",
    )
    parser.add_argument(
        "--merge_base_local_files_only",
        action="store_true",
        help="Force local_files_only=True when loading merge base model.",
    )
    parser.add_argument(
        "--quant_calib_samples",
        type=int,
        default=256,
        help="Calibration sample count for final AutoRound export.",
    )
    parser.add_argument(
        "--quant_iters",
        type=int,
        default=300,
        help="AutoRound tuning iterations for final export.",
    )
    args = parser.parse_args()
    apply_preset(args)
    return args


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "A":
        args.max_samples = 2000
        args.max_seq_length = 1024
        args.learning_rate = 1e-4
        args.quant_calib_samples = 256
        args.quant_iters = 300
        args.output_dir = "./model_final_A"
    elif args.preset == "B":
        args.max_samples = 4000
        args.max_seq_length = 1024
        args.learning_rate = 8e-5
        args.quant_calib_samples = 384
        args.quant_iters = 400
        args.output_dir = "./model_final_B"
    elif args.preset == "C":
        args.max_samples = 3000
        args.max_seq_length = 768
        args.learning_rate = 1e-4
        args.quant_calib_samples = 256
        args.quant_iters = 250
        args.output_dir = "./model_final_C"


def to_qa(example: Dict[str, str]) -> Tuple[str, str]:
    if "question" in example and "answer" in example:
        return str(example["question"]).strip(), str(example["answer"]).strip()
    if "problem" in example and "solution" in example:
        return str(example["problem"]).strip(), str(example["solution"]).strip()
    if "instruction" in example and "output" in example:
        return str(example["instruction"]).strip(), str(example["output"]).strip()
    if "prompt" in example and "response" in example:
        return str(example["prompt"]).strip(), str(example["response"]).strip()
    if "input" in example and "output" in example:
        return str(example["input"]).strip(), str(example["output"]).strip()
    raise ValueError(f"Unsupported dataset schema keys: {list(example.keys())}")


def format_example(example: Dict[str, str], tokenizer: AutoTokenizer) -> Dict[str, str]:
    question, answer = to_qa(example)

    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback if template behavior differs by tokenizer version.
        text = f"<|user|>\n{question}\n<|assistant|>\n{answer}"

    return {"text": text}


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100 * trainable / total if total else 0.0
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")


def build_autoround_calib(dataset, tokenizer, max_seq_length: int, max_samples: int):
    calib = dataset.select(range(min(max_samples, len(dataset))))

    def _tok(ex):
        tok = tokenizer(
            ex["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
        }

    calib = calib.map(_tok, remove_columns=calib.column_names)
    calib.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return [calib[i] for i in range(len(calib))]


def build_train_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer):
    primary = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
    mix_name = (args.mix_dataset_name or "").strip()

    if mix_name:
        secondary = load_dataset(mix_name, args.mix_dataset_config, split=args.mix_split)
        primary_n = int(args.max_samples * (1.0 - args.mix_ratio))
        secondary_n = args.max_samples - primary_n
        primary = primary.select(range(min(primary_n, len(primary))))
        secondary = secondary.select(range(min(secondary_n, len(secondary))))

        primary = primary.map(
            lambda ex: format_example(ex, tokenizer),
            remove_columns=primary.column_names,
            desc="Formatting primary dataset",
        )
        secondary = secondary.map(
            lambda ex: format_example(ex, tokenizer),
            remove_columns=secondary.column_names,
            desc="Formatting secondary dataset",
        )
        dataset = concatenate_datasets([primary, secondary]).shuffle(seed=42)
    else:
        dataset = primary.select(range(min(args.max_samples, len(primary))))
        dataset = dataset.map(
            lambda ex: format_example(ex, tokenizer),
            remove_columns=dataset.column_names,
            desc="Formatting dataset",
        )
    return dataset


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QLoRA script (target: single L4).")

    bf16 = torch.cuda.is_bf16_supported()

    print("[1/7] Loading tokenizer from local quantized model directory...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/7] Loading quantized base model from local ./model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("[3/7] Preparing reasoning dataset(s) with EXAONE chat template...")
    dataset = build_train_dataset(args, tokenizer)
    print(f"[INFO] Train samples: {len(dataset)}")

    print("[4/7] Attaching LoRA adapters (EXAONE-safe target modules only)...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_params(model)

    print("[5/7] Building SFTTrainer with L4-safe training settings...")
    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "_trainer_ckpt"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    # TRL API compatibility:
    # - Older versions accept `tokenizer=...`
    # - Newer versions replaced it with `processing_class=...`
    sft_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=dataset,
    )
    if "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    if "dataset_text_field" in sft_params:
        trainer_kwargs["dataset_text_field"] = "text"
    elif "formatting_func" in sft_params:
        trainer_kwargs["formatting_func"] = lambda ex: ex["text"]
    if "max_seq_length" in sft_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "packing" in sft_params:
        trainer_kwargs["packing"] = False

    trainer = SFTTrainer(**trainer_kwargs)

    print("[6/7] Starting QLoRA fine-tuning...")
    trainer.train()

    print("[7/9] Saving LoRA adapter...")
    adapter_dir = os.path.join(args.output_dir, "_lora_adapter")
    trainer.model.save_pretrained(adapter_dir)

    print("[8/9] Re-loading full-precision base model for merge...")
    merge_base = AutoModelForCausalLM.from_pretrained(
        args.merge_base_model_id,
        trust_remote_code=True,
        local_files_only=args.merge_base_local_files_only,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
    )
    merge_base.config.use_cache = False
    merged_model = PeftModel.from_pretrained(merge_base, adapter_dir).merge_and_unload()

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("[9/9] Re-quantizing merged model to auto_gptq and exporting...")
    calib_list = build_autoround_calib(
        dataset=dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        max_samples=args.quant_calib_samples,
    )

    autoround = AutoRound(
        merged_model,
        tokenizer,
        bits=4,
        group_size=128,
        sym=True,
        dataset=calib_list,
        seqlen=args.max_seq_length,
        nsamples=len(calib_list),
        iters=args.quant_iters,
        lr=1e-2,
        minmax_lr=1e-2,
        enable_quanted_input=True,
        enable_minmax_tuning=True,
        batch_size=1,
        gradient_accumulate_steps=8,
        scale_dtype=torch.float32,
    )
    autoround.quantize()
    autoround.save_quantized(
        output_dir=args.output_dir,
        format="auto_gptq",
        inplace=True,
    )
    tokenizer.save_pretrained(args.output_dir)

    print(f"[DONE] Exported merged model + tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
