import os
import torch
import shutil
import argparse
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

from dataset import load_dataset, make_calib_dataset
from tuning import build_kd_features, KDTrainer, DataCollatorForCausalLM
from utils import save

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--out_dir", type=str, default="./model")

    parser.add_argument("--do_kd", action="store_true")
    parser.add_argument("--kd_out", type=str, default="./kd_ckpt")
    parser.add_argument("--kd_samples", type=int, default=50_000)
    parser.add_argument("--kd_max_len", type=int, default=1024)
    parser.add_argument("--kd_lr", type=float, default=5e-5)
    parser.add_argument("--kd_steps", type=int, default=1500)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)

    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--calib_samples", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    return parser.parse_args()


def run_kd_if_enabled(args, tokenizer):
    model_path_for_quant = args.base_model
    if not args.do_kd:
        return model_path_for_quant

    print("[INFO] KD 시작: teacher=FP16/BF16, student=tunable")

    teacher = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    student = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable()

    raw = load_dataset(args.dataset_id, split=args.dataset_split)
    raw = raw.shuffle(seed=42).select(range(min(args.kd_samples, len(raw))))

    def _map(ex):
        feat = build_kd_features(tokenizer, ex, args.kd_max_len)
        if feat is not None:
            return feat
        return {"input_ids": [], "attention_mask": [], "labels": []}

    train_ds = raw.map(_map, remove_columns=raw.column_names)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)

    targs = TrainingArguments(
        output_dir=args.kd_out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=args.kd_steps,
        learning_rate=args.kd_lr,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        report_to=[],
    )

    trainer = KDTrainer(
        model=student,
        teacher_model=teacher,
        temperature=args.kd_temp,
        alpha=args.kd_alpha,
        args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForCausalLM(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()
    os.makedirs(args.kd_out, exist_ok=True)
    trainer.model.save_pretrained(args.kd_out, safe_serialization=True)
    tokenizer.save_pretrained(args.kd_out)

    model_path_for_quant = args.kd_out
    print(f"[INFO] KD 완료 -> {args.kd_out}")
    return model_path_for_quant


def run_gptq(args, tokenizer, model_path_for_quant):
    print("[INFO] GPTQ용 모델 로드 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_for_quant,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("[INFO] 캘리브레이션 데이터 준비 중...")
    calib_ds = make_calib_dataset(
        tokenizer=tokenizer,
        dataset_id=args.dataset_id,
        split=args.dataset_split,
        n=args.calib_samples,
        seed=42,
    )

    recipe = [
        GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["re:.*embed_tokens.*", "re:.*lm_head.*"],
            block_size=128,
            dampening_frac=0.01,
            actorder="weight",
            offload_hessians=False,
        )
    ]

    print(f"[INFO] GPTQ 시작 (calib={args.calib_samples}, max_len={args.max_seq_len})")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,
        concatenate_data=True,
        pad_to_max_length=False,
        shuffle_calibration_samples=True,
    )
    print("[INFO] GPTQ 완료")
    return model


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_path_for_quant = run_kd_if_enabled(args, tokenizer)
    model = run_gptq(args, tokenizer, model_path_for_quant)
    save(args.out_dir, model, tokenizer)


if __name__ == "__main__":
    main()
