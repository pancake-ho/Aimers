import os
import torch
import shutil
import argparse
import torch.nn.functional as F

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier


# -----------------------------
# KD Trainer (logits distillation)
# -----------------------------
class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_logits = self.teacher(**inputs).logits

        # shift for causal LM
        student_logits = student_logits[:, :-1, :].contiguous()
        teacher_logits = teacher_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # hard loss
        ce = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # soft loss (token-level KL on answer tokens only)
        T = self.temperature
        log_p = F.log_softmax(student_logits / T, dim=-1)
        q = F.softmax(teacher_logits / T, dim=-1)
        kl_tok = F.kl_div(log_p, q, reduction="none").sum(-1)  # [B, S]

        mask = (shift_labels != -100).float()
        kd = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)

        loss = (1 - self.alpha) * ce + self.alpha * (kd * (T * T))
        return (loss, outputs) if return_outputs else loss


def build_kd_features(tokenizer, example, max_len: int):
    """
    MANTA-1M의 conversations에서 마지막 assistant 응답만 loss에 반영(프롬프트 부분 -100 마스킹)
    """
    messages = example["conversations"]
    # 마지막 assistant turn 찾기
    last_a = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_a = i
            break
    if last_a is None:
        return None

    prompt_msgs = messages[:last_a]  # assistant 직전까지
    answer = messages[last_a]["content"]

    # "assistant 헤더 + 빈 content"까지를 prompt로 보고 마스킹
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )
    full_text = tokenizer.apply_chat_template(
        prompt_msgs + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]

    # prompt 길이가 truncation으로 더 길어질 수 있으니 클램프
    m = min(len(prompt_ids), len(input_ids))
    labels = [-100] * m + input_ids[m:]
    labels = labels[: len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def make_calib_dataset(tokenizer, dataset_id, split, n, seed=42):
    """
    가능하면 앞부분 슬라이스 대신 shuffle을 섞어서 대표성 확보.
    (환경에 따라 streaming이 편한 경우가 많음)
    """
    try:
        it = load_dataset(dataset_id, split=split, streaming=True)
        it = it.shuffle(seed=seed, buffer_size=10_000)
        samples = []
        for ex in it.take(n):
            text = tokenizer.apply_chat_template(
                ex["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
            samples.append({"text": text})
        return Dataset.from_list(samples)
    except Exception:
        # fallback: slice
        ds = load_dataset(dataset_id, split=f"{split}[:{n}]")
        def _pp(ex):
            return {"text": tokenizer.apply_chat_template(ex["conversations"],
                                                        add_generation_prompt=True,
                                                        tokenize=False)}
        return ds.map(_pp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--out_dir", type=str, default="./model")

    # KD
    parser.add_argument("--do_kd", action="store_true")
    parser.add_argument("--kd_out", type=str, default="./kd_ckpt")
    parser.add_argument("--kd_samples", type=int, default=50_000)
    parser.add_argument("--kd_max_len", type=int, default=1024)
    parser.add_argument("--kd_lr", type=float, default=5e-5)
    parser.add_argument("--kd_steps", type=int, default=1500)
    parser.add_argument("--kd_temp", type=float, default=2.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)

    # GPTQ
    parser.add_argument("--dataset_id", type=str, default="LGAI-EXAONE/MANTA-1M")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--calib_samples", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=1024)

    args = parser.parse_args()

    # 1) load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    model_path_for_quant = args.base_model

    # 2) (optional) KD fine-tuning
    if args.do_kd:
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
        student.config.use_cache = False  # 학습 시 메모리 절약
        student.gradient_checkpointing_enable()

        raw = load_dataset(args.dataset_id, split=args.dataset_split)
        raw = raw.shuffle(seed=42).select(range(min(args.kd_samples, len(raw))))

        def _map(ex):
            feat = build_kd_features(tokenizer, ex, args.kd_max_len)
            return feat if feat is not None else {"input_ids": [], "attention_mask": [], "labels": []}

        train_ds = raw.map(_map, remove_columns=raw.column_names)
        # 빈 샘플 제거
        train_ds = train_ds.filter(lambda x: len(x["input_ids"]) > 0)

        def collate(features):
            return tokenizer.pad(features, padding=True, return_tensors="pt")

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
            data_collator=collate,
        )

        trainer.train()
        os.makedirs(args.kd_out, exist_ok=True)
        trainer.model.save_pretrained(args.kd_out, safe_serialization=True)
        tokenizer.save_pretrained(args.kd_out)

        model_path_for_quant = args.kd_out
        print(f"[INFO] KD 완료 → {args.kd_out}")

    # 3) GPTQ (W4A16)
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

    # GPTQModifier 파라미터 튜닝 포인트
    recipe = [
        GPTQModifier(
            scheme="W4A16",
            targets=["Linear"],
            ignore=["re:.*embed_tokens.*", "re:.*lm_head.*"],

            # 아래는 상위권에서 거의 다 만지는 파라미터들
            block_size=128,
            dampening_frac=0.01,
            actorder="weight",        # W4A16 정확도 회복에 유리한 케이스가 많음 :contentReference[oaicite:12]{index=12}
            offload_hessians=False,   # VRAM 부족하면 True로 (느려짐) :contentReference[oaicite:13]{index=13}
        )
    ]

    print(f"[INFO] GPTQ 시작 (calib={args.calib_samples}, max_len={args.max_seq_len})")
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.calib_samples,

        # 데이터 사용 효율 개선(지원 인자) :contentReference[oaicite:14]{index=14}
        concatenate_data=True,
        pad_to_max_length=False,
        shuffle_calibration_samples=True,
    )
    print("[INFO] GPTQ 완료")

    # 4) save + zip
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, save_compressed=True)
    tokenizer.save_pretrained(args.out_dir)

    zip_name = "submit"
    shutil.make_archive(base_name=zip_name, format="zip", root_dir=".", base_dir=args.out_dir)
    print(f"[INFO] 생성 완료: {zip_name}.zip (내부에 model/만 있어야 함)")

if __name__ == "__main__":
    main()
