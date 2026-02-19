"""
AutoRound 기반 EXAONE 4bit 양자화 스크립트
- 제출 폴더: ./model
- 제출 메타 dtype: float16 강제
"""

import os
import shutil

import torch
from auto_round import AutoRound
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

BITS = 4
GROUP_SIZE = 128
SYMMETRIC = True


print("[AutoRound INFO] 모델/토크나이저 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("[AutoRound INFO] pad_token이 없어 eos_token으로 설정했습니다.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map={"": "cuda"},
)

print("[AutoRound INFO] 캘리브레이션 데이터셋 로딩 중...")
ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)


def preprocess(example):
    prompt = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=True,
        tokenize=False,
    )

    tokenized = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt",
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


ds = ds.map(
    preprocess,
    remove_columns=ds.column_names,
)

ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataset_list = [ds[i] for i in range(len(ds))]

print(f"[AutoRound INFO] 샘플 수: {len(dataset_list)}")
print(f"[AutoRound INFO] 첫 샘플 shape: {dataset_list[0]['input_ids'].shape}")

print(f"[AutoRound INFO] 양자화 시작 (bits={BITS}, group_size={GROUP_SIZE})")
print("[AutoRound INFO] vLLM 호환을 위해 균일 4bit만 사용합니다.")

autoround = AutoRound(
    model,
    tokenizer,
    bits=BITS,
    group_size=GROUP_SIZE,
    sym=SYMMETRIC,
    dataset=dataset_list,
    seq_len=MAX_SEQUENCE_LENGTH,
    n_samples=NUM_CALIBRATION_SAMPLES,
    iters=500,
    lr=1e-2,
    minmax_lr=1e-2,
    enable_quanted_input=True,
    enable_minmax_tuning=True,
    batch_size=1,
    gradient_accumulate_steps=8,
    scale_dtype=torch.float32,
)

autoround.quantize()
print("[AutoRound INFO] 양자화 완료")

os.makedirs(OUT_DIR, exist_ok=True)

# 제출 검증에서 dtype이 bfloat16로 보이지 않도록 강제
model.config.dtype = "float16"
model.config.torch_dtype = "float16"

print("[AutoRound INFO] GPTQ 형식으로 저장 중...")
autoround.save_quantized(
    output_dir=OUT_DIR,
    format="auto_gptq",
    inplace=True,
)

tokenizer.save_pretrained(OUT_DIR)

print(f"[AutoRound INFO] 모델 저장 완료: {OUT_DIR}")

zip_name = "submit"
print(f"[AutoRound INFO] {zip_name}.zip 생성 중...")

shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir=".",
    base_dir=OUT_DIR,
)

print(f"[AutoRound INFO] 생성 완료: {zip_name}.zip")
