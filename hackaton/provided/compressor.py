from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot # 양자화 프로세스 실행 함수
from llmcompressor.modifiers.quantization import GPTQModifier

import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false" # 토크나이저 병렬 처리 기능 비활성화 / 멀티프로세싱 문제 방지 목적

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_complete( # 대화 탬플릿에 맞게 원본 데이터 변환
            example["conversations"],
            add_generation_prompt=True,
            tokenizer=False
        )
    }

ds = ds.map(preprocess)

recipe = [GPTQModifier(ignore=["embed_tokens", "lm_head"], scheme="W4A16", targets=["Linear"])] # 모든 Linear 레이어에 양자화
 
oneshot( # 별도의 Fine-Tuning 없이, 한 번의 최적화 과정으로 양자화 과정 수행
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("=========== SAMPLE GENERATION ===========")
message = [{"role": "user", "content": "Who are you?"}]
input_ids = tokenizer.apply_chat_template(message, add_generation_template=True, enable_thinking=False, return_tensors="pt").to(model.device)
output = model.generate(input_ids, max_new_tokens=100, do_sample=False)
print(tokenizer.decode(output[0]))
print("========================================\n\n")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)