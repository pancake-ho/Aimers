"""
0207: AutoRound 기반 양자화 방식 시도
터미널에서, 다음과 같은 사전 준비를 거쳐야 함

pip install auto-round auto-gptq optimum
"""


import os
import torch
import shutil
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# [AutoRound] llmcompressor 대체
from auto_round import AutoRound

# 일단 주석처리
# from llmcompressor import oneshot
# from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"     
OUT_DIR  = "./model_q4_train"          

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

# [AutoRound] 1.2B 소형 모델의 성능 확보를 위해 튜닝 샘플 및 길이 확보
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048 # 논문 반영

# [AutoRound] Uniform 4-bit 설정
# SignRoundv2 논문에서는 mixed-precision 사용하지만, vLLM 호환성을 위해 4-bit 고정
BITS = 4
GROUP_SIZE = 128
SYMMETRIC = True

print("[AutoRound INFO] 모델 로드 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# [안전 장치] 패딩 토큰이 없으면 EOS 토큰을 패딩으로 사용 (매우 중요)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("[AutoRound INFO] 패딩 토큰이 없어 EOS 토큰을 패딩으로 설정했습니다.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map={"": "cuda"}
)

print("[AutoRound INFO] 모델/토크나이저 로드 완료")

print("[AutoRound INFO] 캘리브레이션 데이터 로드 중...")

ds = load_dataset(
    DATASET_ID,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
)

def preprocess(example):
    # 1. 채팅 템플릿을 적용하여 텍스트 생성 (tokenize=False)
    prompt = tokenizer.apply_chat_template(
        example["conversations"],
        add_generation_prompt=True,
        tokenize=False
    )
    
    # 2. 토큰화 (패딩 적용)
    # [핵심 수정] padding="max_length"로 설정하여 모든 데이터를 2048 길이로 맞춤
    # 이렇게 해야 AutoRound가 "길이가 짧다"며 데이터를 버리지 않음
    tokenized = tokenizer(
        prompt,
        padding="max_length", 
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        return_tensors="pt"
    )
    
    # tokenizer 결과는 배치 차원(batch dim)이 포함되므로 [0]으로 인덱싱하여 단일 샘플로 만듦
    return {
        "input_ids": tokenized["input_ids"], 
        "attention_mask": tokenized["attention_mask"]
    }

# 3. remove_columns를 사용하여 'conversations' 등 원본 데이터 제거 (메모리 절약 및 에러 방지)
ds = ds.map(
    preprocess,
    remove_columns=ds.column_names  # 기존 컬럼 모두 제거하고 input_ids, attention_mask만 남김
)

# [핵심 수정 3] 데이터셋을 HuggingFace Dataset 객체가 아닌 'Python List'로 변환
# AutoRound와 HF Dataset 간의 복잡한 배칭(Batching) 충돌을 원천 차단
print("[AutoRound INFO] 데이터셋을 메모리 리스트로 변환 중...")
ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
dataset_list = [ds[i] for i in range(len(ds))]

# 형상 확인 (이제 [1, 2048]이어야 함)
print(f"[AutoRound INFO] 변환 완료. 샘플 개수: {len(dataset_list)}")
print(f"[AutoRound INFO] 첫 번째 샘플 형상: {dataset_list[0]['input_ids'].shape}")

if dataset_list[0]['input_ids'].shape[-1] < MAX_SEQUENCE_LENGTH:
    print("[AutoRound WARNING] 데이터 길이가 설정보다 짧습니다. 패딩 설정 확인 필요!")
else:
    print("[AutoRound INFO] 데이터 길이 검증 완료 (OK).")

print(f"[AutoRound INFO] SignRoundV2 시작 (Bits={BITS}, Group size={GROUP_SIZE})...")

# [AutoRound] 실행 설정
# iters = 500: 기본은 200이나, 소형 모델의 특성을 고려하여 늘림
# n_samples = 512: 충분한 데이터 사용
# enable_minmax_tuning = True: Clipping Scale 최적화 활용
autoround = AutoRound(
    model,
    tokenizer,
    bits=BITS,
    group_size=GROUP_SIZE,
    sym=SYMMETRIC,
    dataset=dataset_list, # 리스트 전달
    seq_len=MAX_SEQUENCE_LENGTH,
    n_samples=NUM_CALIBRATION_SAMPLES,
    iters=500, 
    lr=1e-2,
    minmax_lr=1e-2,
    enable_quanted_input=True,
    enable_minmax_tuning=True,
    batch_size=1, # [핵심 수정 4] 배치 사이즈를 1로 명시하여 차원 꼬임 방지
    gradient_accumulate_steps=8,
    scale_dtype=torch.float32,
)

# [AutoRound] 튜닝 시작
autoround.quantize()

print("[AutoRound INFO] SignRound 튜닝 완료")

os.makedirs(OUT_DIR, exist_ok=True)

# [AutoRound] vLLM 호환성을 위해, GPTQ 포맷으로 내보냄
# "auto_gptq" 포맷 써서 해결
print(f"[AutoRound INFO] GPTQ 호환 모델 저장 중...")
autoround.save_quantized(
    output_dir=OUT_DIR,
    format="auto_gptq",
    inplace=True,
)

# [AutoRound] 토크나이저는 별도로 저장
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
