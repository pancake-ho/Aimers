import optuna
import os
import shutil
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from dataset import get_calib_dataset

# 모델 ID 및 설정
MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
TARGET_SIZE_GB = 1.2 # 목표 용량 (일단 임의 설정)


def quantize(trial):
    """
    Optuna 라이브러리를 이용하여, 하이퍼파라미터 / 레이어 별 보호 정책을 탐색하는 함수
    다음의 논리 흐름도를 따름.
    
    1) Optuna 가 파라미터 제안
    2) LLM Compressor 로 양자화 수행
    3) 결과 모델 저장 및 크기 확인
    4) PPL 로 1차 성능 평가
    """
    # 1. 하이퍼파라미터 탐색 공간 정의
    # group_size: 작을수록 높은 정확도/큰 용량
    # damp_percent: 양자화 계산 (Hessian 역행렬) 시 안정화 주는 비율, 작으면 과적합/크면 정보 손실 발생
    # desc_act: Activation 크기 기반 정렬 여부
    group_size = trial.suggest_categorical("group_size", [32, 64, 128]) # 이 범위에서 하나 골라줘
    damp_percent = trial.suggest_float("damp_percent", 0.01, 0.1)
    desc_act = trial.suggest_categorical("desc_act", [True, False])
    math_ratio = trial.suggest_float("math_ratio", 0.1, 0.5)

    print(f"\n[Trial {trial.number}] Params: {trial.params}")

    # 2. 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # 3. 데이터 준비
    dataset = get_calib_dataset(MODEL_ID, n_samples=512, math_ratio=math_ratio)

    # 4. 양자화 레시피 설정
    # 기본적인 로직만 구현했고, 추가적으로 민감한 레이어 보호 로직을 구현할 수 있을 것 같음
    recipe = GPTQModifier(
        ignore=["embed_tokens", "lm_head"], # 임베딩과 헤드는 보호 / 여기에 추가 가능
        scheme="W4A16", # Weight 는 4bit, Activation 은 16bit
        targets=["Linear"], # Linear 레이어 대상
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=desc_act
    )

    # 5. 양자화 수행
    save_dir = f"./trials/trial_{trial.number}"
    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=512
    )

    model.save_pretrained(save_dir, save_compressed=True)
    tokenizer.save_pretrained(save_dir)W