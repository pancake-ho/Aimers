import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from feature import ActivationStats

# CUDA 사용 가능 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 장치: {device}")

# 모델 설정
model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
print(f"모델 로드 중이며, 모델명은 다음과 같습니다: {model_id}")

# dtype 설정
# CUDA 라면, FP16 사용하고 / CPU 라면, FP32 사용
dtype = torch.float16 if device == "cuda" else torch.float32

# 모델 및 tokenizer 로드
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    dtype=dtype, 
    trust_remote_code=True
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

# Hook 준비
hook_manager = ActivationStats()

# 테스트할 레이어 지정 (ex. 첫 번째 Decoder layer 의 Attention 부분)
target_layer = model.model.layers[0].self_attn.q_proj
print(f"Target Layer: {target_layer}")

# Hook 등록
hook_manager.register(target_layer)

# Dummy Input Data 를 이용해서, 임시 테스트 진행
text = "EXAONE is a powerful language model."
inputs = tokenizer(text, return_tensors="pt").to(device)

print("레이어 별 activation 탐색을 진행합니다.")
with torch.no_grad():
    model(**inputs)

# 결과 확인
stats = hook_manager.current_input_stats
print("\n--- 특징 결과 ---")
print(f"Mean: {stats[0]:.4f}")
print(f"Std:  {stats[1]:.4f}")
print(f"Kurtosis: {stats[2]:.4f}")
print(f"Skewness: {stats[3]:.4f}")
print(f"Max Val:  {stats[4]:.4f}")

hook_manager.remove_hooks()