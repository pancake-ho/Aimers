import torch
import json
import numpy as np

from stable_baselines3 import PPO
from transformers import AutoModelForCausalLM, AutoTokenizer
from env import LyapunovAWQEnv
from util import get_calib_dataset, get_layer_inputs

def extract_alpha():
    # 장치 설정 및 모델 로드
    model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"모델을 로드합니다. 사용 장치: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device=="cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Calibration 데이터셋 준비
    calib_data = get_calib_dataset(tokenizer, n_samples=1, seq_len=128)
    layer_inputs = get_layer_inputs(model, calib_data, device)

    target_layers = model.model.layers

    # env 및 agent 로드
    env = LyapunovAWQEnv(target_layers, layer_inputs, target_mse=0.005, V=0.5)
    
    print("훈련된 PPO Agent 를 로드합니다.")
    try:
        model = PPO.load("ppo_awq_agent")
    except Exception as e:
        print("Agent 로드에 실패했습니다.")
        print("구체적인 에러 원인은 다음을 참고하세요: {e}")
        return
    
    # 최적의 alpha 값 추출
    print("\n--- 최적의 alpha 값 추출을 진행합니다 ---")
    obs, _ = env.reset()
    best_alphas = {}

    for i in range(len(target_layers)):
        # 가장 확률이 높은 행동 선택
        action, _ = model.predict(obs, deterministic=True)
        alpha_value = float(action[0] * 0.05)

        layer_name = f"model.layers.{i}" # 실제 EXAONE 레이어 이름을 따름
        best_alphas[layer_name] = alpha_value
        
        print(f"[Layer {i}] 선택된 alpha 값: {alpha_value:.3f}")

        # env step
        obs, rewards, dones, truncated, info = env.step(action)
        if dones:
            break

    # 파일 저장
    with open("best_alphas.json", "w") as f:
        json.dump(best_alphas, f, indent=4)
    print("best_alphas.json 파일에 최적의 alpha 값들 저장을 완료했습니다.")

if __name__ == "__main__":
    extract_alpha()