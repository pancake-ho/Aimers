import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from stable_baselines3 import PPO # PPO 사용
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env import LyapunovAWQEnv
from util import get_calib_dataset, get_layer_inputs

def main():
    # 모델 및 device 설정
    model_id = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")

    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device=="cuda" else torch.float32,
        trust_remote_code=True,
        device="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # calibration 데이터셋 준비 및 layer input 캡쳐
    raw_calib_data = get_calib_dataset(tokenizer, n_samples=1, seq_len=128) # 테스트용이라 소량으로 설정해둠
    layer_inputs = get_layer_inputs(model, raw_calib_data, device)

    # Env 생성
    # EXAONE 의 실제 디코더 리스트 추출
    target_layers = model.model.layers

    def make_env():
        env = LyapunovAWQEnv(
            model_layers=target_layers,
            calib_data=layer_inputs,
            target_mse=0.005, # 목표 오차 (수정 필요)
            V = 0.5
        )
        return env

    vec_env = DummyVecEnv([make_env])

    # Agent 설정 (PPO)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        gamma=0.99,
    )
    print("Agent 학습을 시작합니다.")

    # 학습 수행
    model.learn(total_timesteps=5000)
    print("학습이 완료되었습니다.")

    # 모델 저장
    model.save("ppo_awq_agent")
    print("Agent 가 ppo_awq_agent 폴더에 저장되었습니다.")

    # 최적의 policy 확인
    # 학습된 Agent 로 모든 레이어를 훑으면서 어떤 alpha 값을 선택하는 지 확인
    obs = vec_env.reset()
    print("\n--- Alpha Sequence ---")
    for i in range(len(target_layers)):
        action, _ = model.predict(obs, deterministic=True)
        alpha = action[0] * 0.005 # discrete
        print(f"Layer {i}: Alpha = {alpha:.3f}")
        obs, rewards, dones, info = vec_env.step(action)
        if dones[0]:
            break

if __name__ == "__main__":
    main()