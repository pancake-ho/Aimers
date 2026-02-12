from vllm import LLM, SamplingParams
import os
import time

MODEL_PATH = "./submit/model"

def test_model():
    print(f"[{MODEL_PATH} 모델 로드 테스트 시작...]")

    try:
        print(">>> vLLM 엔진 초기화를 시작합니다...")
        llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,            # 대회 규정: 1
        gpu_memory_utilization=0.85,       # 대회 규정: 0.85
        max_model_len=2048,                # EXAONE 스펙에 맞게 조정 (대회 max_gen_toks 고려)
        dtype="auto",
        trust_remote_code=True,            # EXAONE 모델 로드 시 필수
        enforce_eager=True                 # 구버전 vLLM에서 CUDA 그래프 에러 방지용 (선택)
        )

        print("\n [성공] vLLM 이 모델을 정상적으로 로드했습니다.")
        print("config.json 파일이 호환됩니다. 제출 가능합니다.")

    except Exception as e:
        print("\n [실패] 모델 로드 중 에러가 발생했습니다.")
        print(f"\n 구체적인 에러는 다음과 같습니다: {e}")

    # ==========================================
    # 3. 추론 테스트 (Sampling Params 규정 준수)
    # ==========================================
    # 대회 평가용 프롬프트 예시 (가상의 데이터)
    prompts = [
        "인공지능의 미래에 대해 설명해줘.",
        "Python에서 리스트를 정렬하는 방법을 코드로 보여줘.",
        "LG Aimers 해커톤의 목적은 무엇인가?"
    ]

    # Sampling Params (대회 규정에는 명시되지 않았으나, 평가 시 보통 사용되는 값)
    # max_tokens는 대회 규정의 max_gen_toks=16384 내에서 설정
    sampling_params = SamplingParams(
        temperature=0.0,       # 정량 평가를 위해 보통 0으로 설정 (Greedy Decoding)
        max_tokens=256,        # 테스트용 길이
        stop=None
    )

    print(f"\n>>> 추론 시작 ({len(prompts)} 개 예제)...")
    start_time = time.time()

    outputs = llm.generate(prompts, sampling_params)

    end_time = time.time()

    # ==========================================
    # 4. 결과 출력
    # ==========================================
    print("\n" + "="*50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[질문]: {prompt}")
        print(f"[답변]: {generated_text}")
        print("-" * 50)

    print(f">>> 총 소요 시간: {end_time - start_time:.2f}초")

if __name__ == "__main__":
    test_model()