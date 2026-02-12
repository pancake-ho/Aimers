from vllm import LLM, SamplingParams
import os

MODEL_PATH = "./model"

def test_model():
    print(f"[{MODEL_PATH} 모델 로드 테스트 시작...]")

    try:
        # vLLM 으로 모델 로드 시도
        llm = LLM(model=MODEL_PATH, quantization="gptq", dtype="half", gpu_memory_utilization=0.9)

        print("\n [성공] vLLM 이 모델을 정상적으로 로드했습니다.")
        print("config.json 파일이 호환됩니다. 제출 가능합니다.")

    except Exception as e:
        print("\n [실패] 모델 로드 중 에러가 발생했습니다.")
        print(f"\n 구체적인 에러는 다음과 같습니다: {e}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"오류: '{MODEL_PATH}' 폴더를 찾을 수 없습니다.")
    else:
        test_model()