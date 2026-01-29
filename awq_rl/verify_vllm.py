import os
import glob
from vllm import LLM, SamplingParams

def get_latest_model_path(base_name="EXAONE-1.2B-RL-AWQ-4bit-*"):
    """
    현 디렉토리에서 base_name 패턴과 일치하는 폴더 중
    가장 최근에 수정 및 생성된 폴더의 경로를 반환하는 함수
    """
    # 패턴에 맞는 모든 폴더 찾기
    candidates = glob.glob(base_name)

    if not candidates:
        raise FileNotFoundError(f"'{base_name}' 패턴의 모델을 찾을 수 없습니다. 양자화 과정을 먼저 수행하세요.")

    # 수정 시간 순으로 정렬하여 "가장 최신" 모델 선택
    latest_model = max(candidates, key=os.path.getmtime)

    print(f"가장 최신 모델 [{latest_model}] 을 감지했습니다. vllm inference 과정에는 이 모델이 사용됩니다.")
    return latest_model


def test_vllm():
    """
    최종 양자화된 모델을 이용해 vllm 환경에서 inference 수행하는 함수
    """
    # 자동으로 최신 경로 가져옴
    try:
        model_path = get_latest_model_path()
        print(f"vLLM 에 {model_path} 모델을 로딩중입니다.")
    except Exception as e:
        print("모델 로딩 중 에러가 발생했습니다.")
        print("\n구체적인 에러 원인은 다음을 참고하세요: {e}")
        exit(1)

    try:
        # vLLM 로드
        llm = LLM(model=model_path, quantization="awq", dtype="half", trust_remote_code=True)
    except Exception as e:
        print(f"vLLM 로드에 실패했습니다.")
        print("\n구체적인 에러 원인은 다음을 참고하세요: {e}")
        return
    
    prompts = [
        "인공지능의 미래에 대해 설명해줘.",
        "경희대학교 전자공학과에 대해 설명해줘.",
        "Explain the concept of Reinforcement Learning",
    ]

    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=200)

    print("답변을 생성 중입니다.")
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n[Prompt]: {prompt}")
        print(f"\n[Output]: {generated_text}")
        print("-" * 50)

if __name__ == "__main__":
    test_vllm()