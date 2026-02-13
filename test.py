from vllm import LLM, SamplingParams
import os
import time
import zipfile 
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "submit", "model")
ZIP_FILE_PATH = os.path.join(BASE_DIR, "submit.zip")

def prepare_dir():
    print(">>> 모델 디렉토리 점검 중...")
    
    if os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"[확인] 모델이 이미 {MODEL_PATH} 경로에 준비되어 있습니다.")
        return True
    
    # 압축 파일 존재 여부 확인
    if not os.path.exists(ZIP_FILE_PATH):
        print(f"[오류] 압축 파일이 존재하지 않습니다. 경로를 확인하세요: {ZIP_FILE_PATH}")
        print("\n파일명을 확인하거나 해당 경로에 submit.zip 을 업로드해주세요.")
        return False
    
    try:
        print(f"{ZIP_FILE_PATH} 압축 해제 시작...")
        os.makedirs(MODEL_PATH, exist_ok=True)

        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_PATH)

        # 압축 해제 후 구조 확인
        # 압축 풀었는데 submit/model/다른폴더/config.json 형태가 되는 경우 대응
        subdirs = [d for d in os.listdir(MODEL_PATH) if os.path.isdir(os.path.join(MODEL_PATH, d))]
        if not os.path.exists(os.path.join(MODEL_PATH, "config.json")) and len(subdirs) == 1:
            inner_path = os.path.join(MODEL_PATH, subdirs[0])
            print(f"중첩 구조 감지됨. 파일을 상위 폴더로 이동합니다: {inner_path}")
            for file_name in os.listdir(inner_path):
                shutil.move(os.path.join(inner_path, file_name), os.path.join(MODEL_PATH, file_name))
            os.rmdir(inner_path)
        
        print(f"압축 해제 완료: {MODEL_PATH}")
        return True
    
    except Exception as e:
        print(f"압축 해제 중 다음과 같은 오류가 발생했습니다: {e}")
        return False


def test_model():
    # 실행 전 압축 해제 확인
    if not prepare_dir():
        return
    
    print(f"[{MODEL_PATH} 모델 로드 테스트 시작...]")

    try:
        print(">>> vLLM 엔진 초기화를 시작합니다...")
        llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,            # 대회 규정: 1
        gpu_memory_utilization=0.85,       # 대회 규정: 0.85
        dtype="auto",
        trust_remote_code=True,            # EXAONE 모델 로드 시 필수
        enforce_eager=True                 # 구버전 vLLM에서 CUDA 그래프 에러 방지용 (선택)
        )

        print("\n [성공] vLLM 이 모델을 정상적으로 로드했습니다.")
        print("config.json 파일이 호환됩니다. 제출 가능합니다.")

        # ==========================================
        # 3. 추론 테스트 (Sampling Params 규정 준수)
        # ==========================================
        # 대회 평가용 프롬프트 예시 (가상의 데이터)
        # apply_chat_template=true 환경 모사를 위해 대화형 프맷 사용
        prompts = [
            [{"role": "user", "content": "인공지능의 미래에 대해 설명해줘."}],
            [{"role": "user", "content": "Python에서 리스트를 정렬하는 방법을 코드로 보여줘."}],
            [{"role": "user", "content": "양자화(Quantization) 기술이 모델 경량화에 미치는 영향을 설명해줘."}]
        ]

        # Sampling Params (대회 규정에는 명시되지 않았으나, 평가 시 보통 사용되는 값)
        # max_tokens는 대회 규정의 max_gen_toks=16384 내에서 설정
        sampling_params = SamplingParams(
            temperature=0.0,       # 정량 평가를 위해 보통 0으로 설정 (Greedy Decoding)
            max_tokens=16384,        # 테스트용 길이
            stop=None
        )

        print(f"\n>>> 추론 시작 ({len(prompts)} 개 예제)...")
        outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
        start_time = time.time()
        end_time = time.time()

        # ==========================================
        # 4. 결과 출력
        # ==========================================
        print("\n" + "="*50)
        for output in outputs:
            generated_text = output.outputs[0].text
            print(f"[답변 미리보기]: {generated_text.strip()[:200]} ... (이하 생략)")
            print("-" * 50)

        print(f">>> 총 소요 시간: {end_time - start_time:.2f}초")

    except Exception as e:
        print("\n [실패] 모델 로드 중 에러가 발생했습니다.")
        print(f"\n 구체적인 에러는 다음과 같습니다: {e}")

if __name__ == "__main__":
    test_model()