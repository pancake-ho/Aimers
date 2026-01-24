import os
import sys
from dotenv import load_dotenv
from friendli import Friendli

# 환경 변수 로드
load_dotenv()

def get_exaone_response(prompt: str) -> str:
    """
    K-EXAONE 모델에 프롬프트를 전송하고 응답을 반환하는 함수
    """
    token = os.getenv("FRIENDLI_TOKEN")
    if not token:
        print("에러 발생: .env 파일에 토큰이 설정되지 않았습니다.")
        sys.exit(1)
    
    client = Friendli(token=token)
    try:
        # API 호출
        model_id = "LGAI-EXAONE/K-EXAONE-236B-A23B"
        print("[K-EXAONE] 모델에 질의 중...\n")

        chat_completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 LG AI Research 에서 개발한 EXAONE 입니다. 유용하고 안전한 답변을 제공하세요."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=False, # 스트리밍 없이, 전체 응답을 한 번에 받음
            # temperature=0.7, # 창의성 조절 (0 ~ 1)
            max_tokens=1024 # 전체 응답 길이
        )

        response_text = chat_completion.choices[0].message.content
        return response_text

    except Exception as e:
        return f"API 호출 중 오류 발생, 에러 메시지를 확인하세요: {e}"

if __name__ == "__main__":
    user_question = "경희대학교 전자공학과에 대해 설명해줘."

    answer = get_exaone_response(user_question)

    print("-" * 50)
    print(f"Q: {user_question}")
    print("-" * 50)
    print(f"A: {answer}")
    print("-" * 50)