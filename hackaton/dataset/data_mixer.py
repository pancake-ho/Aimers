import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def get_calib_dataset(tokenizer_path: str, n_samples: int=512, ratio: float=0.3):
    """
    일반 대화 데이터셋(MANTA-1M) 과 수학 데이터(GSM8K) 를 혼합하여 Calibration 데이터셋을 만드는 함수
    """
    print(f"Loading Datasets... (Ratio: {ratio})")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # 1. 기본 데이터셋 (MANTA-1M) 으로 일반적인 언어 능력 유지
    ds_general = load_dataset("LGAI-EXAONE/MANTA-1M", split="train", streaming=True)

    # 2. 타겟 데이터셋(GSM8K) 으로 수학 추론 능력 강화
    ds_math = load_dataset("gsm8k", "main", split="train", streaming=True)

    n_math = int(n_samples * ratio) # 수학 데이터 몇개쓸래?
    n_general = n_samples - n_math # 일반 데이터 몇개쓸래

    # 3. 데이터 섞기
    mixed_samples = []

    iter_general = iter(ds_general)
    for _ in range(int(n_samples * 0.8)): # 비율은 여기 조정하면 됨
        sample = next(iter_general)
        # 채팅 탬플릿 적용
        text = tokenizer.apply_chat_template(sample["conversations"], tokenize=False)
        mixed_samples.append(text)
    
    iter_math = iter(ds_math)
    for _ in range(int(n_samples * 0.2)): # 비율은 여기 조정하면 됨
        sample = next(iter_math)
        # GSM8K 데이터는 question / answer 로 구성됨. 따라서, EXAONE 채팅 포맷으로 변환해야 함
        chat = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]}
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        mixed_samples.append(text)
    
    print(f"Prepared {len(mixed_samples)} samples (General: {n_general}, Math: {n_math})")
    return mixed_samples