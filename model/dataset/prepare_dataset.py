import datasets
from datasets import load_dataset, Dataset

def prepare_dataset(ds_id, ds_split, num_train, num_calib):
    """
    데이터셋을 로드하는 함수
    """
    ds = load_dataset(ds_id, split=ds_split)

    # 260208 수정 - 데이터 섞기 추가 및 seed 고정
    ds = ds.shuffle(seed=42)

    # 학습용과 캘리브레이션용 분리
    training_dataset = ds.select(range(num_train))
    calib_dataset = ds.select(range(num_train, num_train + num_calib))

    return training_dataset, calib_dataset


def make_calib_dataset(tokenizer, dataset_id, split, n, seed=42):
    """
    양자화를 위해, calibration dataset 을 구축하는 함수
    """
    try:
        # streaming 사용
        it = load_dataset(dataset_id, split=split, streaming=True)
        it = it.shuffle(seed=seed, buffer_size=10000)
        samples = []
        for ex in it.take(n):
            text = tokenizer.apply_chat_template(
                ex["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
            samples.append({"text": text})
        return Dataset.from_list(samples)
    
    except Exception as e:
        print("스트리밍 경로가 실패하여 직접적으로 데이터셋을 로딩합니다.")
        print(f"\n구체적인 원인은 다음을 참고하세요: {e}")
        
        ds = load_dataset(dataset_id, split=f"{split}[:{n}]")

        def _pp(ex):
            return {
                "text": tokenizer.apply_chat_template(
                    ex["conversations"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            }
        return ds.map(_pp)
