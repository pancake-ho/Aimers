from datasets import load_dataset, concatenate_datasets

def prepare_dataset(ds_id, ds_split, num_train, num_calib):
    """
    기존 해커톤의 MANTA-1M 데이터셋을 기본적으로 로드하고,
    영어와 한국어 데이터를 추가적으로 혼합해 로드하는 함수
    """
    # 1. 기존 MANTA-1M 데이터 로드 및 데이터 shuffle & 일부만 샘플링
    print(f"[Dataset] 1. Main Dataset {ds_id} 로드 중...")
    main_ds = load_dataset(ds_id, split=ds_split)
    main_ds = main_ds.shuffle(seed=42)
    training_dataset = main_ds.select(range(num_train))

    # 2. 양자화용 Calibration 데이터 구성
    # 여기에서는 영어와 한국어 반반 섞음
    print(f"[Dataset] 2. Calibration Dataset 을 위해 영어 및 한국어 데이터 섞는 중...")

    # 영어 (allenai/c4 데이터 사용)
    eng_ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    eng_ds_iter = iter(eng_ds)
    eng_samples = [next(eng_ds_iter) for _ in range(num_calib // 2)]

    # 한국어 (lcw99/wikipedia-korean-20240501 데이터 사용)
    kor_ds = load_dataset("lcw99/wikipedia-korean-20240501", split="train", streaming=True) # 얘는 train split 밖에 없음..
    kor_ds_iter = iter(kor_ds)
    kor_samples = [next(kor_ds_iter) for _ in range(num_calib // 2)]