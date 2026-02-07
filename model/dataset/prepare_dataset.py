from datasets import load_dataset

def prepare_dataset(ds_id, ds_split, num_train, num_calib):
    """
    데이터셋을 로드하는 함수
    """
    ds = load_dataset(ds_id, split=ds_split)

    # 학습용과 캘리브레이션용 분리
    training_dataset = ds.select(range(num_train))
    calib_dataset = ds.select(range(num_train, num_train + num_calib))

    return training_dataset, calib_dataset