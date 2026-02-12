import argparse
import sys
from config import ModelConfig, QuantizationConfig, TuningConfig, load_model
from dataset import prepare_dataset
from tuning import Fine_tuning
from quantizing import AutoRoundquantize
from utils import save

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_train", type=int, default=1000, help="Tuning 에 train 데이터 몇 개 쓸건지")
    p.add_argument("--quant", type=str, choices=['awq', 'autoround'], default='autoround', help="어떤 양자화 방법을 사용할 건지")
    p.add_argument("--num_calib", type=int, default=512, help="양자화 과정에서 calib 데이터 얼마나 쓸 건지")
    
    args = p.parse_args()

    model_id = ModelConfig.MODEL_ID
    dataset_id = ModelConfig.DATASET_ID
    out_dir = ModelConfig.OUT_DIR
    dataset_split = ModelConfig.DATASET_SPLIT

    num_training_samples = args.num_train

    quant_method = args.quant
    num_calib_samples = args.num_calib
    seq_length = QuantizationConfig.MAX_SEQUENCE_LENGTH

    print("1. Pipeline 초기화 중...")

    print("2. 모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model(model_id)
    print("3. 모델 로드 완료 (FP16)")
    
    train_ds, calib_ds = prepare_dataset(dataset_id, dataset_split, num_training_samples, num_calib_samples)
    print("4. 데이터 준비 완료")

    tuner = Fine_tuning(model=model, tokenizer=tokenizer, seq_length=seq_length, train_ds=train_ds)
    model = tuner.setup_lora()

    # 다양한 양자화 설정에 대비
    if quant_method == "autoround":
        quantizer = AutoRoundquantize(model=model, tokenizer=tokenizer, calib_dataset=calib_ds)
    else:
        print("현재 AWQ 양자화 방식은 사용 불가합니다. autoround 방식을 사용하세요.")
        sys.exit(1)

    autoround_wrapper = quantizer.execute()
    print("5. 양자화 완료")

    save(out_dir, autoround_wrapper, tokenizer)

if __name__ == "__main__":
    main()
