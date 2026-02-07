from config import ModelConfig, QuantizationConfig, TuningConfig, load_model
from dataset import prepare_dataset
from tuning import Fine_tuning
from quantizing import AutoRoundquantize
from utils import save

def main():
    model_id = ModelConfig.MODEL_ID
    dataset_id = ModelConfig.DATASET_ID
    out_dir = ModelConfig.OUT_DIR
    dataset_split = ModelConfig.DATASET_SPLIT

    num_training_samples = TuningConfig.TRAINING_SAMPLES
    tuning_flag = TuningConfig.USE_FINE_TUNING

    num_calib_samples = QuantizationConfig.NUM_CALIBRATION_SAMPLES
    seq_length = QuantizationConfig.MAX_SEQUENCE_LENGTH

    print("1. Pipeline 초기화 중...")

    print("2. 모델 및 토크나이저 로드 중...")
    model, tokenizer = load_model(model_id)
    print("3. 모델 로드 완료 (FP16)")
    
    train_ds, calib_ds = prepare_dataset(dataset_id, dataset_split, num_training_samples, num_calib_samples)
    print("4. 데이터 준비 완료")

    tuner = Fine_tuning(flag=tuning_flag, model=model, tokenizer=tokenizer, seq_length=seq_length, train_ds=train_ds)
    model = tuner.setup_lora()

    quantizer = AutoRoundquantize(model=model, tokenizer=tokenizer, calib_dataset=calib_ds)

    autoround_wrapper = quantizer.execute()
    print("5. 양자화 완료")

    save(out_dir, autoround_wrapper, tokenizer)

if __name__ == "__main__":
    main()