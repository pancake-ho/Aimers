from dataclasses import dataclass

@dataclass
class ModelConfig:
    MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
    OUT_DIR = "./model"
    DATASET_ID = "LGAI-EXAONE/MANTA-1M"
    DATASET_SPLIT = "train"

@dataclass
class TuningConfig:
    USE_FINE_TUNING = True
    TRAINING_SAMPLES = 1000

@dataclass
class QuantizationConfig:
    NUM_CALIBRATION_SAMPLES=512
    MAX_SEQUENCE_LENGTH = 2048