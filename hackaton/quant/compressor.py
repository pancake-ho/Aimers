from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_type=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION 