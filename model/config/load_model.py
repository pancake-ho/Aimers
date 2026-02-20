from transformers import AutoModelForCausalLM, TrainingArguments
import torch

from utils import load_tokenizer

def load_model(model_id):
    """
    모델 및 토크나이저를 로드하는 함수
    """
    tokenizer = load_tokenizer(model_id, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    model.gradient_checkpointing_enable()

    return model, tokenizer
