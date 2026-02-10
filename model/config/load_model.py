from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch

def load_model(model_id):
    """
    모델 및 토크나이저를 로드하는 함수
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    model.gradient_checkpointing_enable()

    return model, tokenizer