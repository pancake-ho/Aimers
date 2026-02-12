import torch
from transformers import TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import gc

class Fine_tuning():
    """
    Fine_tuning 클래스
    """
    def __init__(self, 
                 model=None,
                 tokenizer=None,
                 seq_length: int=2048,
                 train_ds=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = seq_length
        self.train_dataset = train_ds

    def setup_lora(self, r: int=16, alpha: int=32, dropout: float=0.05, epochs: int=1, lr: float=2e-4):
        """
        Quantized EXAONE 4.0-1.2B 모델에 대한 Fine-tuning 적용 함수
        """
        print("==========================================")
        print("[Step 1] LoRA Fine-Tuning 시작")
        print("==========================================")

        # LoRA 설정
        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj" # 다 건드림
            ],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # SFT TRAINER 설정
        sft_config = SFTConfig(
            output_dir="./trainer",
            num_train_epochs=epochs, # 보통 1~3
            per_device_train_batch_size=4, # OOM 걸리면 줄이기
            gradient_accumulation_steps=4, # OOM 걸리면 줄이기
            learning_rate=lr,
            logging_steps=10,
            bf16=True,
            optim="adamw_torch",
            save_strategy="no",
            gradient_checkpointing=True,
            max_length=self.max_seq_length,
        )

        def formatting_prompts_func(example):
            text = self.tokenizer.apply_chat_template(
                example["conversations"],
                tokenize=False,
                add_generation_prompt=False
            )
            return text
            
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            peft_config=peft_config,
            args=sft_config,
            processing_class=self.tokenizer,
            formatting_func=formatting_prompts_func
        )

        trainer.train()
        print("[Step 1] 학습 완료, 병합 중...")
        # 학습된 어댑터를 원본 모델에 영구 병합 (AutoRound를 위해 필수)

        self.model = trainer.model.merge_and_unload()

        # 메모리 정리
        del trainer
        torch.cuda.empty_cache()
        gc.collect()

        print("[Step 1] 병합 완료. 이제 모델은 Fine-tuned 상태입니다.")
        return self.model