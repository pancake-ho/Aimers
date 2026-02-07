import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

class Fine_tuning():
    """
    Fine_tuning 클래스
    """
    def __init__(self, 
                 flag: bool=True, 
                 model: str="LGAI-EXAONE/EXAONE-4.0-1.2B", 
                 tokenizer: str="none", 
                 seq_length: int=2048,
                 train_ds: str="none"):
        self.flag = flag # fine-tuning 사용 여부 (기본값은 true)
        self.model = model
        self.tokenizer = tokenizer
        self.max_seq_length = seq_length
        self.train_dataset = train_ds

    def setup_lora(self, r: int=16, alpha: int=32, dropout: float=0.05, epochs: int=1, lr: float=2e-5):
        """
        Quantized EXAONE 4.0-1.2B 모델에 대한 Fine-tuning 적용 함수
        """
        if self.flag is True:
            print("==========================================")
            print("[Step 1] LoRA Fine-Tuning 시작")
            print("==========================================")

            # LoRA 설정
            peft_config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                lora_dropout=dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )

            # SFT TRAINER 설정
            training_args = TrainingArguments(
                output_dir="./trainer",
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=lr,
                logging_steps=10,
                bf16=True,
                optim="adamw_torch",
                save_strategy="no"
            )

            def formatting_prompts_func(example):
                output_texts = []
                for conversation in example["conversations"]:
                    text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                    output_texts.append(text)
                return output_texts
            
            trainer = SFTTrainer(
                model=self.model,
                train_dataset=self.train_dataset,
                peft_config=peft_config,
                max_seq_length=self.max_seq_length,
                tokenizer=self.tokenizer,
                args=training_args,
                formatting_func=formatting_prompts_func
            )

            trainer.train()
            print("[Step 1] LoRA Fine-Tuning 시작")
            # 학습된 어댑터를 원본 모델에 영구 병합 (AutoRound를 위해 필수)
            model = trainer.model.merge_and_unload()
            print("[Step 1] 병합 완료. 이제 모델은 Fine-tuned 상태입니다.")