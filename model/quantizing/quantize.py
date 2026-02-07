from auto_round import AutoRound

class AutoRoundquantize():
    def __init__(self, model, tokenizer, calib_dataset, seq_length=2048, 
                 bits=4, group_size=128, sym=False, iters=500, lr=1e-2):
        self.model = model
        self.tokenizer = tokenizer
        self.calib_data = calib_dataset
        self.seq_length = seq_length
        
        # 양자화 하이퍼파라미터 저장
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.iters = iters
        self.lr = lr
    
    def preprocess_autoround(self, example):
        prompt = self.tokenizer.apply_chat_template(
            example["conversations"],
            add_generation_prompt=True,
            tokenize=False
        )
        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        
    def execute(self):
        print("[AutoRound] 캘리브레이션 데이터 변환 중...")

        calib_ds = self.calib_data.map(self.preprocess_autoround, remove_columns=self.calib_data.column_names)
        calib_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset_list = [calib_ds[i] for i in range(len(calib_ds))]

        print(f"[AutoRound] 양자화 시작 (Bits: {self.bits}, Group: {self.group_size})")

        autoround = AutoRound(
            model=self.model,
            tokenizer=self.tokenizer,
            bits=self.bits,
            group_size=self.group_size,
            sym=self.sym,
            dataset=dataset_list,
            seq_len=self.seq_length,
            n_samples=len(dataset_list), # 실제 데이터 개수만큼 사용
            iters=self.iters,
            lr=self.lr,
            minmax_lr=self.lr,
            enable_quanted_input=True,
            enable_minmax_tuning=True,
            batch_size=1,
            gradient_accumulate_steps=8,
        )
        autoround.quantize()
        
        return autoround