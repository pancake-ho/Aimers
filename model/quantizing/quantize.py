from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier


class AutoRoundQuantizer:
    def __init__(
        self,
        model,
        tokenizer,
        calib_dataset,
        seq_length=2048,
        bits=4,
        group_size=128,
        sym=False,
        iters=500,
        lr=1e-2,
    ):
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

    def preprocess(self, example):
        if "conversations" in example:
            prompt = self.tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        elif "text" in example:
            prompt = example["text"]
        else:
            return {"input_ids": [], "attention_mask": []}

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.seq_length,
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    def execute(self):
        print("[AutoRound] 캘리브레이션 데이터 변환 중...")

        # 1) dataset 전처리 및 torch format 설정
        calib_ds = self.calib_data.map(
            self.preprocess,
            remove_columns=self.calib_data.column_names,
        )
        calib_ds = calib_ds.filter(lambda x: len(x["input_ids"]) > 0)
        calib_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

        print(f"[AutoRound] 양자화 시작 (Bits: {self.bits}, Group: {self.group_size})")

        # AutoRoundModifier 레시피 정의
        recipe = AutoRoundModifier(
            iters=self.iters,
            ignore=["lm_head"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": self.bits,  # 4
                        "type": "int",
                        "symmetric": self.sym,  # True/False
                        "strategy": "group",
                        "group_size": self.group_size,  # 128
                    },
                }
            },
        )

        oneshot(
            model=self.model,
            dataset=calib_ds,
            recipe=recipe,
            max_seq_length=self.seq_length,
            num_calibration_samples=len(calib_ds),
            shuffle_calibration_samples=False,
        )

        return self.model


# Backward compatibility
GPTQquantize = AutoRoundQuantizer
