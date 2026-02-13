# import os
# import sys
# import shutil
# import torch
# from auto_round import AutoRound

# try:
#     # awq 대체 라이브러리
#     from compressed_tensors import (
#         QuantizationConfig,
#         QuantizationStatus,
#         ModelCompressor,
#         CompressionFormat
#     )

# except ImportError as e:
#     print("compressed_tensors 라이브러리를 불러오지 못했습니다.\n")
#     print("다음 에러를 확인하고, 라이브러리 설치 혹은 타 오류를 확인하세요: {e}")
#     sys.exit(1)

from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier

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

        # 1) dataset 전처리 및 torch format 설정
        calib_ds = self.calib_data.map(
            self.preprocess_autoround, 
            remove_columns=self.calib_data.column_names,
        )
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
                        "num_bits": self.bits,          # 4
                        "type": "int",
                        "symmetric": self.sym,          # True/False
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
    

# class CompressedTensorWrapper:
#     """
#     CompressedTensor (유사 AWQ) 모델을 /utils 의 save 인터페이스와 호환되도록 감싸는 클래스
#     """
#     def __init__(self, model):
#         self.model = model

#     def save_quantized(self, out_dir, format=None, inplace=True):
#         # utils.save 함수는 "auto_gptq" 사용하지만 AWQ 는 이와 다르므로 무시하게 함
#         print(f"[Compressed-Tensors(AWQ)] 모델 저장 중... (out_dir: {out_dir})")
#         self.model.save_pretrained(out_dir)


# 호환성 문제로 인해, 일단은 AWQ 제외
# class AWQquantize():
#     def __init__(self, model, tokenizer, calib_dataset, seq_length=2048,
#                  bits=4, group_size=128, version="GEMM"):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.calib_data = calib_dataset
#         self.seq_length = seq_length
#         self.bits = bits
#         self.group_size = group_size

#     def execute(self):
#         print("[Compressed-Tensors] AWQ 스타일 양자화 설정 적용 중...")

#         # Quantization Config 생성
#         # int4, group size 128, asymmetric 적용
#         config = QuantizationConfig(
#             config_groups={
#                 "group_0": {
#                     "weights": {
#                         "num_bits": self.bits, # 4비트
#                         "type": "int",
#                         "strategy": "group",
#                         "group_size": self.group_size,
#                         "symmetric": False,
#                         "actorder": False,
#                     },
#                     "targets": ["Linear"] # 모든 linear 레이어에 적용
#                 }
#             },
#             quant_method="compressed-tensors",
#             format="pack-quantized",
#             ignore=["lm_head"]
#         )

#         print("[Compressed-Tensors] 양자화 수행 중...")

#         # 압축은 ModelCompressor 로 수행
#         compressor = ModelCompressor(quantization_config=config)
#         compressed_state_dict = compressor.compress(self.model)

#         self.model.load_state_dict(compressed_state_dict, strict=False)

#         print("[Compressed-Tensors] 양자화 완료.")
#         return CompressedTensorWrapper(self.model)