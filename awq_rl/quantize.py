import json
import torch
import time
from datetime import datetime
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# AutoAWQ 의 Scale 검색 함수를 RL Agent 가 뽑아낸 RL Alpha 값으로 덮어쓸거임 (monkey patch)
from awq.quantize.quantizer import AwqQuantizer

# config (alpha, group size) 를 저장할 전역 변수 선언
RL_CONFIGS = {}

def load_configs():
    """
    RL Agent Inference 과정으로 구한 최적의 config 값들을 로드하는 함수
    """
    global RL_CONFIGS
    try:
        with open("best_configs.json", "r") as f:
            RL_CONFIGS = json.load(f)
        print("RL Config 파일을 로드 완료했습니다.")
    except Exception as e:
        print("best_configs.json 파일을 찾을 수 없습니다.")
        print("구체적인 에러 원인은 다음을 참고하세요: {e}")
        exit(1)

def rl_search_best_scale(self, module, name, input_feat):
    """
    기존 AutoAWQ 의 _search_best_scale 메서드를 대체할 함수
    - module: 현재 처리 중인 Linear layer (추가 가능)
    - name: Layer 이름
    """
    # 현재 레이어가 몇 번째 블록(레이어 idx) 에 속하는 지 파싱
    # ex. "model.layers.0.self_attn_q_proj"
    try:
        # 이름에서 숫자 추출
        parts = name.split('.')
        layer_idx = -1
        for p in parts:
            if p.isdigit():
                layer_idx = int(p)
                break
        
        # RL Inference 로 구한 alpha 조회
        key = f"model.layers.{layer_idx}"
        best_alpha = 1.0

        if layer_idx != -1 and key in RL_CONFIGS:
            config = RL_CONFIGS[key]
            best_alpha = config["alpha"]
            print(f"{name} 레이어에 대하여 {best_alpha} 값의 RL alpha 를 사용합니다.")
    
    except Exception as e:
        print(f"{name} 레이어에 대하여 RL alpha 값을 찾을 수가 없습니다. default alpha (value: 1) 값을 사용합니다.")
        print("구체적인 에러 원인은 다음을 참고하세요: {e}")
        best_alpha = 1.0
    
    # AWQ 공식 적용 (s = s_X ^ alpha)
    # 여기에서는 input_feat['abs_mean'] 이 s_X 에 해당함
    s_x = input_feat["abs_mean"]

    # scale 계산
    best_scales = torch.pow(s_x, best_alpha).clamp(min=1e-5)

    # Normalization (AWQ 표준)
    best_scales = best_scales / (best_scales.max() * best_scales.min()).sqrt()

    return best_scales, best_alpha

# monkey patch
AwqQuantizer._search_best_scale = rl_search_best_scale

def run_quantization():
    """
    최종 양자화 수행 함수
    """
    load_configs()

    current_time = datetime.now().strftime("%Y%n%d_%H%M")
    model_path = f"LGAI-EXAONE/EXAONE-4.0-1.2B"

    quant_path = f"EXAONE-1.2B-RL-AWQ-mixed-{current_time}"
    quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

    print("양자화 실행을 위한 모델 로딩을 수행합니다.")
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        safetensors=True,
        strict=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, train_remote_code=True)

    print("RL Alpha 값을 이용해 양자화 과정 수행을 시작합니다.")
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        # Calib data 는 AutoAWQ 가 알아서 로드해주거나, 지정 가능
        # 여기서는 간단한 과정을 위해 기본값 32 사용
        n_parallel_calib_samples=32
    )

    print(f"양자화된 모델은 {quant_path} 에 저장됩니다.")
    try:
        model.save_quantized(quant_path)
        tokenizer.save_quantized(quant_path)
        print("양자화 과정이 완료되었습니다.")  
    except Exception as e:
        print("양자화된 모델 저장 중 에러가 발생했습니다.")
        print("\n구체적인 에러 원인은 다음을 참고하세요: {e}")
    
if __name__ == "__main__":
    run_quantization()