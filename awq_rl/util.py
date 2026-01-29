import functools
import torch
import torch.nn as nn
from datasets import load_dataset

def get_calib_dataset(tokenizer, n_samples: int=128, seq_len: int=512):
    """
    데이터셋을 로드해서 토크나이징된 텐서로 변환하는 함수
    현재는, HuggingFace wikitext 데이터셋을 로드해서 반환하고 있는 상태임
    (AWQ 논문에서는 일반적인 텍스트 데이터를 사용)
    """
    print("Calibration Dataset 을 로드 중이며, 현재는 wikitext-2 데이터셋을 사용합니다.")
    data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    encodings = tokenizer("\n\n".join(data['text']), return_tensors='pt')

    # 샘플 개수(n_samples) 만큼 seq_len 길이로 자르는 과정
    batch_input_ids = []
    for i in range(0, encodings.input_ids.size(1), seq_len):
        if len(batch_input_ids) >= n_samples:
            pass
        chunk = encodings.input_ids[:, i : i + seq_len]
        if chunk.size(1) == seq_len:
            batch_input_ids.append(chunk)
    
    # 데이터셋이 부족할 경우 or 로드 실패할 경우 대비
    if not batch_input_ids:
        print("데이터셋 로드 실패 또는 데이터 부족으로, 더미 데이터를 사용합니다.")
        return [torch.randint(0, 1000, (1, seq_len)) for _ in range(n_samples)]
    
    return batch_input_ids

def get_layer_inputs(model, input_ids_list, device):
    """
    Inference (Forward Hook) 과정을 통해, 각 Decoder Layer 에 들어가는 입력값을 캡쳐하는 함수
    """
    print("모델 레이어의 입력값 캡쳐를 시작합니다.")
    layers = model.model.layers

    # 각 레이어의 입력 저장할 딕셔너리 선언
    layer_inputs = {i: [] for i in range(len(layers))}
    handles = []

    # Hook 함수 정의
    def hook_fn(module, input, output, layer_idx):
        """
        - input: (tensor, ) shape 의 튜플로 들어옴
        """
        # 메모리 절약을 위해, 첫 번쨰 배치만 캡쳐한다고 가정
        if len(layer_inputs[layer_idx]) == 0:
            hidden_states = input[0].detach().cpu()
            layer_inputs[layer_idx].append(hidden_states)
    
    # 모든 레이어에 Hook 등록
    for i, layer in enumerate(layers):
        # functools.partial 사용하여 loop 변수 i 고정
        handle = layer.register_forward_hook(
            functools.partial(hook_fn, layer_idx=i)
        )
        handles.append(handle)

    # Forward pass 실행
    print("Forward Pass 과정 수행을 시작합니다.")
    try:
        with torch.no_grad():
            sample = input_ids_list[0].to(device)
            model(sample)
    except Exception as e:
        print("Forward Pass 과정 중 에러가 발생했습니다.")
        print("구체적인 에러 원인은 다음을 참고하세요: {e}")
    finally:
        # hook 제거
        for handle in handles:
            handle.remove()
        print("Hook 제거 완료")
    
    # 결과 정리 단계
    final_inputs = [layer_inputs[i][0] for i in range(len(layers))]
    return final_inputs