import torch
import torch.nn as nn

class AWQSearcher:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    def get_scale(self, x):
        """
        Input Activation 으로 주어지는 x 로부터 채널별 magnitude 추출하는 함수
        """
        # x shape: [batch, seq_len, hidden_dim] -> [hidden_dim] 변환
        mag = x.abs().view(-1, x.shape[-1]).mean(0)
        return mag
    
    def awq_scaling(self, layer, x, alpha):
        """
        주어진 alpha 값을 이용해 Weight 값과 Input 값을 scaling 하는 함수
        (s = S_x ^ alpha)
        """
        scale = self.get_scale(x)
        scale_clamp = (scale.pow(alpha)).clamp(min=1e-5)
        scale_norm = scale / (scale_clamp.max() * scale_clamp.min()).sqrt()
        return scale_norm
    
    def quantization_error(self, layer: nn.Linear, x: torch.Tensor, alpha: float, bit=4, group_size=128):
        """
        alpha 를 적용하여 quantization, MSE Error 를 측정하는 함수
        또한, Group Size 를 반영하여 시뮬레이션 수행
        
        - layer: 최적화 수행하려고 하는 EXAONE 의 layer
        - x: Calibration 데이터셋에서 추출한 해당 레이어의 Activation
        - alpha: 0.0 ~ 1.0 사이의 값으로, Weight 보호 강도를 조절
        """
        w = layer.weight.data.clone().float() # 현재 layer의 weight을 복제하고, 정밀한 계산을 위해 FP32로 변환
        
        # Scaling vector s 계산
        # s = (s_x)^alpha
        scale = self.awq_scaling(layer, x, alpha).to(w.device)

        # Weight Scaling (AWQ 양자화 적용 부분 - Salient Weight을 size-up)
        # 원본 weight W 에 계산된 scale s 를 곱하는 과정
        w_scaled = w * scale.view(1, -1)

        # Group-wise Reshape
        org_shape = w_scaled.shape
        if group_size > org_shape[1] % group_size == 0:
            w_reshaped = w_scaled.reshape(org_shape[0], -1, group_size)
        else:
            w_reshaped = w_scaled.reshape(org_shape[0], -1, 1)        

        # Psuedo-Quantization (W4A16 시뮬레이션)
        # 실제로는 압축하지 않았는데, 만약 압축하면 데이터가 얼마나 망가질지를 측정하는 부분
        max_val = w_reshaped.abs().amax(dim=1, keepdim=True) # 각 행별로, 절댓값이 가장 큰 값을 찾는 과정 (양자화 범위 결정 목적)
        quant_scale = max_val / (2**(bit-1) - 1) # step-size 계산
        
        # w_quant 자체는 float type 이지만, 값 자체는 양자화 - 역양자화 과정에서 오차가 발생
        w_quant = (w_scaled / quant_scale).round().clamp(-(2**(bit-1)), 2**(bit-1)-1) * quant_scale 

        # Shape 복원
        w_reconstr = w_quant.reshape(org_shape)
        
        # Scale 복원
        # scalining 으로 발생한 효과를 남기고, 값만 복원
        w_reconstr = w_quant / scale.view(1, -1)
        
        # MSE 계산
        with torch.no_grad():
            original_out = torch.matmul(x, w.t())
            quant_out = torch.matmul(x, w_reconstr.t())
            loss = (original_out - quant_out).pow(2).mean()
        
        return loss.item()