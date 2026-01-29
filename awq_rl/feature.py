import torch
import torch.nn as nn

class ActivationStats:
    """
    AWQ 의 핵심 term 인, activation 분포를 관찰하는 클래스
    """
    def __init__(self):
        self.stats = {} # {layer_name: stats_vector}
        self.hooks = []
    
    def _compute_stats(self, x):
        """
        입력 activation tensor x 의 통계적 특징을 계산하는 함수
        State Vector: [mean, std, kurtosis, skewness, max]

        - std: Activation 값들이 얼마나 퍼져있는지
        - kurtosis: Activation 분포가 얼마나 뾰족한지 (Outlier 가 많을수록 값이 커짐)
        - skewness: 분포가 한쪽으로 쏠려있는지
        - max: 절댓값 최대치는 얼마인지
        """
        # x shape: [Batch, Seq, Dim]
        # Layer 전체의 난이도를 확인하기 위해 shape 을 [N, Dim] 으로 flatten
        x_flat = x.detach().float().view(-1)

        mean = x_flat.mean()
        std = x_flat.std()
        max_val = x_flat.abs().max()

        # Kurtosis 계산: Exp[( (x-mu) / sigma )^4]
        # 값이 클수록 outlier 많음 - 양자화하기 빡셈 - alpha 조절해야 함
        centered = x_flat - mean
        fourth_moment = torch.mean(centered ** 4)
        kurtosis = fourth_moment / (std ** 4 + 1e-6) # 1e-6 은 0 으로 나누기 방지

        # skewness 계산: Exp[( (x-mu) / sigma )^3]
        third_moment = torch.mean(centered ** 3)
        skewness = third_moment / (std ** 3)

        return torch.tensor([mean.item(), std.item(), kurtosis.item(), skewness.item(), max_val.item()])

    def hook_fn(self, module, input, output):
        """
        레이어 별 stats 계산하는 함수
        """
        x = input[0]
        self.current_input_stats = self._compute_stats(x)
    
    def register(self, layer):
        """
        특정 레이어에 stats 계산 함수 (hook) 등록하는 함수
        """
        handle = layer.register_forward_hook(self.hook_fn)
        self.hooks.append(handle)
        return handle

    def remove_hooks(self):
        """
        메모리 누수 방지 목적으로, hook 제거하는 함수
        """
        for h in self.hooks:
            h.remove()
        self.hooks = []