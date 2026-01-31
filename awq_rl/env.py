import gymnasium as gym
import numpy as np

from searcher import AWQSearcher
from feature import ActivationStats

class LyapunovAWQEnv(gym.Env):
    """
    RL Agent 가 각 레이어의 상태를 보고 최적의 alpha 값을 선택하게 하는 Env 클래스
    """
    def __init__(self, model_layers, calib_data, target_cost=1.3, V=10.0):
        """
        - target_cost: 평균 Group Cost 목표
        - Cost 1.0: 모든 레이어가 Group Size 128 사용
        - Cost 2.0: 모든 레이어가 Group Size 64 사용
        """
        self.layers = model_layers
        self.calib_data = calib_data
        self.searcher = AWQSearcher()

        # Lyapunov 파라미터
        self.Z = 0.0 
        self.target_cost = target_cost
        self.V = V 

        self.current_layer_index = 0

        # Action Space (현재는 40개)
        # 0~19: Group Size 128 (alpha 는 0.05 간격) : Low Cost (1.0)
        # 20~39: Group Size 64 (역시 마찬가지로 alpha 는 0.05 간격): High Cost (2.0)
        self.action_space = gym.spaces.Discrete(40)
        
        # Observation Space; [Layeridx, Mean, Std, Kurt, Skew, Max, Queue(Z)]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Feature Extractor
        self.hook_manager = ActivationStats()        
    
    def step(self, action_idx):
        # Action 해석(decoding) 단계
        if action_idx < 20:
            group_size = 128
            cost = 1.0
            alpha = action_idx * 0.05
        else:
            group_size = 64
            cost = 2.0
            alpha = (action_idx - 20) * 0.05

        layer = self.layers[self.current_layer_index]
        x = self.calib_data[0]

        # hook 등록 (현 레이어 feature 확인)
        handle = self.hook_manager.register(layer)

        # Forward Pass
        # bit 도 추가적으로 전달
        mse = self.searcher.quantization_error(layer, x, alpha, bit=4, group_size=group_size)

        # layer 통계량
        # layer_stats = self.hook_manager.current_input_stats.numpy()

        # Hook 해제
        handle.remove()
        self.hook_manager.hooks = []

        # Lyapunov Queue 업데이트
        # 현재 Cost 가 Target Cost 보다 크면 Z 증가 (목표는 time-averaged Z 발산 방지)
        self.Z = max(0.0, self.Z + cost - self.target_cost)

        # Reward 계산
        # Min (MSE + V * Z) | Max -(MSE + V * Z)
        # MSE 는 작을수록 좋고, Z 도 작을수록 좋음
        reward = -1.0 * (mse + self.V * self.Z)

        # next state
        self.current_layer_index += 1
        # 모든 layer 다 훑으면 종료
        done = self.current_layer_index >= len(self.layers)
        next_state = self._get_state(self.current_layer_index, self.Z)
        
        return next_state, reward, done, False, {"mse": mse, "queue": self.Z, "group": group_size}
    

    def _get_state(self, layer_idx, Z):
        """
        실제 feature.py 에서 계산된 값을 통해 state 를 구성하는 함수
        """
        stats = self.hook_manager.current_input_stats

        # 훈련 과정을 위해 layer idx 정규화
        norm_idx = layer_idx / len(self.layers)
        state = np.concatenate(([norm_idx], stats.numpy(), [Z]))
        return state