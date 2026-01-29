import gymnasium as gym
import numpy as np

from searcher import AWQSearcher
from feature import ActivationStats

class LyapunovAWQEnv(gym.Env):
    """
    RL Agent 가 각 레이어의 상태를 보고 최적의 alpha 값을 선택하게 하는 Env 클래스
    """
    def __init__(self, model_layers, calib_data, target_mse=0.01, V=10.0):
        self.layers = model_layers
        self.calib_data = calib_data
        self.searcher = AWQSearcher()

        # Lyapunov 파라미터
        self.Z = 0.0 # Virtual Queue backlog
        self.target_mse = target_mse # 허용 가능한, layer 별 평균 mse 오차
        self.V = V # Penalty Weight

        self.current_layer_index = 0

        # Action Space - alpha 값 결정 (0~1)
        # continuous / discrete 값 모두 처리할 수 있음
        # 여기에서는, 일단 간단 구현 목적으로 Discrete 하게 보기로
        self.action_space = gym.spaces.Discrete(21) # 0.05 step
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Feature Extractor
        self.hook_manager = ActivationStats()        
    
    def step(self, action_idx):
        alpha = action_idx * 0.05

        layer = self.layers[self.current_layer_index]
        x = self.calib_data[0]

        # hook 등록 (현 레이어 feature 확인)
        handle = self.hook_manager.register(layer)

        # Forward Pass
        mse = self.searcher.quantization_error(layer, x, alpha)

        # layer 통계량
        # layer_stats = self.hook_manager.current_input_stats.numpy()

        # Hook 해제
        handle.remove()
        self.hook_manager.hooks = []

        # Lyapunov Queue 업데이트
        prev_Z = self.Z
        self.Z = max(0.0, self.Z + mse - self.target_mse)

        # Reward 계산
        # Min (MSE + V * Z) | Max -(MSE + V * Z)
        reward = -1.0 * (mse + self.V * self.Z)

        # next state
        self.current_layer_index += 1
        # 모든 layer 다 훑으면 종료
        done = self.current_layer_index >= len(self.layers)
        next_state = self._get_state(self.current_layer_index, self.Z)
        
        return next_state, reward, done, False, {"mse": mse, "queue": self.Z}
    

    def _get_state(self, layer_idx, Z):
        return np.array([layer_idx, 0.5, 1.0, Z], dtype=np.float32) # 0.5, 1.0 은 실제 데이터로 대체되어야함