# lstm_obs_wrapper.py
import numpy as np
import gymnasium as gym

from lstm_sequence import LSTMPredictionSequence


class LSTMObsWrapper(gym.Wrapper):
    """
    這個 wrapper 會在 observation 後面多加一個 feature：
      - LSTM 預測的 chiller energy consumption (kWh)

    所以：
    - 原本 obs: [x1, x2, ..., xn]
    - 現在 obs: [x1, x2, ..., xn, pred_kWh]
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.seq = LSTMPredictionSequence()

        orig_space = env.observation_space
        assert isinstance(orig_space, gym.spaces.Box), \
            "This wrapper assumes a Box observation space."

        # 新的 obs space = 原本維度 + 1
        low = np.concatenate([orig_space.low, np.array([-np.inf], dtype=np.float32)])
        high = np.concatenate([orig_space.high, np.array([np.inf], dtype=np.float32)])

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        # Gymnasium API: reset() -> (obs, info)
        obs, info = self.env.reset(**kwargs)
        self.seq.reset()
        obs_aug = self._augment_obs(obs)
        return obs_aug, info

    def step(self, action):
        # Gymnasium API: step() -> (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_aug = self._augment_obs(obs)
        return obs_aug, reward, terminated, truncated, info

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """把一個 scalar 的 LSTM 預測值貼在 observation 後面。"""
        pred_kwh = self.seq.next()
        return np.concatenate(
            [obs, np.array([pred_kwh], dtype=np.float32)],
            axis=0,
        )
