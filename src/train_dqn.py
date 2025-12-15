import os
import shutil
import glob
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList

import sinergym
from sinergym.utils.wrappers import *
from sinergym.utils.callbacks import *

from train_lstm import PredictionPipeline, LSTMModel

# --- Configuration ---
pipeline_path = 'artifacts/lstm_pipeline.pth' 

environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'
env_kwargs = {
    # Fixed runperiod to avoid EnergyPlus crashes with TMY3 files
    'building_config': {'runperiod': (1, 1, 1991, 31, 12, 1991), 'timesteps_per_hour': 1},
    'time_variables': ['month', 'day_of_month', 'hour'],
    'weather_variability': {'Dry Bulb Temperature': (1.0, 0.0, 24.0), 'Relative Humidity': ((2.0, 5.0), 0.0, 24.0, (0, 100))},
    'seed': 42
}
episodes = 100
eval_freq = 2

# --- Setup Output Directories ---

# Directory for Sinergym episode outputs
log_dir = 'outputs/DQN_Training-1.0'

# Validate output directory
try:
    for path in [p for p in glob.glob(f"{log_dir}*") if os.path.isdir(p)]:
        shutil.rmtree(path)
    print(f"Removed existing output directory: {log_dir}")
except Exception as e:
    print(f"No existing output directory to remove: {e}")

# Create the base directory
os.makedirs(log_dir, exist_ok=True)
print(f"Created experiment directory: {log_dir}")

# Define specific sub-folders for each run
dqn_train_name = os.path.join(log_dir, "DQN_Train")
dqn_eval_name = os.path.join(log_dir, "DQN_Eval")

# --- 1. LSTMObsWrapper ---

class LSTMObsWrapper(gym.Wrapper):
    """
    Adds LSTM predicted chiller energy consumption to the observation.
    Maintains a history buffer to create sequences for the LSTM.
    """
    def __init__(self, env: gym.Env, pipeline_path: str):
        super().__init__(env)
        
        # Load the pipeline
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.pipeline = PredictionPipeline.load(pipeline_path, device)
        self.lookback = self.pipeline.lookback
        
        # Buffer to store history for LSTM (sliding window)
        self.history = deque(maxlen=self.lookback)
        
        # Extend observation space by 1 (for the prediction)
        orig_space = env.observation_space
        low = np.concatenate([orig_space.low, np.array([-np.inf], dtype=np.float32)])
        high = np.concatenate([orig_space.high, np.array([np.inf], dtype=np.float32)])
        
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )
        
        # Cache variable names for DataFrame construction
        self.obs_vars = env.get_wrapper_attr('observation_variables')

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Clear and Warm-up buffer
        self.history.clear()
        # Since we don't have past data on reset, we duplicate the initial frame
        for _ in range(self.lookback):
            self.history.append(obs)
            
        obs_aug = self._augment_obs(obs)
        return obs_aug, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update buffer
        self.history.append(obs)
        
        obs_aug = self._augment_obs(obs)
        return obs_aug, reward, terminated, truncated, info

    def _augment_obs(self, obs):
        """
        1. Convert history buffer to DataFrame
        2. Run Pipeline Prediction
        3. Append prediction to current observation
        """
        df_history = pd.DataFrame(list(self.history), columns=self.obs_vars)
        
        # The pipeline's transform_input iterates range(lookback, len(df)).
        # If len(df) == lookback (24), the range is empty.
        # We append a dummy row to make len(df) == 25, forcing generation of 1 sequence.
        if len(df_history) == self.lookback:
            # Duplicate the last row. The values don't matter because this row 
            # is only used as the "target" (which we ignore) and not part of the input window.
            dummy_row = df_history.iloc[-1:].copy()
            df_history = pd.concat([df_history, dummy_row], ignore_index=True)

        # Handle edge case where history is not full yet
        if len(df_history) < self.lookback:
            # Pad with last row or zeros to match required length
            missing = self.lookback - len(df_history)
            last_row = df_history.iloc[-1:]
            padding = pd.concat([last_row] * missing, ignore_index=True)
            df_history = pd.concat([df_history, padding], ignore_index=True)

        try:
            # This returns an array of predictions. With 25 rows input, we get 1 prediction.
            predictions = self.pipeline.predict(df_history)
            pred_val = predictions[-1]
        except Exception as e:
            # Fallback for debugging, though the dummy row fix prevents the 1D error
            print(f"LSTM Prediction Error: {e}. Defaulting to 0.")
            pred_val = 0.0

        # Clip to observation space bounds to suppress warnings and ensure stability
        pred_val = np.clip(
            pred_val, 
            self.observation_space.low[-1], 
            self.observation_space.high[-1]
        )

        # Concatenate original obs with prediction
        return np.concatenate([obs, [pred_val]], axis=-1).astype(np.float32)

# --- 2. Create Environments ---

def make_env(env_name):
    # 1. Create Base Env
    env = gym.make(environment, env_name=env_name, **env_kwargs)
    
    # 2. Apply LSTMObsWrapper FIRST (Inner-most)
    # It receives RAW observations from EnergyPlus and produces RAW augmented observations.
    # This is crucial because the LSTM pipeline expects raw (unscaled) values.
    env = LSTMObsWrapper(env, pipeline_path='artifacts/lstm_pipeline.pth')
    
    # 3. Apply NormalizeObservation LAST (Outer-most)
    # It normalizes the combined state (Physical Variables + LSTM Prediction).
    # This ensures the DQN agent sees a fully normalized state space.
    env = NormalizeObservation(env)
    
    # 4. LoggerWrapper for console output
    env = LoggerWrapper(env)

    # 5. Apply CSVLogger (it grabs data from the LoggerWrapper)
    env = CSVLogger(env)
    
    return env

env = make_env(dqn_train_name)
eval_env = make_env(dqn_eval_name)

# --- 3. AGENT TRAINING ---

# Define Model
model = DQN('MlpPolicy', env, verbose=1, device="auto")

# Callbacks
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    eval_freq_episodes=eval_freq,
    n_eval_episodes=1,
    deterministic=True
)
callback = CallbackList([eval_callback])

# Training
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

print(f"--- Starting DQN Training for {timesteps} timesteps ---")
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=100
)
print("--- DQN Training Complete ---")

# Close
env.close()
eval_env.close()