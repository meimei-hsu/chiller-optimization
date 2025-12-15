import os
import shutil
import glob
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList

from sinergym.utils.wrappers import *
from sinergym.utils.callbacks import *

noise_levels = [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]

# --- Class Definition ---

class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super(LSTMModel, self).__init__()
        # LSTM(64, return_sequences=True) equivalent
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        # LSTM(32) equivalent
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        # Dense(16, activation='relu')
        self.fc1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        # Dense(1)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)  
        out, _ = self.lstm2(out) 
        out = out[:, -1, :] # Take last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class PredictionPipeline:
    """
    A class to handle data transformation, prediction, and inverse transformation.
    Can be saved and loaded for future inference.
    """
    def __init__(self, model, scaler_x, scaler_y, feature_cols, target_col, lookback, device):
        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.device = device
        
    def transform_input(self, df):
        """
        Transforms raw DataFrame into sequences for the model.
        Returns: (inputs_array, targets_array_or_None)
        """
        # 1. Extract raw features
        x_raw = df[self.feature_cols].values
        
        # 2. Extract and Log-transform target (if present, else None)
        if self.target_col in df.columns:
            y_raw = np.log1p(df[self.target_col].values).reshape(-1, 1)
        else:
            raise ValueError(f"Target column '{self.target_col}' needed for lag generation.")

        # 3. Scale
        x_scaled = self.scaler_x.transform(x_raw)
        y_scaled = self.scaler_y.transform(y_raw)
        
        # 4. Create sequences
        # Input features for model = [features(t-lag) + target(t-lag)]
        x_seq = []
        ground_truth_scaled = []
        
        for t in range(self.lookback, len(x_scaled)):
            x_past = x_scaled[t - self.lookback:t]
            y_past = y_scaled[t - self.lookback:t]
            # Concatenate features and past targets
            seq = np.concatenate((x_past, y_past), axis=1)
            x_seq.append(seq)
            ground_truth_scaled.append(y_scaled[t])
            
        return np.array(x_seq), np.array(ground_truth_scaled)

    def inverse_transform_prediction(self, y_pred_scaled):
        """
        Inverse transforms model predictions to original scale.
        """
        # Inverse Scale
        y_pred_log = self.scaler_y.inverse_transform(y_pred_scaled)
        # Inverse Log1p (expm1)
        y_pred = np.expm1(y_pred_log)
        return y_pred.ravel()

    def predict(self, df):
        """
        End-to-end prediction: Raw Data -> Preprocess -> Predict -> Postprocess
        """
        self.model.eval()
        
        # Preprocess
        inputs, _ = self.transform_input(df)
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds_scaled = self.model(inputs_tensor).cpu().numpy()
            
        # Postprocess
        return self.inverse_transform_prediction(preds_scaled)

    def save(self, path):
        """Saves the pipeline state (model weights + scalers + config)"""
        # Note: We save model parameters, not the whole class, to avoid pickle issues
        state = {
            'model_state_dict': self.model.state_dict(),
            'model_input_size': self.model.lstm1.input_size, 
            'scaler_x': self.scaler_x,
            'scaler_y': self.scaler_y,
            'feature_cols': self.feature_cols,
            'target_col': self.target_col,
            'lookback': self.lookback
        }
        torch.save(state, path)
        print(f"Pipeline saved to {path}")
        
    @classmethod
    def load(cls, path, device):
        """Loads the pipeline from a file"""
        # Set weights_only=False to allow loading StandardScaler objects
        state = torch.load(path, map_location=device, weights_only=False)
        
        # Reconstruct Model
        model = LSTMModel(input_size=state['model_input_size'])
        model.load_state_dict(state['model_state_dict'])
        model.to(device)
        
        return cls(
            model=model,
            scaler_x=state['scaler_x'],
            scaler_y=state['scaler_y'],
            feature_cols=state['feature_cols'],
            target_col=state['target_col'],
            lookback=state['lookback'],
            device=device
        )

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

# --- Noise Seep ---

for noise_std in noise_levels:
    
    # --- Configuration ---
    lstm_path = 'artifacts/lstm_pipeline.pth' 
    dqn_path = 'artifacts/dqn_model.zip'

    environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'
    env_kwargs = {
        # Fixed runperiod to avoid EnergyPlus crashes with TMY3 files
        'building_config': {'runperiod': (1, 1, 1991, 31, 12, 1991), 'timesteps_per_hour': 1},
        'time_variables': ['month', 'day_of_month', 'hour'],
        'weather_variability': {'Dry Bulb Temperature': (noise_std, 0.0, 24.0), 'Relative Humidity': ((2.0, 5.0), 0.0, 24.0, (0, 100))},
        'seed': 42
    }

    # --- Setup Output Directories ---

    # Directory for Sinergym episode outputs
    log_dir = f'outputs/DQN_Noise-{noise_std}'

    # Validate output directory
    try:
        for path in [p for p in glob.glob(f"{log_dir}*") if os.path.isdir(p)]:
            shutil.rmtree(path)
        print(f"Removed existing output directory: {log_dir}")
    except Exception as e:
        print(f"No existing output directory to remove: {e}")

    # Create environments
    env = gym.make(environment, env_name=log_dir, **env_kwargs)
    env = LSTMObsWrapper(env, pipeline_path=lstm_path)
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)
    env = CSVLogger(env)

    # Run the Data Collection Loop
    obs, info = env.reset()
    terminated = False
    truncated = False

    print(f"Starting data collection for period: {env.get_wrapper_attr('runperiod')}")
    print(f"This will run for approximately {env.get_wrapper_attr('timestep_per_episode')} timesteps.")

    # Define Model
    model = DQN.load(dqn_path, env)

    while not (terminated or truncated):
        # Sample random action from the environment
        action = model.predict(obs, deterministic=True)[0]

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

    # Close
    env.close()