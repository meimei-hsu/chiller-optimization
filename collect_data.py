"""
collect_data.py

This script loads a trained DQN model and collects unnormalized observation data
by running the agent for multiple episodes. The collected data is saved to a CSV
file for downstream use (e.g., LSTM training).

Usage:
    python collect_data.py
"""

import os
import gymnasium as gym
import pandas as pd
from stable_baselines3 import DQN
import sinergym
from sinergym.utils.wrappers import *
import shutil
import glob

# --- Configuration ---
environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'
model_path = 'artifacts/best_model.zip'
output_csv = 'artifacts/chiller_data.csv'
output_dir = 'outputs/Data_Collection'  # Directory for Sinergym episode outputs
runperiod = (1,1,2001,31,12,2010)  # Run period for data collection

# --- Validate Model Path ---
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found at {model_path}. "
        f"Please ensure the trained model exists in the ./artifacts directory."
    )

print(f"Loading trained model from: {model_path}")

# --- Load the Trained Model ---
model = DQN.load(model_path)
print("Model loaded successfully!")

# --- Validate Output Directory ---
try:
    for path in [p for p in glob.glob(f"{output_dir}*") if os.path.isdir(p)]:
        shutil.rmtree(path)
    print(f"Removed existing output directory: {output_dir}")
except Exception as e:
    print(f"No existing output directory to remove: {e}")

# --- Create the Environment ---
# Apply the same wrappers used during training
env = gym.make(environment, env_name=output_dir, building_config={'runperiod': runperiod})
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)

print(f"Environment created: {environment}")

# --- Data Collection ---

# Initialize
obs, info = env.reset()
episode_step = 0
terminated = False
truncated = False

# Simulate the episode
while not (terminated or truncated):
    # Get agent's action using the trained model
    action, _ = model.predict(obs, deterministic=True)
    
    # Take a step in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    episode_step += 1

# Collect observations
env.reset()  # Reset to finalize logging
observation_csv = os.path.join(env.get_wrapper_attr('workspace_path'), 'episode-1/monitor/observations.csv')

episode_data = pd.read_csv(observation_csv)

# --- Close Environment ---
env.close()
print("\nData collection complete!")

# --- Save Data to CSV ---
episode_data.to_csv(output_csv, index=False)

print(f"\nSuccess! Data saved to '{output_csv}'")
print(f"DataFrame shape: {episode_data.shape}")
print(f"\nFirst few rows:")
print(episode_data.head())
print(f"\nColumn names:")
print(episode_data.columns.tolist())
