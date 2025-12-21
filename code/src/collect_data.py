"""
collect_data.py

This script loads a trained DQN model and collects unnormalized observation data
by running the agent for multiple episodes. The collected data is saved to a CSV
file for downstream use (e.g., LSTM training).

Usage:
    python collect_data.py
"""

import os
import shutil
import glob

import gymnasium as gym
import pandas as pd

import sinergym
from sinergym.utils.wrappers import LoggerWrapper, CSVLogger

# --- Configuration ---
output_csv = 'artifacts/chiller_data.csv'
log_dir = 'outputs/Data_Collection'  # Directory for Sinergym episode outputs

environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'
env_kwargs = {
    'building_config': {'runperiod': (1,1,1991,12,3,2000), 'timesteps_per_hour': 1},
    'time_variables': ['month', 'day_of_month', 'hour'],
    'seed': 42
}

# Validate output directory
try:
    for path in [p for p in glob.glob(f"{log_dir}*") if os.path.isdir(p)]:
        shutil.rmtree(path)
    print(f"Removed existing output directory: {log_dir}")
except Exception as e:
    print(f"No existing output directory to remove: {e}")

# Create environments
env = gym.make(environment, env_name=log_dir, **env_kwargs)
env = LoggerWrapper(env)
env = CSVLogger(env)

# Run the Data Collection Loop
obs, info = env.reset()
terminated = False
truncated = False

print(f"Starting data collection for period: {env.get_wrapper_attr('runperiod')}")
print(f"This will run for approximately {env.get_wrapper_attr('timestep_per_episode')} timesteps.")

while not (terminated or truncated):
    # Sample random action from the environment
    action = env.action_space.sample()

    # Step the environment
    obs, reward, terminated, truncated, info = env.step(action)

# Collect observations
env.reset()  # Reset to finalize logging
observation_csv = os.path.join(env.get_wrapper_attr('workspace_path'), 'episode-1/monitor/observations.csv')

episode_data = pd.read_csv(observation_csv)

# Close Environment
env.close()
print("\nData collection complete!")

# --- Save Data to CSV ---
# episode_data.sort_values(by=['year','month','day_of_month','hour'], inplace=True)
episode_data.to_csv(output_csv, index=False)

print(f"\nSuccess! Data saved to '{output_csv}'")
print(f"DataFrame shape: {episode_data.shape}")
print(f"\nFirst few rows:")
print(episode_data.head())
print(f"\nColumn names:")
print(episode_data.columns.tolist())