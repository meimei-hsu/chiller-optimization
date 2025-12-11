# Synergym reference: https://ugr-sail.github.io/sinergym/compilation/main/index.html

import os
import sys
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList

import sinergym
from sinergym.utils.callbacks import *
from sinergym.utils.constants import *
from sinergym.utils.rewards import *
from sinergym.utils.wrappers import *
from sinergym.utils.logger import TerminalLogger

# --- 1. EXPERIMENT CONFIGURATION ---

# Environment ID: Use a DISCRETE action space for DQN
environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'

# Training episodes
episodes = 100

# Evaluation frequency
eval_freq = 2

# Random seed
seed = 42

# --- Setup Output Directories ---

# Directory for Sinergym episode outputs
log_dir = 'outputs/Baseline_Training'

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
dqn_train_name = os.path.join(log_dir, "Baseline_Train")
dqn_eval_name = os.path.join(log_dir, "Baseline_Eval")

# --- 2. AGENT TRAINING ---

# --- Create environments ---
# env_name is now the full path for the output folder
env = gym.make(environment, env_name=dqn_train_name, seed=seed)
eval_env = gym.make(environment, env_name=dqn_eval_name, seed=seed)

# --- Apply Wrappers ---
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env) # Logs training data

# Apply same wrappers to eval_env
eval_env = NormalizeObservation(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env) # Logs step-by-step eval data

# --- Define Model ---
# device="auto" will default to "cpu" since GPU is not available
model = DQN('MlpPolicy', env, buffer_size=100000, verbose=1, device="auto")

# --- Set up Callbacks ---
callbacks = [] 
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    eval_freq_episodes=eval_freq, # Evaluate every 5 training episodes
    n_eval_episodes=1, # Run 1 full evaluation episode
    deterministic=True)

callbacks.append(eval_callback)
callback = CallbackList(callbacks)

# --- Training ---
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

print(f"--- Starting DQN Training for {timesteps} timesteps ---")
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=100)
print("--- DQN Training Complete ---")

# Get paths before closing
dqn_train_path = env.get_wrapper_attr('workspace_path')
dqn_eval_path = eval_env.get_wrapper_attr('workspace_path')

env.close()
eval_env.close()