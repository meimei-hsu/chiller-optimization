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

# --- Setup Output Directories ---
# Create a single, time-stamped folder for this experiment
# Use a format safe for filenames (no colons)
experiment_date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
base_output_dir = os.path.join('outputs', experiment_date_str)

# Define specific sub-folders for each run
dqn_train_name = os.path.join(base_output_dir, "DQN_Train")
dqn_eval_name = os.path.join(base_output_dir, "DQN_Eval")

# Create the base directory
os.makedirs(base_output_dir, exist_ok=True)
print(f"Created experiment directory: {base_output_dir}")


# --- 2. AGENT TRAINING ---

# --- Create environments ---
# env_name is now the full path for the output folder
env = gym.make(environment, env_name=dqn_train_name)
eval_env = gym.make(environment, env_name=dqn_eval_name)

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
    n_eval_episodes=1, # Run 1 full evaluation episode
    eval_freq_episodes=5, # Evaluate every 5 training episodes
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