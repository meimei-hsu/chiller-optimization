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

# 🔥 加入兩個 LSTM 相關的模組（你新增的 py 檔）
from lstm_obs_wrapper import LSTMObsWrapper   # <= 必加

# --- 1. EXPERIMENT CONFIGURATION ---

environment = 'Eplus-5zone-mixed-discrete-stochastic-v1'
episodes = 100

# --- Setup Output Directories ---
experiment_date_str = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
base_output_dir = os.path.join('outputs', experiment_date_str)

dqn_train_name = os.path.join(base_output_dir, "DQN_Train")
dqn_eval_name = os.path.join(base_output_dir, "DQN_Eval")

os.makedirs(base_output_dir, exist_ok=True)
print(f"Created experiment directory: {base_output_dir}")

# --- 2. AGENT TRAINING ---

# --- Create environments ---
env = gym.make(environment, env_name=dqn_train_name)
eval_env = gym.make(environment, env_name=dqn_eval_name)

# --- Apply Wrappers ---
# 訓練環境
env = NormalizeObservation(env)
env = LoggerWrapper(env)
env = CSVLogger(env)
env = LSTMObsWrapper(env)     # 🔥 在這裡加入 LSTM 預測 feature

# 評估環境
eval_env = NormalizeObservation(eval_env)
eval_env = LoggerWrapper(eval_env)
eval_env = CSVLogger(eval_env)
eval_env = LSTMObsWrapper(eval_env)   # 🔥 評估環境也要加入

# --- Define Model ---
model = DQN('MlpPolicy', env, buffer_size=100000, verbose=1, device="auto")

# --- Set up Callbacks ---
callbacks = []
eval_callback = LoggerEvalCallback(
    eval_env=eval_env,
    train_env=env,
    n_eval_episodes=1,
    eval_freq_episodes=5,
    deterministic=True
)
callbacks.append(eval_callback)
callback = CallbackList(callbacks)

# --- Training ---
timesteps = episodes * (env.get_wrapper_attr('timestep_per_episode') - 1)

print(f"--- Starting DQN Training for {timesteps} timesteps ---")
model.learn(
    total_timesteps=timesteps,
    callback=callback,
    log_interval=100
)
print("--- DQN Training Complete ---")

# --- Close environments ---
dqn_train_path = env.get_wrapper_attr('workspace_path')
dqn_eval_path = eval_env.get_wrapper_attr('workspace_path')

env.close()
eval_env.close()
