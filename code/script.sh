#!/bin/bash

echo "Starting data collection..."
python src/collect_data.py

echo "Starting LSTM model training..."
pip install scikit-learn
python src/train_lstm.py

echo "Starting base model training..."
python src/train_base.py

echo "Starting DQN model training..."
python src/train_dqn.py
