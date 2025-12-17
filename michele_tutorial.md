## Introduction

Chiller-based cooling systems are among the main contributors to energy consumption in buildings, as they must satisfy a highly variable demand while maintaining comfortable indoor conditions. In practice, the chilled water supply temperature (***Tchws***) is often set using **rule-based strategies** (fixed schedules by time of day or season), which do not react optimally to load variations. This can lead to over-cooling or insufficient cooling, resulting in energy waste or insufficient indoor temperature comfort.

To solve this issue, we propose an intelligent cooling system which is able to keep the Tchws always optimized by using **Deep Reinforcement Learning**. The model has an explicit trade-off objective: minimize energy consumption while keeping the indoor temperature of the building comfortable.

In this research, we replicate the methodology that He, Fu et al. describe in their paper named “Predictive control optimization of chiller plants based on deep reinforcement learning”. In particular, we follow the same two-stage methodological framework:(i) an LSTM network to estimate energy consumption and (ii) a DQN agent that selects control actions within a simulated environment.

In the conclusion we evaluate the performance of the model (with and without the LSTM predictions), and see the benefits it has.

## Predicting Future Energy Consumption with a LSTM Network

We train an LSTM regressor to predict short-term future energy consumption. This forecast is then used as an additional feature for the RL controller, so the agent can make decisions with a notion of near-future demand rather than relying only on the instantaneous state.

#### Step 1 - Define Feature and Target Variables

We start by defining the target variable and the set of input features. The target variable is `HVAC electricity demand rate`, which is the quantity we aim to forecast at each timestep. All remaining variables in the dataset are features, as they collectively describe the operating conditions of the building, including weather, indoor states, and control-related signals.
Before training, the target variable is transformed using a logarithmic mapping. This reduces the impact of extreme peaks and ensures that the model does not produce negative predictions. The features and the transformed target are then extracted as *NumPy* arrays, preparing the data for subsequent preprocessing.

```python
target_col = "HVAC_electicity_demand_rate" 
feature_cols = [col for col in df.columns if col != target_col] 

x_all = df[feature_cols].values 
y_all = np.log1p(df[target_col].values).reshape(-1, 1) # Avoid negative values
```

#### Step 2 - Time-Ordered Split into Train/Test 

Here the dataset is split into training and test subsets while preserving the original temporal order of the observations. Since the data represent a time series, random shuffling is avoided.
Specifically, the first 80% of the data is used for training, while the remaining 20% is reserved for testing.

```python
split_index = int(0.8 * len(df))
x_train_raw, x_test_raw = x_all[:split_index], x_all[split_index:]
y_train_raw, y_test_raw = y_all[:split_index], y_all[split_index:]
```

#### Step 3 - Sliding Windows

In this step, the time series data are converted into supervised learning samples using a *sliding-window* approach. Rather than predicting energy consumption from a single timestep, the LSTM is provided with a short history of past observations, allowing it to capture temporal patterns and short-term dependencies in the data.

For each prediction time t, the model receives the previous `lookback` timesteps as input and is trained to predict the energy consumption at time t. This process is repeated across the dataset, generating a set of overlapping input sequences and corresponding targets. The resulting data structure has a three dimensional shape samples, timesteps, and features.

```python
def make_sequences(X, y, lookback):
    x_seq, y_seq = [], []
    for t in range(lookback, len(X)):
        seq = X[t-lookback:t]
        x_seq.append(seq)
        y_seq.append(y[t])
    return np.array(x_seq), np.array(y_seq)

x_train_seq, y_train_seq = make_sequences(x_train, y_train, lookback)
x_test_seq, y_test_seq = make_sequences(x_test, y_test, lookback)

n_timesteps = x_train_seq.shape[1]
n_features = x_train_seq.shape[2]
```

#### Step 4 - Model Structure 

Now, we'll define the structure of the network.
The model is built as a stacked recurrent network, where a couple LSTM layers are used to progressively extract temporal features from the input sequences. The first LSTM layer processes the full input sequence and returns a sequence of hidden states, enabling the second LSTM layer to learn higher-level temporal representations.

After the recurrent layers, a dense layer with *ReLU activation* is used to improve learning and it brings us to a final linear neuron that returns the energy consumption forecast. The model is trained using the **Mean Absolute Error (MAE)** as the loss function, which is particularly suitable for energy time series as it helps with large peaks predictions and **lags** issues.

```python
model = Sequential([
    LSTM(64, input_shape=(n_timesteps, n_features), return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(1) # Prediction
])
model.compile(optimizer='adam', loss='mae') # MAE is better with peaks

# Check
model.summary()
```

#### Step 5 - Training

This is central step. The model is trained on the previously constructed input sequences. An early stopping mechanism is introduced to avoid unnecessary training once the model stops improving on the validation data.

The model is trained for 20 epochs without shuffling as it's time series data. Once training is complete, the final model is saved and later reused to generate energy consumption forecasts for the RL agent.

```python
# Set EarlyStopping to avoid useless training
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train_seq, y_train_seq,
    epochs=20,
    batch_size=32,
    validation_data=(x_test_seq, y_test_seq),
    callbacks=[es],
    verbose=1,
    shuffle=False # As it's time series data
)

# Save the trained model
model.save("chiller_lstm_model.keras")
```

#### Step 6 - Back to original units

At this stage, we use the trained model to generate predictions on the test sequences. However, the model outputs values in the same transformed space used during training (standardized and log-scaled), so the raw predictions are not yet directly interpretable in terms of real energy consumption.

To make the results meaningful, we reverse the preprocessing steps.

```python
# Predictions on the test
y_pred_test_scaled = model.predict(x_test_seq, verbose=0)

# Remove standardization 
y_true_log = scaler_y.inverse_transform(y_test_seq)
y_pred_log = scaler_y.inverse_transform(y_pred_test_scaled)
# Removing logarithm -1 (go back to original units)
y_true = np.expm1(y_true_log).ravel()   
y_pred = np.expm1(y_pred_log).ravel()
```

#### Step 7 - Evaluation 

Finally, the LSTM prediction performance is evaluated on test set using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and the coefficient of determination (R²).

The network is then compared with a baseline model that predicts the next energy value as the last observed one. 
The comparison highlights the competitiveness of the LSTM in capturing short-term variations in HVAC energy consumption.

```python
# Baseline -> prediction is the last observed energy consumption 
y_actual = y_true.to_numpy()
y_baseline = np.roll(y_actual, 1)
y_baseline[0] = y_actual[0]

# LSTM metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(f"\nLSTM -> MAE {mae:.3f}, RMSE {rmse:.3f}, R² {r2:.3f}")

# Baseline metrics
mae_base = mean_absolute_error(y_actual, y_baseline)
rmse_base = np.sqrt(mean_squared_error(y_actual, y_baseline))
r2_base = r2_score(y_actual, y_baseline)
print(f"BASELINE -> MAE {mae_base:.3f}, RMSE {rmse_base:.3f}, R² {r2_base:.3f}\n")
```