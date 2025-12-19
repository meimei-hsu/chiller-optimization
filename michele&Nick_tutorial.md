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

## Sensitivity Analysis

Sensitivity analysis aims to identify which input variables have the greatest influence on the target output. By systematically examining feature–output relationships, we can better understand the model behavior, detect dominant drivers, and verify whether the learned patterns are physically and operationally reasonable.

In this tutorial, we apply several complementary sensitivity analysis techniques, starting from simple linear correlation and progressively moving toward model-based importance measures.

---

### Step 1 - Import Libraries and Basic Setup

We first import the core libraries required for data processing, visualization, and reproducibility. These libraries will be reused across all subsequent sensitivity analysis steps.

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# For reproducibility
RANDOM_STATE = 42
```

### Step 2 - Load Dataset and Sampling

To speed up iteration and reduce computational cost, we load the dataset and optionally subsample a fixed number of rows, while keeping reproducibility via a fixed random seed.

```python
df = pd.read_csv("chiller_data.csv")

# Optional subsampling for faster experimentation
df = df.sample(n=20000, random_state=RANDOM_STATE).reset_index(drop=True)

print("Dataset shape:", df.shape)
df.head()
```

### Step 3 - Define Target Variable and Features

We define the prediction target and construct the feature set by excluding the target itself and other non-informative or leakage-prone variables, ensuring a clean separation between inputs and outputs for sensitivity analysis.
```python
target_col = "HVAC_electricity_demand_rate"

drop_cols = [
    target_col,
    "total_electricity_HVAC",
    "month",
    "day_of_month",
    "hour",
]

# Keep only columns that exist in the dataset
drop_cols = [c for c in drop_cols if c in df.columns]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[target_col]

print("Target variable:", target_col)
print("Number of features:", len(feature_cols))
print("X shape:", X.shape, "| y shape:", y.shape)
```

### Step 4 - Pearson Correlation Analysis

As a first-order sensitivity check, we compute the Pearson correlation between each feature and the target variable. Features are ranked by the absolute value of correlation to identify variables with the strongest linear relationships to the target.
```python
# Compute Pearson correlation between features and target
corr_series = df[feature_cols + [target_col]].corr()[target_col].drop(target_col)

# Rank by absolute correlation
corr_df = (
    corr_series.abs()
    .sort_values(ascending=False)
    .reset_index()
)
corr_df.columns = ["feature", "abs_pearson_corr"]

# Display top features
print(corr_df.head(10))

# Optional: visualize correlation as a bar chart
plt.figure(figsize=(8, 4))
sns.barplot(
    data=corr_df.head(15),
    x="abs_pearson_corr",
    y="feature",
    orient="h"
)
plt.title("Top Features by Absolute Pearson Correlation")
plt.tight_layout()
plt.show()
```

### Step 5 - Random Forest and Permutation Importance

To capture nonlinear effects and feature interactions beyond linear correlation, we train a Random Forest regressor and evaluate feature importance using permutation importance. This provides a more reliable sensitivity measure under complex relationships.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# Train Random Forest
rf = RandomForestRegressor(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("Train R^2:", rf.score(X_train, y_train))
print("Test  R^2:", rf.score(X_test, y_test))

# Permutation importance on test set
perm = permutation_importance(
    rf,
    X_test,
    y_test,
    n_repeats=8,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

perm_df = (
    pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    })
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

# Display top features
print(perm_df.head(10))

# Optional: visualize permutation importance
plt.figure(figsize=(8, 4))
sns.barplot(
    data=perm_df.head(15),
    x="importance_mean",
    y="feature",
    orient="h"
)
plt.title("Top Features by Permutation Importance (Random Forest)")
plt.tight_layout()
plt.show()
```

### Step 6 - Comparison and Summary of Sensitivity Results

We summarize the sensitivity analysis by comparing linear (Pearson correlation) and model-based (permutation importance) rankings. This comparison helps assess the consistency between simple linear relationships and nonlinear feature contributions captured by the Random Forest.

```python
# Prepare Pearson correlation dataframe
pearson_df = corr_df.copy()
pearson_df = pearson_df.rename(columns={"abs_pearson_corr": "pearson_corr"})

# Prepare permutation importance dataframe
perm_summary_df = perm_df.copy()
perm_summary_df = perm_summary_df.rename(columns={"importance_mean": "perm_importance"})

# Merge results for comparison
summary_df = (
    pearson_df.merge(perm_summary_df, on="feature", how="outer")
    .fillna(0)
)

# Sort by permutation importance (primary criterion)
summary_df = summary_df.sort_values(
    by="perm_importance", ascending=False
).reset_index(drop=True)

# Display top features
print(summary_df.head(15))

# Optional: visualize comparison
plt.figure(figsize=(8, 5))
summary_plot_df = summary_df.head(10)

plt.scatter(
    summary_plot_df["pearson_corr"],
    summary_plot_df["perm_importance"]
)

for i, feat in enumerate(summary_plot_df["feature"]):
    plt.text(
        summary_plot_df["pearson_corr"].iloc[i],
        summary_plot_df["perm_importance"].iloc[i],
        feat,
        fontsize=8
    )

plt.xlabel("Absolute Pearson Correlation")
plt.ylabel("Permutation Importance")
plt.title("Sensitivity Comparison: Pearson vs. Permutation Importance")
plt.tight_layout()
plt.show()
```

## Noise Robustness Analysis

In practical deployments, input measurements are inevitably affected by sensor noise and disturbances. To evaluate the robustness of the prediction model under imperfect observations, we conduct a noise robustness analysis by injecting controlled Gaussian noise into the input features at test time.

The objective is to assess how prediction performance degrades as the noise level increases, while keeping the trained model unchanged.

### Step 1 - Baseline Prediction without Noise

We first evaluate the prediction performance under clean test inputs (no noise). This baseline serves as a reference point for subsequent noise robustness experiments.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ground truth
y_true = y_test.values if hasattr(y_test, "values") else y_test

# Baseline prediction (no noise)
y_pred_baseline = y_pred.copy()

# Evaluation metrics
mae_base = mean_absolute_error(y_true, y_pred_baseline)
rmse_base = mean_squared_error(y_true, y_pred_baseline, squared=False)
r2_base = r2_score(y_true, y_pred_baseline)

print(f"Baseline MAE  : {mae_base:.3f}")
print(f"Baseline RMSE : {rmse_base:.3f}")
print(f"Baseline R^2  : {r2_base:.3f}")
```

### Step 2 - Inject Gaussian Noise at Test Time

We inject Gaussian noise into the test inputs at inference time while keeping the trained model unchanged. This simulates measurement noise and allows us to isolate the effect of input uncertainty on prediction performance.
```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Noise level (as a fraction of feature standard deviation)
noise_std_ratio = 0.05
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# Compute per-feature standard deviation from training data
# (if X_train is unavailable, use X_test instead)
if 'X_train' in globals():
    feat_std = np.std(X_train, axis=0)
else:
    feat_std = np.std(X_test, axis=0)

# Generate Gaussian noise
noise = rng.normal(
    loc=0.0,
    scale=noise_std_ratio * feat_std,
    size=X_test.shape
)

# Add noise at test time
X_test_noisy = X_test + noise

# Predict with noisy inputs
y_pred_noisy = model.predict(X_test_noisy)

# Ensure shapes are 1D
y_true = y_test.values if hasattr(y_test, "values") else y_test
y_pred_noisy = np.ravel(y_pred_noisy)

# Evaluate performance under noise
mae_noisy = mean_absolute_error(y_true, y_pred_noisy)
rmse_noisy = mean_squared_error(y_true, y_pred_noisy, squared=False)
r2_noisy = r2_score(y_true, y_pred_noisy)

print(f"Noisy MAE  : {mae_noisy:.3f}")
print(f"Noisy RMSE : {rmse_noisy:.3f}")
print(f"Noisy R^2  : {r2_noisy:.3f}")
```

### Step 3 - Noise Level Sweep

We extend the noise robustness analysis by evaluating multiple noise levels. For each noise level, Gaussian noise is injected at test time and the corresponding prediction performance is recorded. This allows us to observe how model performance degrades as input noise increases.
```python
# Define noise levels to evaluate
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]

results = []

for noise_std_ratio in noise_levels:
    rng = np.random.default_rng(RANDOM_STATE)

    # Compute per-feature standard deviation
    if 'X_train' in globals():
        feat_std = np.std(X_train, axis=0)
    else:
        feat_std = np.std(X_test, axis=0)

    # Generate Gaussian noise
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_ratio * feat_std,
        size=X_test.shape
    )

    # Add noise to test inputs
    X_test_noisy = X_test + noise

    # Predict
    y_pred_noisy = model.predict(X_test_noisy)
    y_pred_noisy = np.ravel(y_pred_noisy)

    # Ground truth
    y_true = y_test.values if hasattr(y_test, "values") else y_test

    # Metrics
    mae = mean_absolute_error(y_true, y_pred_noisy)
    rmse = mean_squared_error(y_true, y_pred_noisy, squared=False)
    r2 = r2_score(y_true, y_pred_noisy)

    results.append({
        "noise_std_ratio": noise_std_ratio,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)
print(results_df)
```

### Step 4 - Performance Degradation Analysis

We visualize how prediction performance degrades as the noise level increases. A robust model is expected to exhibit gradual performance degradation rather than abrupt collapse when exposed to increasing input noise.
```python
import matplotlib.pyplot as plt

# Sort by noise level to ensure correct plotting order
results_df = results_df.sort_values("noise_std_ratio")

plt.figure(figsize=(8, 5))

plt.plot(
    results_df["noise_std_ratio"],
    results_df["MAE"],
    marker="o",
    label="MAE"
)

plt.plot(
    results_df["noise_std_ratio"],
    results_df["RMSE"],
    marker="o",
    label="RMSE"
)

plt.plot(
    results_df["noise_std_ratio"],
    results_df["R2"],
    marker="o",
    label="R²"
)

plt.xlabel("Noise standard deviation ratio")
plt.ylabel("Performance metric value")
plt.title("Noise Robustness: Performance Degradation Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```