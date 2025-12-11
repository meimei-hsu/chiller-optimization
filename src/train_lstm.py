import os
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_path = 'artifacts/chiller_data.csv'
pipeline_path = 'artifacts/lstm_pipeline.pth' # Saving the whole pipeline here
log_dir = 'outputs/LSTM_Training'

target_column = 'HVAC_electricity_demand_rate'
lookback_window = 24
seed = 42
batch_size = 32
epochs = 100

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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if __name__ == "__main__":
    # --- Ensure Reproducibility ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- Validate Paths ---
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")

    # ==========================================
    # 3. Data Preparation & Training Setup
    # ==========================================

    # --- Load Data ---
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != target_column]

    # --- Split Data ---
    split_index = int(0.8 * len(df))
    df_train = df.iloc[:split_index].copy()
    df_test = df.iloc[split_index:].copy()

    # --- Fit Scalers (Only on Train) ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    x_train_raw = df_train[feature_cols].values
    y_train_raw_log = np.log1p(df_train[target_column].values).reshape(-1, 1)

    scaler_x.fit(x_train_raw)
    scaler_y.fit(y_train_raw_log)

    # Initialize untrained model to calculate input size
    # Input size = number of features + 1 (for the target lag)
    n_features = len(feature_cols) + 1 
    model = LSTMModel(input_size=n_features).to(device)

    # --- Create Pipeline Instance (used for generating training sequences too) ---
    pipeline = PredictionPipeline(
        model=model,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        feature_cols=feature_cols,
        target_col=target_column,
        lookback=lookback_window,
        device=device
    )

    # --- Generate Sequences using Pipeline Logic ---
    # Note: We use the internal transform_input method to get scaled tensors
    x_train_seq, y_train_seq = pipeline.transform_input(df_train)
    x_test_seq, y_test_seq = pipeline.transform_input(df_test)

    print(f"Sequences created. Train: {x_train_seq.shape}, Test: {x_test_seq.shape}")

    # Create DataLoaders
    train_data = TensorDataset(torch.FloatTensor(x_train_seq), torch.FloatTensor(y_train_seq))
    test_data = TensorDataset(torch.FloatTensor(x_test_seq), torch.FloatTensor(y_test_seq))

    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    # ==========================================
    # 4. Training Loop
    # ==========================================
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters())

    early_stopper = EarlyStopping(patience=10, min_delta=0.001)
    train_losses = []
    val_losses = []

    print("Starting Training...")
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_pred = model(x_val)
                v_loss = criterion(val_pred, y_val)
                val_batch_losses.append(v_loss.item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    # Save the Pipeline (containing the trained model)
    pipeline.save(pipeline_path)

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'))
    plt.close()

    # ==========================================
    # 5. Evaluation using Loaded Pipeline
    # ==========================================
    print("\n--- Evaluating using Loaded Pipeline ---")

    # 1. Load pipeline from file
    loaded_pipeline = PredictionPipeline.load(pipeline_path, device)

    # 2. Predict on Test Data (End-to-End)
    # This handles raw data -> scaling -> sequence creation -> inference -> inverse scaling
    y_pred_final = loaded_pipeline.predict(df_test)

    # 3. Get Ground Truth for comparison
    # Note: The pipeline predicts starting from index 'lookback', so we align truth
    _, y_true_seq_scaled = loaded_pipeline.transform_input(df_test)
    y_true_final = loaded_pipeline.inverse_transform_prediction(y_true_seq_scaled)

    # 4. Metrics
    mae = mean_absolute_error(y_true_final, y_pred_final)
    rmse = np.sqrt(mean_squared_error(y_true_final, y_pred_final))
    r2 = r2_score(y_true_final, y_pred_final)

    print(f"Evaluation Results:")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")

    # 5. Plot
    plt.figure(figsize=(15, 6))
    subset_n = 500
    plt.plot(y_true_final[:subset_n], label='Actual')
    plt.plot(y_pred_final[:subset_n], label='Forecast', color='red', alpha=0.7)
    plt.title(f'LSTM Forecasting (First {subset_n} test points)')
    plt.ylabel('HVAC_electricity_demand_rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'lstm_forecast.png'))
    plt.close()