# Chiller Optimization — Final Project

This repository contains the implementation of a reinforcement learning (DQN) agent for optimizing building HVAC chiller systems using Sinergym. The project includes training scripts, data collection utilities, and visualization tools for building energy control optimization.

## 🎯 Project Overview

The project uses Deep Q-Network (DQN) to learn optimal control policies for a 5-zone building's HVAC system, balancing energy consumption with thermal comfort. The environment is based on EnergyPlus simulation with stochastic weather conditions.

## 📁 Repository Structure

```
.
├── env_demo.py              # DQN training script with evaluation callbacks
├── collect_data.py          # Data collection script for generating training datasets
├── plot_results.ipynb       # Jupyter notebook for visualizing results
├── artifacts/               # Trained models and collected datasets
│   ├── base_model.zip       # Trained DQN model
│   └── chiller_data.csv     # Collected observation data (10 years)
├── outputs/                 # Experiment outputs (gitignored)
│   └── YYYY-MM-DD_HH-MM-SS/ # Timestamped experiment directories
│       ├── DQN_Train-res1/  # Training episodes and logs
│       └── DQN_Eval-res1/   # Evaluation episodes and metrics
└── README.md
```

## 🚀 Quick Start (Docker)

This project uses the official Sinergym Docker image to ensure a consistent environment with EnergyPlus and all dependencies.

### 1. Pull the Sinergym Docker Image

```bash
docker pull sailugr/sinergym:v3.10.0
```

### 2. Run the Docker Container

Navigate to the repository root and start an interactive container:

```bash
cd /path/to/this/repo
docker run -it --rm -v "$(pwd)":/app sailugr/sinergym:v3.10.0 /bin/bash
```

This mounts your local repository to `/app` inside the container, so outputs are written to your local filesystem.

### 3. Inside the Container

```bash
cd /app
python your_script.py
exit
```

### 4. View Results

After exiting the container, open the Jupyter notebook on your host machine:

```bash
jupyter notebook plot_results.ipynb
```

## 📦 Dependencies

All dependencies are included in the Sinergym Docker image:
- Python 3.12
- Sinergym v3.x
- Gymnasium
- PyTorch (LSTM)
- Stable-Baselines3 (DQN)
- EnergyPlus 9.5+
- pandas, numpy, matplotlib

## 📊 Scripts Description

### `env_demo.py` — DQN Training

Trains a DQN agent on the `Eplus-5zone-mixed-discrete-stochastic-v1` environment.

**Features:**
- 100 training episodes with periodic evaluation
- Applies observation normalization and logging wrappers
- Saves training and evaluation outputs to timestamped directories
- Uses discrete action space (10 actions)

**Environment Wrappers:**
- `NormalizeObservation` — normalizes observations to improve learning
- `LoggerWrapper` — logs environment information
- `CSVLogger` — saves step-by-step data to CSV files

**Output:**
- `outputs/YYYY-MM-DD_HH-MM-SS/DQN_Train-res1/` — training episodes and progress
- `outputs/YYYY-MM-DD_HH-MM-SS/DQN_Eval-res1/` — evaluation episodes and metrics

### `collect_data.py` — Data Collection

Loads the trained DQN model and collects raw observation data over 10 simulated years (2001-2010).

**Features:**
- Runs the trained agent for a 10-year simulation period
- Collects unnormalized observations from the environment
- Saves data to `artifacts/chiller_data.csv` (~350k timesteps)

**Output CSV Columns:**
- Temporal: `month`, `day_of_month`, `hour`
- Weather: `outdoor_temperature`, `outdoor_humidity`, `wind_speed`, `wind_direction`, `diffuse_solar_radiation`, `direct_solar_radiation`
- Indoor: `air_temperature`, `air_humidity`, `people_occupant`
- Setpoints: `htg_setpoint`, `clg_setpoint`
- Energy: `co2_emission`, `HVAC_electricity_demand_rate`, `total_electricity_HVAC`

**Usage:**
```bash
python collect_data.py
```

### `plot_results.ipynb` — Visualization

Jupyter notebook for analyzing and visualizing:
- Training progress curves
- Evaluation metrics
- Episode-level observations, actions, and rewards
- Energy consumption and comfort metrics

## 📈 Output Structure

When you run `env_demo.py`, outputs are organized as follows:

```
outputs/2025-11-04_11-47-51/
├── DQN_Train-res1/
│   ├── episode-1/
│   │   ├── monitor/
│   │   │   ├── observations.csv        # Raw observations per timestep
│   │   │   ├── normalized_observations.csv
│   │   │   ├── actions.csv
│   │   │   ├── rewards.csv
│   │   │   └── infos.csv
│   │   └── output/                     # EnergyPlus simulation outputs
│   ├── episode-2/ ...
│   └── progress.csv                    # Overall training progress
└── DQN_Eval-res1/
    ├── episode-18/ ...
    ├── episode-19/ ...
    └── evaluation/
        └── evaluation_metrics.csv      # Evaluation summary
```

## 🔍 Inspecting Results

### Quick CSV Inspection

```bash
# View evaluation metrics
head -n 30 outputs/*/DQN_Eval-res1/evaluation/evaluation_metrics.csv

# View observations from a training episode
head -n 40 outputs/*/*/monitor/observations.csv

# View collected chiller data
head -n 20 artifacts/chiller_data.csv
```

### Using the Notebook

The `plot_results.ipynb` notebook provides:
- Training reward curves
- Energy vs comfort trade-off analysis
- Action distribution visualization
- Temperature and setpoint tracking

## ⚙️ Environment Details

- **Environment ID:** `Eplus-5zone-mixed-discrete-stochastic-v1`
- **Building Model:** 5-zone mixed-use building with autosized DX VAV system
- **Weather:** New York JFK International Airport TMY3 with stochastic noise
- **Action Space:** Discrete(10) — heating and cooling setpoint combinations
- **Observation Space:** 17-dimensional continuous (temporal, weather, indoor conditions)
- **Timestep:** 15 minutes (900 seconds)
- **Episode Length:** 10 years (2001-2010)

## 🎓 References

- **Sinergym Documentation:** https://ugr-sail.github.io/sinergym/compilation/main/index.html
- **Stable-Baselines3:** https://stable-baselines3.readthedocs.io/
- **EnergyPlus:** https://energyplus.net/

## ⚠️ Troubleshooting

**Issue:** `FileExistsError` when running `collect_data.py`
- **Solution:** The script automatically cleans up old output directories. If it fails, manually delete `outputs/Data_Collection*` folders.

**Issue:** Docker container exits with code 1
- **Solution:** Ensure Docker has sufficient memory (at least 4GB) and disk space. EnergyPlus simulations are resource-intensive.

**Issue:** Missing observations in collected data
- **Solution:** Ensure the `CSVLogger` wrapper is applied to the environment. Check that episode completed successfully without errors.
