## File Structure

```text
.
├── artifacts/              # Pre-trained models and data
│   ├── chiller_data.csv
│   ├── chiller_lstm_model.keras
│   ├── dqn_model.zip
│   └── lstm_pipeline.pth
├── notebooks/              # Jupyter notebooks for analysis and plotting
│   ├── eda.ipynb
│   ├── lstm.ipynb
│   ├── plot_env.ipynb
│   └── plot_results.ipynb
├── outputs/                # Generated outputs (logs, results)
├── sensitivty_analysis/    # Scripts/results for sensitivity analysis
│   ├── dqn/
│   └── lstm/
├── src/                    # Source code
│   ├── collect_data.py
│   ├── train_base.py
│   ├── train_dqn.py
│   └── train_lstm.py
└── README.md
```

## Quick Start (Docker)

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
bash script.sh
exit
```
