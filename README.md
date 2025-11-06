# ORA_HW — Final Project

This repository holds the final project for an energy control problem using reinforcement learning (DQN) to control building HVAC systems. The repository stores experiment outputs, per-episode simulation artifacts, and helper code for inspecting results.

## Highlights

- Experiments and simulation outputs are stored under `outputs/` (large, generated files).
- The environment wrapper and experiment entry point is `env.py`.
- `plot_results.ipynb` is provided to visualize training and evaluation logs.

## Getting started (Docker + Sinergym)

This project was developed against Sinergym. We utilize the official Sinergym Docker image. 

The steps below run `env.py` inside the container and mount the repository into `/app` so generated outputs are written to your local `outputs/` directory.

1. Open a terminal and change into the repository root:

```bash
cd /path/to/this/repo
```

2. Pull the official Sinergym image:

```bash
docker pull sailugr/sinergym:v3.10.0
```

3. Start an interactive container and mount the repository:

```bash
docker run -it --rm -v "$(pwd)":/app sailugr/sinergym:v3.10.0 /bin/bash
```

4. Inside the container, run:

```bash
cd /app
python env.py
```

Notes:
- The specific Sinergym version above (v3.10.0) was used when developing the experiments; adjust the tag if you need a different release.
- If you don't use Docker, ensure the Python environment has the required packages (Sinergym, gym, numpy, pandas, etc.).

## Repository layout

- `env.py` — environment wrapper / runner used to launch experiments or evaluate saved policies.
- `plot_results.ipynb` — Jupyter notebook for visualizing `progress.csv`, monitor CSVs, and evaluation metrics.
- `outputs/` — generated experiment outputs (EnergyPlus files, per-episode folders, monitor CSVs).

Sample `outputs/` structure:

```
outputs/
  2025-11-04_11-47-51/
    DQN_Eval-res1/
      episode-18/
        monitor/ (agent_actions.csv, observations.csv, rewards.csv, ...)
      episode-19/
    DQN_Train-res1/
      episode-100/
      evaluation/ (evaluation_metrics.csv)
```

## Inspecting results

- Open `plot_results.ipynb` in Jupyter/Lab to visualize training curves and episode traces.
- For a quick CSV peek, inspect:

```bash
head -n 30 outputs/*/DQN_*/evaluation/evaluation_metrics.csv
head -n 40 outputs/*/*/monitor/observations.csv
```
