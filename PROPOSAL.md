# ORA Term Project Proposal

### 1. Title

Predictive Reinforcement Learning for Stochastic HVAC Control: An LSTM-Augmented DQN Approach

### 2. Background and Motivation

#### 2.1 Motivation
Building energy consumption constitutes a significant portion of global energy demand, with Heating, Ventilation, and Air Conditioning (HVAC) systems being a primary contributor. The development of intelligent, autonomous control systems to optimize HVAC operations can yield substantial economic savings through reduced energy expenditure and societal benefits through a smaller environmental footprint, all while maintaining occupant comfort. This project seeks to apply operations research methodologies to this real-world optimization problem.

#### 2.2 Background
The control of HVAC systems is a complex sequential decision-making problem. The environment is highly **stochastic**, driven by uncertain factors such as external weather (temperature, humidity, solar radiation) and internal occupancy loads. Traditional controllers, such as Rule-Based Controllers (RBCs), are purely reactive and often fail to operate optimally in such dynamic conditions.

Deep Reinforcement Learning (RL) has emerged as a powerful framework for solving complex control problems, capable of learning optimal policies directly from interaction. We will utilize `Sinergym`, a standardized research-oriented Python library, which provides a simulation environment by wrapping the industry-standard **EnergyPlus** simulator in a `gymnasium` interface.

#### 2.3 Problem Definition
This project proposes and evaluates a predictive reinforcement learning framework to optimize HVAC control setpoints in a simulated 5-zone office building. The objective is to learn a control policy that **minimizes a total cost function**, defined as a weighted sum of energy consumption costs and occupant thermal comfort violations. We will formally implement and benchmark two distinct agent architectures:
1.  **A Baseline Model:** A purely reactive Deep Q-Network (DQN) agent.
2.  **A Proposed Model:** A proactive, LSTM-augmented DQN agent that leverages predictive load forecasting.

### 3. Methodology

#### 3.1 Method Justification
Per the project guidelines, we first establish a **baseline model** for performance comparison.
* **Baseline Model (Reactive DQN):** We employ a Deep Q-Network (DQN) as our baseline. DQN is a state-of-the-art, value-based RL algorithm well-suited for high-dimensional state spaces and discrete action spaces, which aligns with our environment's configuration. This agent learns a **reactive policy**, $\pi_A(a_t | s_t)$, based only on the *current* observed state $s_t$. Its primary limitation is its inability to anticipate and plan for future disturbances.
* **Proposed Model (Predictive LSTM+DQN):** To address the baseline's limitation, our proposed model enhances the agent's decision-making by providing it with "a peek into the future." This innovative approach  involves two components:
    1.  **A Predictor (LSTM):** A Long Short-Term Memory (LSTM) network is chosen for its proven ability to model and learn patterns from time-series data. It will be trained to predict the next timestep's HVAC energy load.
    2.  **A Controller (DQN):** This DQN agent learns a **proactive policy**, $\pi_B(a_t | s_t, \hat{l}_{t+1})$, where the state $s_t$ is *augmented* with the LSTM's load prediction $\hat{l}_{t+1}$. We hypothesize this predictive information will allow the agent to learn a more robust and efficient policy.

#### 3.2 Theoretical Introduction
We formulate this problem as a **Markov Decision Process (MDP)** with the following components:

* **State ($s_t$):** A vector of all observations from the `Sinergym` environment at time $t$, including `outdoor_temperature`, `air_temperature`, `people_occupant`, `HVAC_electricity_demand_rate`, etc.
* **Action ($a_t$):** A discrete action $a \in A$, which the environment maps to a specific pair of heating and cooling setpoints.
* **Reward ($r_t$):** A linear combination of penalties, $r_t = - (w_e \times \text{EnergyPenalty}_t + w_c \times \text{ComfortPenalty}_t)$, where $w_e$ and $w_c$ are weighting factors.
* **Objective:** To find an optimal policy $\pi^*$ that maximizes the expected discounted return, $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$.

**Model A (DQN):**
Approximates the optimal action-value function, $Q^*(s, a)$, using a neural network $Q(s, a; \theta)$ trained to minimize the Bellman error. The policy is $\epsilon$-greedy with respect to the learned Q-values.


**Model B (LSTM+DQN):**
This is a two-stage model:
1.  **Predictor Training:** An LSTM is trained via supervised learning on historical data to predict the next-step load, $\hat{l}_{t+1} \approx l_{t+1}$, where $l_t$ is the `HVAC_electricity_demand_rate` at time $t$. The loss function is $L_{\text{LSTM}} = \mathbb{E}[(\hat{l}_{t+1} - l_{t+1})^2]$.
    
2.  **Controller Training:** A new DQN is trained, but its state input is the augmented vector $s_t' = \text{concat}(s_t, \hat{l}_{t+1})$.

### 4. Data Collection and Analysis Result

#### 4.1 Data Collection
To train the LSTM predictor, a time-series dataset is required. As we do not have a real-world dataset, we will utilize the `Sinergym` environment as a **Data Generating Process (DGP)**. We will execute our trained Baseline Model (Model A) policy for multiple simulated years, logging the full state vector $s_t$ and the target variable $l_t$ at every timestep. This will generate a comprehensive `chiller_data.csv` file.

#### 4.2 Analysis
The analysis will be conducted as a formal, multi-step experiment:
1.  **LSTM Training:** Pre-process the `chiller_data.csv` into time-series sequences. Train and validate the LSTM model, saving the final model (`.pth`) and the data scalers (`.pkl`) for inference.
2.  **Integration:** Implement a custom `gymnasium.ObservationWrapper` (a "Predictive Wrapper") that loads the trained LSTM and scalers. This wrapper will intercept observations, maintain a state history, and perform on-the-fly prediction to append $\hat{l}_{t+1}$ to the state vector for the agent.
3.  **Controller Training (Model B):** Train a new DQN agent from scratch using the environment stacked with the `PredictiveWrapper`.
4.  **Benchmarking:** Run both the Baseline (Model A) and Predictive (Model B) agents through a series of identical evaluation episodes (simulated years) to collect performance data.

#### 4.3 Results and Managerial Implications
We will present a comparative analysis of the two models using figures and tables.
* **Primary Result:** The primary metric for comparison will be the `mean_reward` vs. `episode_num` plot, visualizing the learning speed and final converged performance of both agents.
* **Managerial Implication:** We hypothesize that the Predictive Model will converge to a higher (less negative) final reward. If this hypothesis is supported, the key managerial implication is that **investing in predictive load-forecasting technology can yield a quantifiable reduction in operational energy costs** and/or improvement in occupant comfort, providing a clear justification for its implementation over purely reactive control systems.

### 5. Conclusion
This project will design, implement, and formally benchmark a novel predictive-RL architecture for stochastic HVAC control. By comparing a standard, reactive DQN baseline against an LSTM-augmented predictive agent, this work will provide a data-driven, quantitative analysis of the operational value of predictive information in this complex domain. The findings will serve as both a practical application of ORA methodologies and a tutorial for future students.