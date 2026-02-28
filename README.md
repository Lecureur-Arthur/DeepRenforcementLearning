# ShepherdRL – Progressive Robotic Shepherding (PPO vs DQN)

**ShepherdRL** is a custom reinforcement learning environment for robotic shepherding inspired by biologically grounded flocking dynamics. This project serves as a research testbed to compare **Continuous Vector-based RL (PPO)** against **Visual Image-based RL (DQN)** across progressively difficult environments.

The environment provides a **Gym-style API**, features real-time **Pygame visualization**, and supports automated **Curriculum Learning**.

---

## Key Features & Research Focus

* **Algorithmic Comparison:** Benchmarking PPO (using coordinate vectors) vs. DQN (using CNNs on pixel inputs).
* **Automated Curriculum Learning (`-cl`):** Built-in mechanics to transfer weights, reset optimizers, and halve training time when moving to harder environment levels.
* **Robust Baselines:** Includes deterministic Expert (Rule-Based), Random (Tipsy), and Immobile (Lazy) agents for statistical comparison.
* **Dimensionality Curse Analysis:** Evaluates the models' capacity to scale from a single sheep to a multi-entity flock (3 sheep).

---

## Installation

It is recommended to use a dedicated Python environment:

```bash
conda create -n shepherding python=3.12
conda activate shepherding
pip install -r requirements.txt

```

*(Note: Ensure you have installed `stable-baselines3`, `torch`, and `pygame` via the requirements file).*

---

## Project Structure

```text
shepherd_rl/
│
├─ envs/
│  └─ shepherd_env.py         # Custom Gym environment (physics & rules)
│
├─ agents/
│  ├─ rule_based_agent.py     # Heuristic expert agent
│  ├─ rl_agent.py             # RL utilities & PPO setup
│  └─ CNN_QN.py               # DQN custom agent with PyTorch
│
├─ generate_graphs.py         # 📊 Script to extract Excel data & plot metrics with ± error bars
├─ test.py                    # Run simulations (Rule-based or RL) over N episodes
├─ train.py                   # Train RL agents (Supports Scratch & Curriculum Learning)
├─ models/                    # Saved RL models (.zip / .pth)
└─ README.md

```

---

## The Curriculum: Problem Levels

The environment is structured into progressive levels to train and evaluate the agents. We modified the core physics to introduce stochasticity (`active_sheep`), forcing the agent to anticipate rather than just react.

### Level 1 – Sleepy Sheep (Sanity Check)

* **Rules:** 1 to 3 Sheep. No obstacles. Sheep are static unless the shepherd is within their repulsion radius.
* **Goal:** Learn fundamental pushing trajectories.

### Level 2 – Active Sheep (Stochastic Motion)

* **Rules:** Sheep now possess continuous random motion (`np.random.uniform`) when the shepherd is far.
* **Goal:** Introduce stochasticity and force the agent to correct dynamic trajectories.

### Level 3 – Obstacle-Constrained Shepherding

* **Rules:** A circular obstacle (radius = 0.2) is placed in the center. Neither the sheep nor the shepherd can pass through it.
* **Goal:** Force the agent to unlearn straight-line trajectories and develop obstacle-avoidance strategies.

### Level 4 – Multi-Agent (Work in Progress)

* **Rules:** Two trained shepherd agents alternating actions to control a larger flock.

---

## Agents Available

### Reinforcement Learning Agents

1. **PPO (Proximal Policy Optimization):** Exploits a continuous observation vector (X/Y coordinates).
2. **DQN (Deep Q-Network):** Exploits a visual representation of the environment using a Convolutional Neural Network (CNN).

### Baseline Agents

* **Rule-Based (Expert):** Computes a driving point behind the furthest sheep to push it toward the goal. Deterministic and highly efficient.
* **Lazy Shepherd:** Stays completely static (used to measure the environment's base entropy).
* **Tipsy Shepherd:** Takes entirely random actions.

---

## Usage & CLI Examples

The project is driven by highly parameterized CLI scripts (`train.py` and `test.py`).

**1. Test a Baseline (e.g., Rule-Based on 3 sheep):**

```bash
python test.py -t ruleBase -s 3 -n 1000

```

**2. Train PPO from Scratch (Level 1):**

```bash
python train.py -a ppo -s 1

```

**3. Curriculum Learning (Train Level 2 using Level 1 weights):**

```bash
python train.py -a ppo -s 1 --active_sheep -c models/ppo_sheep1_obst0_sleepy.zip -cl

```

**4. Evaluate a trained DQN model on Level 3 with Obstacles:**

```bash
python test.py -t DQN -a models/dqn_sheep1_obst2_active_best.pth -s 1 -r 0.2 --active_sheep -n 100

```

---

## Key Findings

* **State Representation Matters:** The coordinate-based PPO agent consistently outperformed the image-based DQN agent in both Success Rate and Average Reward stability.
* **Curriculum Learning is double-edged:** Fine-tuning via Curriculum Learning drastically improved performance when adapting to dynamic motion (Level 1 → Level 2). However, when facing a topological shift (adding an obstacle in Level 3), training from scratch yielded better results, as the agent didn't have to "unlearn" its straight-line bias.
* **The Dimensionality Curse:** Scaling the observation vector from 1 to 3 sheep without collective metrics (like center of mass) caused deep learning models to collapse (sub-5% success rate), highlighting the need for advanced state representations in multi-agent tracking.

---

## Future Extensions

* Implementing Reward Shaping (Curriculum on the goal radius size for high-precision herding).
* Providing collective flock metrics (center of gravity, dispersion radius) to solve the 3-sheep dimensionality curse.
* Training Level 4 (Multi-Agent shepherds).
