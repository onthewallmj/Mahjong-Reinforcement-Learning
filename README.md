# Mahjong Reinforcement Learning Environment

A Python-based simulation environment for Mahjong (Hong Kong Old Style variants), designed for Multi-Agent Reinforcement Learning (MARL). This project implements the core mechanics of Mahjong, including tile dealing, meld mechanics (Pong, Kong, Chow), scoring, and game flow management, wrapped in standard RL interfaces (Gymnasium and PettingZoo).

## Project Structure

-   **`train.py`**: Script to train a PPO agent using Stable Baselines3 and the PettingZoo environment.
-   **`main.py`**: A standalone simulation runner for testing the game logic without RL overhead.
-   **`evaluate.py`**: Script to benchmark a trained agent against bots.
-   **`mahjong/`**: Package containing the core game logic and RL wrappers.
    -   **`pettingzoo_env.py`**: **(New)** PettingZoo Parallel Environment for 4-player self-play training.
    -   **`game.py`**: Manages the game state machine, turn logic, and rules. Refactored to support step-wise execution for RL.
    -   **`player.py`**: Defines the `Player` class and decision interfaces.
    -   **`observation.py`**: **(New)** Encodes the game state into a (33, 34) tensor for Neural Networks.
    -   **`action_space.py`**: **(New)** Defines the discrete action space (0-41).
    -   **`hand_scorer.py`**: Implements scoring logic and win condition checks.

## System Architecture & Concepts

### 1. Environment Frameworks

This project utilizes two distinct frameworks to support different training paradigms:

-   **PettingZoo (`mahjong/pettingzoo_env.py`)**:
    -   **Purpose**: Multi-Agent RL (MARL).
    -   **Setup**: The RL policy controls **all 4 players** simultaneously.
    -   **Use Case**: Self-play training. This allows the agent to learn advanced strategies by playing against copies of itself, creating an "arms race" of capability. We use the `ParallelEnv` API for compatibility with high-performance vectorization tools.

### 2. State Representation (Observation Space)

The complex game state is encoded into a **`(33, 34)` floating-point tensor**, optimized for neural networks (like CNNs) to process:

-   **Dimensions**: 33 Feature Channels × 34 Unique Tile Types.
-   **34 Columns**: Represent the 34 distinct tiles (Characters 1-9, Bamboo 1-9, Dots 1-9, Winds E/S/W/N, Dragons R/G/W).
-   **33 Channels**: Binary planes capturing specific information, such as:
    -   **Hand Composition**: Which tiles are currently in the agent's hand (one-hot encoded for count).
    -   **Public Information**: Open Melds (Pong/Kong/Chow) of all players.
    -   **Discard History**: What has been played on the table (crucial for defensive play).
    -   **Game Context**: Dealer status, current Table Wind, and "Riichi" status (Tenpai).

### 3. Decision Making (Action Space)

The agent interacts with the game via **42 Discrete Actions**:

-   **Actions 0-33 (Discard)**: Discard a specific tile corresponding to the 34 tile types.
-   **Action 34 (Skip)**: Pass priority (decline to call Chow/Pong/Kong).
-   **Actions 35-41 (Declarations)**: Specific calls for Chow (Low/Mid/High), Pong, Kong, Self-Kong, and Win.

**Action Masking**: The environment calculates a validity mask at every step. This prevents the agent from making illegal moves (e.g., discarding a tile it doesn't hold), significantly speeding up the learning process.

### 4. Multi-Agent Reinforcement Learning (MARL)

-   **Self-Play with Parameter Sharing**: The training script (`train.py`) uses a single Neural Network (PPO Policy) to control **all 4 players**. The AI learns by playing against copies of itself, evolving from random moves to strategic play.
-   **Full Rotation Episodes**: A Mahjong match isn't just one hand. The environment simulates a **Full Rotation** (East Round → South Round → West Round → North Round), comprising 16+ individual hands. This forces the agent to consider long-term score preservation.

### 5. Reward Structure

The environment utilizes a **Sparse Tournament Reward** signal to encourage long-term strategic planning over greedy, immediate point accumulation.

-   **Immediate Rewards**: 0. The agent receives no reward for intermediate actions (discards, melding) or even for winning individual hands during the rotation.
-   **Terminal Rewards**: At the end of the full table rotation (East → North), players are ranked by their total accumulated score.
-   **Payoff Matrix**:
    -   **1st Place**: `+100`
    -   **2nd Place**: `+50`
    -   **3rd Place**: `-50`
    -   **4th Place**: `-100`

This structure mimics the actual incentives of competitive Mahjong, where the goal is to finish the match with the highest standing, sometimes requiring players to play defensively to protect a lead or aggressively to close a gap.

## Usage

### Training an Agent (RL)

To train a Mahjong agent using Proximal Policy Optimization (PPO) in a self-play setup:

```bash
python train.py
```

This script will:

1.  Initialize the `MahjongPettingZooEnv`.
2.  Wrap it using `SuperSuit` to vectorize the 4-player parallel environment.
3.  Train a PPO agent via `Stable Baselines3`.
4.  Save the trained model to `mahjong_ppo_model.zip`.

### Evaluation

To evaluate a trained agent (or a random agent if no model is found):

```bash
python evaluate.py
```

This script will run a specified number of episodes against the internal bots and report the Win Rate, Average Reward, and Average Episode Length.

### Running a Simulation (No RL)

To see the game logic in action with random moves:

```bash
python main.py
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

-   `numpy`
-   `gymnasium`
-   `pettingzoo`
-   `stable-baselines3`
-   `supersuit`

## Development Status

-   [x] Core Game Logic & Scoring
-   [x] RL Observation & Action Space
-   [x] Gymnasium Wrapper (Single Agent)
-   [x] PettingZoo Wrapper (Multi-Agent)
-   [x] Action Masking (Discard Phase)
-   [x] Training Pipeline (PPO)
-   [ ] **Advanced Action Masking**: Currently, agents control Discards. Reaction logic (deciding to Chow/Pong/Kong) uses heuristics or is skipped. Implementing full agent control for reactions is the next major milestone.
