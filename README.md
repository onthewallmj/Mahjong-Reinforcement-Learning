# Mahjong Reinforcement Learning Environment

A Python-based simulation environment for Mahjong (Hong Kong Old Style variants), designed for Multi-Agent Reinforcement Learning (MARL). This project implements the core mechanics of Mahjong, including tile dealing, meld mechanics (Pong, Kong, Chow), scoring, and game flow management, wrapped in standard RL interfaces (Gymnasium and PettingZoo).

## Project Structure

-   **`train.py`**: Script to train a PPO agent using Stable Baselines3 and the PettingZoo environment.
-   **`main.py`**: A standalone simulation runner for testing the game logic without RL overhead.
-   **`mahjong/`**: Package containing the core game logic and RL wrappers.
    -   **`pettingzoo_env.py`**: **(New)** PettingZoo Parallel Environment for 4-player self-play training.
    -   **`env.py`**: **(New)** Gymnasium Environment for single-agent training against bots.
    -   **`game.py`**: Manages the game state machine, turn logic, and rules. Refactored to support step-wise execution for RL.
    -   **`player.py`**: Defines the `Player` class and decision interfaces.
    -   **`observation.py`**: **(New)** Encodes the game state into a (33, 34) tensor for Neural Networks.
    -   **`action_space.py`**: **(New)** Defines the discrete action space (0-41).
    -   **`hand_scorer.py`**: Implements scoring logic and win condition checks.

## Features

-   **Reinforcement Learning Ready**:
    -   **PettingZoo Integration**: Supports Multi-Agent training (4 agents) with full rotation episodes.
    -   **Gymnasium Integration**: Supports Single-Agent training.
    -   **Tensor Observation**: Efficient state representation for CNNs/MLPs.
    -   **Action Masking**: Prevents illegal moves to stabilize training.
    -   **Sparse Rewards**: Supports tournament-style rewards based on final rank.
-   **Complete Game Engine**:
    -   Handles all Meld types (Chow, Pong, Kong, Concealed/Promoted Kongs).
    -   Scoring system for standard and special hands (Thirteen Orphans, Seven Pairs).
    -   Correct dealer rotation and table wind rules.

## Usage

### Training an Agent (RL)

To train a Mahjong agent using Proximal Policy Optimization (PPO) in a self-play setup:

```bash
python train.py
```

This script will:
1.  Initialize the `MahjongPettingZooEnv`.
2.  Wrap it for compatibility with Stable Baselines3.
3.  Train a single policy that controls all 4 players (Parameter Sharing).
4.  Save the trained model to `mahjong_ppo_model.zip`.

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
