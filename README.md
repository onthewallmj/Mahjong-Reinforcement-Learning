# Mahjong Reinforcement Learning Environment

A Python-based simulation environment for Mahjong (Hong Kong Old Style variants), designed for Multi-Agent Reinforcement Learning (MARL). This project implements the core mechanics of Mahjong, including tile dealing, meld mechanics (Pong, Kong, Chow), scoring, and game flow management, wrapped in standard RL interfaces (Gymnasium and PettingZoo).

## Project Structure

-   **`train.py`**: Script to train a PPO agent using `MaskablePPO` (from `sb3-contrib`) and the PettingZoo environment.
-   **`train_curriculum.py`**: Advanced training script using a 2-phase curriculum (Bots -> Self-Play).
-   **`evaluate.py`**: Script to benchmark a trained agent against bots or in self-play.
-   **`main.py`**: A standalone simulation runner for testing the game logic without RL overhead.
-   **`mahjong/`**: Package containing the core game logic and RL wrappers.
    -   **`pettingzoo_env.py`**: PettingZoo Parallel Environment for 4-player self-play training. Implements Action Masking.
    -   **`envs/vs_bot_env.py`**: Gymnasium wrapper for Single-Agent vs Heuristic Bots.
    -   **`agents/heuristic_agent.py`**: Rule-based bot logic.
    -   **`feature_extractor.py`**: Custom 1D CNN for feature extraction.
    -   **`game.py`**: Manages the game state machine, turn logic, and rules.
    -   **`player.py`**: Defines the `Player` class and decision interfaces.
    -   **`observation.py`**: Encodes the game state into a (33, 34) tensor for Neural Networks.
    -   **`action_space.py`**: Defines the discrete action space (0-41).
    -   **`hand_scorer.py`**: Implements scoring logic and win condition checks.

## System Architecture & Concepts

### 1. Environment Frameworks

This project utilizes two distinct frameworks to support different training paradigms:

-   **PettingZoo (`mahjong/pettingzoo_env.py`)**:
    -   **Purpose**: Multi-Agent RL (MARL).
    -   **Setup**: The RL policy controls **all 4 players** simultaneously.
    -   **Use Case**: Self-play training. This allows the agent to learn advanced strategies by playing against copies of itself, creating an "arms race" of capability. We use the `ParallelEnv` API for compatibility with high-performance vectorization tools.

### 2. Neural Network Architecture (Custom CNN)

The project utilizes a specialized Deep Learning architecture designed to capture the spatial dependencies of Mahjong tiles:

-   **Custom Feature Extractor**: A 1D Convolutional Neural Network (ResNet-like).
    -   **Input**: `(33, 34)` Tensor.
    -   **Architecture**: 3 layers of `Conv1d` (kernel size 3) with ReLU activations and Batch Normalization.
    -   **Purpose**: The `Conv1d` layers scan across the 34 tile types to detect **Sequences** (Chows) and **Triplets** (Pongs) regardless of their suit, mimicking how human players recognize patterns.
-   **Policy Network**: `MaskablePPO` with an MLP head on top of the extracted features.

### 3. State Representation (Observation Space)

The complex game state is encoded into a **`(33, 34)` floating-point tensor**, optimized for the CNN to process:

-   **Dimensions**: 33 Feature Channels × 34 Unique Tile Types.
-   **34 Columns**: Represent the 34 distinct tiles (Characters 1-9, Bamboo 1-9, Dots 1-9, Winds E/S/W/N, Dragons R/G/W).
-   **33 Channels**: Binary planes capturing specific information, such as:
    -   **Hand Composition**: Which tiles are currently in the agent's hand (one-hot encoded for count).
    -   **Public Information**: Open Melds (Pong/Kong/Chow) of all players.
    -   **Discard History**: What has been played on the table (crucial for defensive play).
    -   **Game Context**: Dealer status, current Table Wind, and "Riichi" status (Tenpai).

### 4. Decision Making (Action Space)

The agent interacts with the game via **42 Discrete Actions**:

-   **Actions 0-33 (Discard)**: Discard a specific tile corresponding to the 34 tile types.
-   **Action 34 (Skip)**: Pass priority (decline to call Chow/Pong/Kong).
-   **Actions 35-41 (Declarations)**: Specific calls for Chow (Low/Mid/High), Pong, Kong, Self-Kong, and Win.

**Action Masking**: The environment calculates a validity mask at every step. This prevents the agent from making illegal moves (e.g., discarding a tile it doesn't hold), significantly speeding up the learning process. We use `MaskablePPO` to leverage this.

### 5. Multi-Agent Reinforcement Learning (MARL)

-   **Self-Play with Parameter Sharing**: The training script (`train.py`) uses a single Neural Network (PPO Policy) to control **all 4 players**. The AI learns by playing against copies of itself, evolving from random moves to strategic play.
-   **Full Rotation Episodes**: A Mahjong match isn't just one hand. The environment simulates a **Full Rotation** (East Round → South Round → West Round → North Round), comprising 16+ individual hands. This forces the agent to consider long-term score preservation.

### 6. Reward Structure

The environment utilizes a combination of **Dense Intermediate Rewards** and **Sparse Tournament Rewards** to balance immediate feedback with long-term strategic goals.

-   **Dense Intermediate Rewards (Per Hand)**:
    -   **Winning a Hand (Win)**: `+10.0`. Awarded immediately to the winner of any hand.
    -   **Dealing In (Penalty)**: `-10.0`. Applied to the player who discards the winning tile (feeding the winner).
    -   **Other Players**: `0`.

-   **Sparse Tournament Rewards (Per Game Rotation)**:
    At the end of the full table rotation (East → North), players are ranked by their total accumulated score.
    -   **1st Place**: `+100`
    -   **2nd Place**: `+50`
    -   **3rd Place**: `-50`
    -   **4th Place**: `-100`

### 7. Curriculum Learning Strategy

To bootstrap the agent's learning, we employ a two-phase curriculum:

1.  **Phase 1: Rule-Based Training**:
    -   The agent trains against 3 **Heuristic Bots** (`mahjong/agents/heuristic_agent.py`).
    -   These bots follow basic strategies (discard isolated winds/terminals, win when able), providing a stable baseline for the agent to learn the rules and basic hand composition.
    -   Run via `train_curriculum.py`.

2.  **Phase 2: Self-Play Fine-Tuning**:
    -   The trained model from Phase 1 is loaded into the Multi-Agent Self-Play environment.
    -   The agent plays against copies of itself to discover advanced strategies that exploit the specific dynamics of high-level play.

## Usage

### Training

#### Curriculum Learning (Recommended)
To train using the 2-phase curriculum (Bots -> Self-Play):
```bash
python train_curriculum.py
```
This is the standard way to train a strong agent. Logs are saved to `./mahjong_curriculum_logs/`.

#### Standard Self-Play (From Scratch)
To train from scratch using only self-play:
```bash
python train.py
```

### Evaluation

To evaluate a trained agent (or a random agent if no model is found):

```bash
python evaluate.py --episodes 100 --vs-random
```
-   `--episodes`: Number of games to play.
-   `--vs-random`: If set, plays against 3 random bots. If omitted, runs self-play evaluation.

### Visualization (TensorBoard)

To view training progress (rewards, losses):

```bash
tensorboard --logdir ./mahjong_curriculum_logs/
```
Open the link (usually `http://localhost:6006`) in your browser.

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
-   `sb3-contrib` (for MaskablePPO)
-   `supersuit`
-   `tensorboard`
-   `shimmy`

## Development Status

-   [x] Core Game Logic & Scoring
-   [x] RL Observation & Action Space
-   [x] Gymnasium Wrapper (Single Agent)
-   [x] PettingZoo Wrapper (Multi-Agent)
-   [x] Action Masking (Discard Phase)
-   [x] Training Pipeline (PPO)
-   [x] Custom CNN Feature Extractor
-   [x] Curriculum Learning (Bots -> Self-Play)
-   [ ] **Advanced Action Masking**: Currently, agents control Discards. Reaction logic (deciding to Chow/Pong/Kong) uses heuristics or is skipped. Implementing full agent control for reactions is the next major milestone.
