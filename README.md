# Mahjong AI (Reinforcement Learning)

A Reinforcement Learning project for training AI agents to play Hong Kong Style Mahjong using Multi-Agent Reinforcement Learning (MARL) and Deep Learning techniques.

This project implements a complete Mahjong game environment compatible with PettingZoo and Stable Baselines3, enabling training of RL agents through self-play and curriculum learning strategies.

---

## Quick Start

### Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

### Training

The training pipeline uses a two-phase curriculum learning approach:

-   **Phase 1**: Train the agent against heuristic rule-based bots to learn basic gameplay
-   **Phase 2**: Continue training through self-play to develop advanced strategies

Run training:

```bash
python train_curriculum.py
```

Training progress can be monitored using TensorBoard (see Visualization section).

### Evaluation

Evaluate a trained model against random agents:

```bash
python evaluate.py --episodes 100 --vs-random
```

For detailed per-game output including winning hands and point sources:

```bash
python evaluate.py --episodes 50 --vs-random --render
```

---

## Architecture

### Neural Network Architecture

The agent uses a custom Convolutional Neural Network (CNN) feature extractor designed for Mahjong's tile-based observation space. The CNN processes the 33-channel observation tensor (representing game state across 34 tile types) to detect patterns such as sequences, triplets, and honor tiles.

### Action Masking

Action masking prevents illegal moves by dynamically filtering the action space based on the current game state. The agent can only select valid actions, such as:

-   Discarding tiles that exist in the player's hand
-   Declaring wins when a valid winning hand is present
-   Reacting (Chow/Pong/Kong) when conditions are met

This constraint significantly improves learning efficiency by eliminating invalid exploration.

### Reward Structure

The reward system combines dense intermediate rewards with sparse tournament rewards:

-   **Intermediate Rewards**:

    -   `+10.0`: Winning a hand (Ron/Tsumo)
    -   `-10.0`: Dealing into an opponent's win

-   **Tournament Rewards** (end of full rotation):
    -   `+100`: Highest final table score
    -   `+50`: Second place
    -   `-50`: Third place
    -   `-100`: Lowest score

This dual reward structure encourages both immediate hand-winning strategies and long-term tournament performance.

---

## Project Structure

-   **`train_curriculum.py`**: Main training script implementing the two-phase curriculum
-   **`evaluate.py`**: Evaluation script for testing trained models
-   **`mahjong/`**: Core game engine and RL environment
    -   **`game.py`**: Game state management, turn logic, and rule enforcement
    -   **`pettingzoo_env.py`**: PettingZoo ParallelEnv implementation for multi-agent RL
    -   **`feature_extractor.py`**: Custom CNN feature extractor for tile pattern recognition
    -   **`action_space.py`**: Action space definition (42 discrete actions)
    -   **`observation.py`**: Observation space builder (33 channels × 34 tile types)
    -   **`agents/heuristic_agent.py`**: Rule-based bot for Phase 1 training
    -   **`envs/vs_bot_env.py`**: Single-agent Gym environment wrapper for Phase 1
    -   **`sb3_wrappers.py`**: Compatibility wrapper for Stable Baselines3 integration

---

## Visualization

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir ./mahjong_curriculum_logs/
```

Access the dashboard at `http://localhost:6006` to view metrics including:

-   Average episode reward
-   Policy loss
-   Value loss
-   Entropy (exploration)

---

## Development Status

-   [x] Core game engine with full Mahjong rule implementation
-   [x] Custom CNN feature extractor for tile pattern recognition
-   [x] Curriculum learning pipeline (heuristic bots → self-play)
-   [x] Action masking for illegal move prevention
-   [x] Dense intermediate rewards and sparse tournament rewards
-   [x] Multi-agent PettingZoo environment with parameter sharing
-   [ ] Advanced reaction strategy learning (currently uses heuristic rules for Chow/Pong/Kong decisions)

---

## Technical Details

### Environment

-   **Framework**: PettingZoo (ParallelEnv) for multi-agent RL
-   **RL Library**: Stable Baselines3 (MaskablePPO)
-   **Observation Space**: `Box(0, 1, (33, 34), float32)` - 33 channels × 34 tile types
-   **Action Space**: `Discrete(42)` - 34 discard actions + 8 reaction/declaration actions

### Training Configuration

-   **Algorithm**: MaskablePPO (Proximal Policy Optimization with action masking)
-   **Learning Rate**: `3e-4`
-   **Batch Size**: `64`
-   **Gamma**: `0.99` (discount factor)
-   **Entropy Coefficient**: `0.01` (encourages exploration)

---

## License

[Add your license here]
