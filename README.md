# Mahjong Reinforcement Learning Environment

A Python-based simulation environment for Mahjong, designed for reinforcement learning experiments. This project implements the core mechanics of Mahjong (Hong Kong Old Style variants), including tile dealing, meld mechanics (Pong, Kong, Chow), scoring, and game flow management.

## Project Structure

-   **`main.py`**: The entry point for the simulation. Runs a game loop until a full table revolution (East, South, West, North rounds) is completed.
-   **`mahjong/game.py`**: Contains the `Game` class, which manages the overall game state, turn logic, dealer rotation, and table wind.
-   **`mahjong/player.py`**: Defines the `Player` class, handling individual player state (hand, melds, score) and decision-making interface.
-   **`mahjong/hand_scorer.py`**: Implements the scoring logic, including checks for standard winning hands (4 sets + 1 pair) and special hands (e.g., Thirteen Orphans).
-   **`mahjong/tile.py`**: Defines the `Tile` class and enums for suits and values.
-   **`mahjong/meld.py`**: Defines the `Meld` class for representing sets of tiles (Chow, Pong, Kong).
-   **`mahjong/game_history.py`**: Tracks the history of played games.
-   **`mahjong/win.py`**: Defines win conditions and score calculation.
-   **`mahjong/point_source.py`**: Manages point values for different scoring elements.
-   **`mahjong/common.py`**: Shared enums and constants (e.g., `Wind`).

## Features

-   **Complete Game Loop**: Handles dealing, drawing, discarding, and turn progression.
-   **Meld Support**: Fully supports Chow, Pong, and Kong (including concealed and promoted/self-Kongs).
-   **Scoring System**:
    -   Detects standard hands (4 melds + 1 pair).
    -   Supports special hands like Seven Pairs and Thirteen Orphans.
    -   Calculates points based on hand composition and table rules.
-   **Game Flow**:
    -   Dealer rotation and retention rules.
    -   Table wind progression.
    -   Win on discard and self-draw logic.
    -   Draw game handling.

## Usage

Run the simulation by executing `main.py`:

```bash
python main.py
```

This will simulate a series of Mahjong games until a full table cycle is complete, printing the final scores of all players.

## Development Status

The core game logic is implemented, including:

-   [x] Tile shuffling and dealing
-   [x] Turn mechanics
-   [x] Valid move generation (Chow, Pong, Kong, Win)
-   [x] Hand scoring and validation
-   [ ] **Policy Implementation**: The `Player` class currently uses placeholder logic for decisions (`wants_to_...` methods). Implementing AI policies (Random, Heuristic, RL-based) is the next step.
