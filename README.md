# 🀄 Mahjong AI (Reinforcement Learning)

A project to teach a computer how to play Hong Kong Style Mahjong using Artificial Intelligence.

Imagine training a dog: you give it a treat when it does something right (like sitting) and ignore it (or say "no") when it does something wrong. We do the same here: we let the AI play thousands of games, and every time it wins a hand or the tournament, we give it "points" (rewards). Over time, it learns which moves lead to points.

This project uses advanced techniques like **Multi-Agent Reinforcement Learning (MARL)** (where 4 AIs play against each other to get smarter) and **Deep Learning** (using a brain-like network to "see" the board).

---

## 🚀 Quick Start

### 1. Install Dependencies
First, make sure you have Python installed. Then run:
```bash
pip install -r requirements.txt
```

### 2. Train the AI (The "School")
We use a **Curriculum** strategy, like sending a kid to school:
-   **Phase 1 (Elementary School)**: The AI plays against simple bots to learn the basic rules (how to form sets, how to win).
-   **Phase 2 (University)**: The AI plays against itself (Self-Play) to learn advanced strategies and bluffing.

Run the training with:
```bash
python train_curriculum.py
```
*This will take a while! You can watch the progress using TensorBoard (see below).*

### 3. Watch it Play
Once trained, you can see how good it is by making it play 100 games against random players:
```bash
python evaluate.py --episodes 100 --vs-random
```

---

## 🧠 How It Works (Simplified)

### The "Brain" (Neural Network)
Think of the AI's brain as a pair of special glasses that looks at the Mahjong table.
-   **Standard AI**: Sees the table as a boring list of numbers.
-   **Our AI (CNN)**: Uses a **Convolutional Neural Network**. It scans the tiles like a human does, looking for patterns: *"Oh, I have a 1-2-3 Bamboo sequence here!"* or *"I have three Red Dragons there!"*.

### The "Referee" (Action Masking)
In Mahjong, you can't just do anything. You can't declare a win if you don't have a winning hand.
-   To help the AI learn faster, we have a built-in "Referee" (Action Masking).
-   If the AI tries to make an illegal move (like discarding a tile it doesn't have), the Referee blocks it immediately. This forces the AI to only think about *valid* moves.

### The "Scoreboard" (Rewards)
How does the AI know if it's doing well? We give it points:
-   **+10 Points**: For winning a single hand (Ron/Tsumo).
-   **-10 Points**: For dealing into someone else's win (feeding the winner).
-   **+100 Points**: For winning the entire tournament (4 rounds).

This mix encourages the AI to win hands quickly but also play defensively to win the long game.

---

## 📂 Project Structure for Techies

-   **`train_curriculum.py`**: The main script to run the 2-phase training.
-   **`mahjong/`**: The core game engine.
    -   **`game.py`**: The rulebook. Handles turns, drawing, and discarding.
    -   **`pettingzoo_env.py`**: The "gym" where 4 agents play together.
    -   **`feature_extractor.py`**: The CNN "glasses" code.
    -   **`agents/heuristic_agent.py`**: The simple bots used in Phase 1.

---

## 📊 Visualization

Want to see graphs of the AI getting smarter?
```bash
tensorboard --logdir ./mahjong_curriculum_logs/
```
Open the link it gives you (usually `http://localhost:6006`) in your browser. You should see the "Average Reward" line going up over time!

---

## Development Status

-   [x] **Core Game Engine**: Dealing, turns, winning logic.
-   [x] **The "Brain"**: Custom CNN to recognize tile patterns.
-   [x] **The "School"**: Curriculum training (Bots -> Self-Play).
-   [ ] **Advanced Reactions**: Currently, the AI is great at deciding what to *discard*. The next step is teaching it exactly when to call *Pong*, *Kong*, or *Chow* (currently it uses simple rules for this).
