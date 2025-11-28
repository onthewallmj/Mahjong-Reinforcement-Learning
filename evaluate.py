import os

import gymnasium as gym
from mahjong.env import MahjongEnv
from mahjong.pettingzoo_env import MahjongPettingZooEnv
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss


def evaluate_agent(model_path=None, num_episodes=10):
    """
    Evaluates an agent (Random or PPO) against the environment.
    """
    print(f"Starting evaluation over {num_episodes} episodes...")

    # We use the Single-Agent Gymnasium Env for evaluation against bots
    # This is simpler for measuring "Win Rate" vs "The Field"
    env = MahjongEnv()

    # Metrics
    total_rewards = []
    wins = 0
    steps_per_episode = []

    # Load model if provided
    model = None
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("No model provided or found. Using Random Agent.")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            if model:
                # Predict action using trained model
                # deterministic=True gives the best move (greedy)
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Random action
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)

        # Check if agent won (Reward > 0 implies score increase, usually win or huge points)
        # Better check: Did the game end with Agent 0 as winner?
        # We can peek at env.game.winner_seat_index (internal access)
        if env.game.winner_seat_index == 0:
            wins += 1

        print(
            f"Episode {episode + 1}: Reward={episode_reward}, Steps={steps}, Winner={env.game.winner_seat_index}")

    # Summary
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    win_rate = wins / num_episodes * 100

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Episodes: {num_episodes}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps:  {avg_steps:.1f}")
    print("="*30)


if __name__ == "__main__":
    # Check if a trained model exists
    model_file = "mahjong_ppo_model"
    evaluate_agent(model_path=model_file, num_episodes=5)
