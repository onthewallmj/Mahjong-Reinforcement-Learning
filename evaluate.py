import os

import gymnasium as gym
from mahjong.pettingzoo_env import MahjongPettingZooEnv
import numpy as np
from stable_baselines3 import PPO
import supersuit as ss


def evaluate_agent(model_path=None, num_episodes=10):
    """
    Evaluates a trained agent by running it in self-play (controlling all 4 agents).
    """
    print(f"Starting evaluation over {num_episodes} episodes...")

    # Use the PettingZoo environment wrapped exactly like in training
    env = MahjongPettingZooEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # Load model if provided
    model = None
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print("No model provided or found. Using Random Agent.")

    # Metrics
    total_rewards = [] # Per agent
    steps_per_episode = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        # To track rewards per agent, we need to unpack the vectorized reward
        # The vec env returns rewards as an array [r0, r1, r2, r3]
        episode_rewards = np.zeros(4)

        while not done:
            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Random action: sample from the vector env's action space
                action = [env.action_space.sample() for _ in range(4)] # Incorrect, vec_env.action_space handles batch
                # Actually, env.action_space is likely MultiDiscrete or similar? 
                # No, concat_vec_envs makes it look like a single env with batch dim.
                # env.action_space.sample() returns a list/array of actions.
                action = [env.action_space.sample() for _ in range(env.num_envs)]
                # Flatten if necessary, but SB3 env usually returns array
                # Actually, DummyVecEnv actions are usually np arrays.
                action = np.array([env.action_space.sample() for _ in range(env.num_envs)])
                # The action space of the *Vectorized* env is just the space for ONE agent? No.
                # pettingzoo_env_to_vec_env makes num_envs = 4.
                # So env.action_space is the space for 1 agent.
                # To sample for all 4, we need [space.sample() for _ in range(4)]
                action = [env.action_space.sample() for _ in range(env.num_envs)]

            obs, rewards, dones, infos = env.step(action)
            episode_rewards += rewards
            episode_steps += 1
            
            # In a vectorized env, 'dones' is an array.
            # Mahjong ends for everyone at the same time.
            done = any(dones)

        # Log results for this episode
        # Just take the reward of Player 0 as representative, or average?
        # Since it's self play, average reward should be 0 (zero sum).
        # We can log Player 0's score.
        total_rewards.append(episode_rewards[0])
        steps_per_episode.append(episode_steps)
        
        # We can't easily check "Winner" without peeking into the internal env
        # But the reward tells us the rank.
        
        print(f"Episode {episode + 1}: P0 Reward={episode_rewards[0]}, Steps={episode_steps}")

    # Summary
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_per_episode)

    print("\n" + "="*30)
    print("EVALUATION RESULTS (Self-Play)")
    print("="*30)
    print(f"Episodes: {num_episodes}")
    print(f"Avg Reward (Player 0): {avg_reward:.2f}")
    print(f"Avg Steps:  {avg_steps:.1f}")
    print("="*30)


if __name__ == "__main__":
    # Check if a trained model exists
    model_file = "mahjong_ppo_model"
    evaluate_agent(model_path=model_file, num_episodes=5)
