import argparse
import os

from mahjong.pettingzoo_env import MahjongPettingZooEnv
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
import supersuit as ss
from train import SupersuitSB3Wrapper


def evaluate_vs_random(model, num_episodes=10):
    """
    Evaluates the Trained Agent (Player 0) against 3 Random Bots.
    Uses the raw PettingZoo environment to allow different policies per agent.

    This mode is crucial for verifying true skill improvement, as self-play
    can sometimes lead to degenerate strategies that only work against copies of itself.
    """
    print(f"\nStarting VS RANDOM evaluation over {num_episodes} episodes...")

    # Use raw environment, no vectorization
    env = MahjongPettingZooEnv()

    # Metrics
    p0_wins = 0
    p0_total_reward = 0

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}...")
        observations, infos = env.reset()
        # env.reset() returns (obs_dict, info_dict) in PettingZoo standard

        terminations = {a: False for a in env.possible_agents}
        truncations = {a: False for a in env.possible_agents}

        episode_steps = 0
        spinner = ['|', '/', '-', '\\']

        while not all(terminations.values()) and not all(truncations.values()):
            actions = {}

            # Decide actions for all agents who need to act
            # Only proceed if we have observations (PettingZoo ParallelEnv provides them)

            for agent in env.agents:
                obs = observations[agent]
                # Check if agent is Player 0 (Trained) or others (Random)
                if agent == "player_0":
                    # Trained Agent
                    # MaskablePPO predict requires action mask
                    mask = infos[agent]["action_mask"]

                    # Predict with mask. predict() handles single observation automatically?
                    # SB3 models usually expect batch dim if env wasn't vectorized during load?
                    # But predict() usually handles unbatched input by adding dim.
                    action, _ = model.predict(
                        obs, action_masks=mask, deterministic=True)

                    if isinstance(action, np.ndarray):
                        action = action.item()
                    actions[agent] = action
                else:
                    # Random Agent
                    mask = infos[agent]["action_mask"]
                    valid_actions = np.flatnonzero(mask)
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        action = 0
                    actions[agent] = action

            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(
                actions)
            episode_steps += 1

            if episode_steps % 100 == 0:
                print(
                    f"\r{spinner[(episode_steps // 100) % 4]} Steps: {episode_steps}", end="", flush=True)

            if episode_steps > 10000:
                print("Max steps.")
                break

        # Episode finished
        p0_reward = env.rewards["player_0"]
        p0_total_reward += p0_reward
        if p0_reward > 0:
            p0_wins += 1

        print(f"\nEpisode {episode + 1} Result: P0 Reward = {p0_reward}")

    win_rate = (p0_wins / num_episodes) * 100
    avg_reward = p0_total_reward / num_episodes

    print("\n" + "="*30)
    print("EVALUATION RESULTS (Vs Random)")
    print("="*30)
    print(f"Episodes: {num_episodes}")
    print(f"Win Rate (Positive Score): {win_rate:.1f}%")
    print(f"Avg Reward (Player 0): {avg_reward:.2f}")
    print("="*30)


def evaluate_agent(model_path=None, num_episodes=10, vs_random=False):
    """
    Entry point for evaluation.

    Modes:
    1. Self-Play (Default): The trained agent plays all 4 seats. Useful for stability checks.
    2. Vs Random (--vs-random): Trained Agent (Player 0) vs 3 Random Bots.
    """
    print(f"Starting evaluation over {num_episodes} episodes...")

    # Use the PettingZoo environment wrapped exactly like in training
    env = MahjongPettingZooEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # Load model if provided
    # We use MaskablePPO if possible, as it respects action masks for legal play.
    model = None
    if model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading trained model from {model_path}...")
        try:
            model = MaskablePPO.load(model_path)
        except:
            print("Could not load as MaskablePPO, trying PPO...")
            model = PPO.load(model_path)
    else:
        print("No model provided or found. Using Random Agent.")

    if vs_random and model:
        evaluate_vs_random(model, num_episodes)
        return

    # Metrics
    total_rewards = []  # Per agent
    steps_per_episode = []

    for episode in range(num_episodes):
        print(f"Starting Episode {episode + 1}...")
        obs = env.reset()
        done = False
        episode_steps = 0
        # To track rewards per agent, we need to unpack the vectorized reward
        # The vec env returns rewards as an array [r0, r1, r2, r3]
        episode_rewards = np.zeros(4)

        # Spinner for liveness
        spinner = ['|', '/', '-', '\\']

        while not done:
            # Force stop if too many steps (infinite loop guard)
            if episode_steps > 10000:
                print(
                    f"\nMax steps reached ({episode_steps}). Force ending episode.")
                break

            # Print progress every 100 steps to show it's not stuck
            if episode_steps % 100 == 0:
                print(
                    f"\r{spinner[(episode_steps // 100) % 4]} Steps: {episode_steps}", end="", flush=True)

            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Random action: sample from the vector env's action space
                # Incorrect, vec_env.action_space handles batch
                action = [env.action_space.sample() for _ in range(4)]
                # Actually, env.action_space is likely MultiDiscrete or similar?
                # No, concat_vec_envs makes it look like a single env with batch dim.
                # env.action_space.sample() returns a list/array of actions.
                action = [env.action_space.sample()
                          for _ in range(env.num_envs)]
                # Flatten if necessary, but SB3 env usually returns array
                # Actually, DummyVecEnv actions are usually np arrays.
                action = np.array([env.action_space.sample()
                                  for _ in range(env.num_envs)])
                # The action space of the *Vectorized* env is just the space for ONE agent? No.
                # pettingzoo_env_to_vec_env makes num_envs = 4.
                # So env.action_space is the space for 1 agent.
                # To sample for all 4, we need [space.sample() for _ in range(4)]
                action = [env.action_space.sample()
                          for _ in range(env.num_envs)]

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

        print(
            f"\nEpisode {episode + 1} Finished: P0 Reward={episode_rewards[0]}, Steps={episode_steps}")

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes")
    parser.add_argument("--vs-random", action="store_true",
                        help="Play against random bots instead of self-play")
    args = parser.parse_args()

    evaluate_agent(model_path=model_file,
                   num_episodes=args.episodes, vs_random=args.vs_random)
