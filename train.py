import gymnasium as gym
from mahjong.pettingzoo_env import MahjongPettingZooEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import supersuit as ss


def train():
    # 1. Instantiate the PettingZoo environment
    # This environment supports 4-player Mahjong where agents play against each other.
    env = MahjongPettingZooEnv()

    # 2. Wrap it for Stable Baselines3 compatibility
    # ss.pettingzoo_env_to_vec_env_v1 takes the Multi-Agent environment and
    # "black-boxes" the agent switching, presenting it as a Vectorized Environment
    # with num_envs = num_agents.
    # This enables "Parameter Sharing": A single PPO policy learns to play for ALL 4 seats.
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 3. Concatenate environments (Vectorization)
    # We run 1 instance of the game, but since it has 4 agents, SB3 sees 4 environments.
    # concat_vec_envs_v1 ensures the output is a standard Gym VectorEnv.
    # num_vec_envs=1 means we run 1 independent game (table) in parallel.
    # Increasing this would run multiple tables at once for faster data collection.
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # Add a Monitor wrapper to track episode statistics (rewards, lengths)
    env = VecMonitor(env)

    # 4. Define the PPO Model
    # We use a Multi-Layer Perceptron (MlpPolicy) because our observation is a flat matrix/tensor.
    # - n_steps=2048: Number of steps to run for each environment per update.
    # - batch_size=64: Minibatch size for gradient updates.
    # - ent_coef=0.01: Entropy coefficient to encourage exploration (crucial for sparse rewards).
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01
    )

    print("Starting training...")
    # 5. Train the Agent
    # total_timesteps is the total number of environment steps across all agents.
    # With 4 agents, the actual number of game turns is total_timesteps / 4.
    model.learn(total_timesteps=500_000)

    # 6. Save the Trained Model
    model.save("mahjong_ppo_model")
    print("Model saved to mahjong_ppo_model.zip")


if __name__ == "__main__":
    train()
