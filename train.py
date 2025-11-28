import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import supersuit as ss

from mahjong.pettingzoo_env import MahjongPettingZooEnv

def train():
    # 1. Instantiate the PettingZoo environment
    env = MahjongPettingZooEnv()

    # 2. Wrap it to be compatible with Stable Baselines3
    # This wrapper converts the Multi-Agent Parallel environment into a 
    # Vectorized Environment (num_envs = num_agents = 4).
    # This allows SB3 to treat the 4 players as 4 independent environments 
    # (sharing the same policy).
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 3. Concatenate multiple instances if desired (e.g. run 4 games in parallel)
    # For now, we just run 1 game instance (which contains 4 agents).
    # effectively batch size = 4 agents.
    # We wrap in ss.concat_vec_envs_v1 to ensure standard SB3 VecEnv behavior if needed,
    # but pettingzoo_env_to_vec_env_v1 returns a DummyVecEnv mostly.
    
    # Note: SB3 likes its envs wrapped in VecMonitor for stats logging
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = VecMonitor(env)

    # 4. Define the Model
    # MlpPolicy will Flatten the (33, 34) observation.
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01 # Encourage exploration
    )

    print("Starting training...")
    # 5. Train
    # total_timesteps is total steps across ALL parallel envs.
    model.learn(total_timesteps=500_000)

    # 6. Save
    model.save("mahjong_ppo_model")
    print("Model saved to mahjong_ppo_model.zip")

if __name__ == "__main__":
    train()

