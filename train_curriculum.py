import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import supersuit as ss

from mahjong.envs.vs_bot_env import MahjongVsBotEnv
from mahjong.pettingzoo_env import MahjongPettingZooEnv
from mahjong.feature_extractor import MahjongFeatureExtractor
from train import SupersuitSB3Wrapper

def train_curriculum():
    """
    Executes a 2-Phase Curriculum Learning Strategy.
    """
    
    # ==========================================
    # PHASE 1: Rule-Based Training (Agent vs Bots)
    # ==========================================
    print("\n=== Starting Phase 1: Training vs Heuristic Bots ===")
    print("Goal: Learn basic rules and winning patterns against decent opponents.\n")
    
    # 1. Setup Single-Agent Env
    # MahjongVsBotEnv presents a standard Gym interface
    env_phase1 = MahjongVsBotEnv()
    env_phase1 = DummyVecEnv([lambda: env_phase1]) # Vectorize
    env_phase1 = VecMonitor(env_phase1)
    
    # 2. Define Policy with Custom CNN
    policy_kwargs = dict(
        features_extractor_class=MahjongFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    
    model = MaskablePPO(
        "MlpPolicy",
        env_phase1,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./mahjong_curriculum_logs/"
        # tensorboard_log=None
    )
    
    # Train for 100,000 steps (approx 250 games if game len ~400 steps)
    # Adjust this based on training time availability
    model.learn(total_timesteps=100_000)
    model.save("mahjong_phase1_vs_bots")
    print("Phase 1 Complete. Model saved to 'mahjong_phase1_vs_bots.zip'.")
    
    # Clean up
    env_phase1.close()
    del env_phase1
    del model
    
    # ==========================================
    # PHASE 2: Self-Play Training
    # ==========================================
    print("\n=== Starting Phase 2: Self-Play Fine-Tuning ===")
    print("Goal: Develop advanced strategies by playing against itself.\n")
    
    # 1. Setup Multi-Agent Env (Self-Play)
    env_phase2 = MahjongPettingZooEnv()
    env_phase2 = ss.pettingzoo_env_to_vec_env_v1(env_phase2)
    env_phase2 = SupersuitSB3Wrapper(env_phase2)
    env_phase2 = VecMonitor(env_phase2)
    
    # 2. Load the model from Phase 1 into the new environment
    # This transfers the weights learned against bots to the self-play setting
    model_phase2 = MaskablePPO.load("mahjong_phase1_vs_bots", env=env_phase2)
    
    # Train for 400,000 more steps
    model_phase2.learn(total_timesteps=400_000, reset_num_timesteps=False)
    model_phase2.save("mahjong_ppo_model")
    print("Phase 2 Complete. Final model saved to 'mahjong_ppo_model.zip'.")
    
    env_phase2.close()

if __name__ == "__main__":
    train_curriculum()

