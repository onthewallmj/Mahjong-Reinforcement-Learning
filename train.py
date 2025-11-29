import gymnasium as gym
from mahjong.pettingzoo_env import MahjongPettingZooEnv
from mahjong.feature_extractor import MahjongFeatureExtractor
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnvWrapper
import supersuit as ss
import numpy as np


class SupersuitSB3Wrapper(VecEnvWrapper):
    """
    Wrapper to adapt Supersuit's VecEnv for Stable Baselines3's MaskablePPO.

    Fixes:
    1. Missing 'has_attr' method.
    2. Tuple observation format (Gym API) vs SB3 expected format.
    3. Missing 'action_masks' method (required by MaskablePPO).
    4. Missing 'env_method' implementation in Supersuit.
    """

    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, tuple):
            return obs[0]
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        step_result = self.venv.step_wait()
        if len(step_result) == 5:
            obs, rewards, terminations, truncations, infos = step_result
            dones = np.logical_or(terminations, truncations)
            return obs, rewards, dones, infos
        return step_result

    def has_attr(self, attr_name):
        if attr_name == "action_masks":
            return True
        try:
            return hasattr(self.venv, attr_name)
        except:
            return False

    def env_is_wrapped(self, wrapper_class, indices=None):
        try:
            return self.venv.env_is_wrapped(wrapper_class, indices)
        except TypeError:
            return self.venv.env_is_wrapped(wrapper_class)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "action_masks":
            masks = self.action_masks()
            if indices is None:
                return masks
            if isinstance(indices, int):
                return [masks[indices]]
            return [masks[i] for i in indices]
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def action_masks(self):
        if hasattr(self.venv, 'par_env'):
            masks_dict = self.venv.par_env.action_mask()
            return [masks_dict[agent] for agent in self.venv.par_env.possible_agents]
        raise NotImplementedError(
            "Could not retrieve action masks from Supersuit wrapper")


def train():
    # 1. Instantiate the PettingZoo environment
    env = MahjongPettingZooEnv()

    # 2. Wrap it for Stable Baselines3 compatibility
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 3. Adapt for MaskablePPO
    env = SupersuitSB3Wrapper(env)

    # 4. Add a Monitor wrapper to track episode statistics
    env = VecMonitor(env)

    # 4. Define the PPO Model
    # We use a Multi-Layer Perceptron (MlpPolicy) as the base, but we inject a custom
    # Feature Extractor (CNN) to handle the spatial structure of the tiles.
    
    policy_kwargs = dict(
        features_extractor_class=MahjongFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs
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
