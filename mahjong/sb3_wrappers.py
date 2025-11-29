import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


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
        except Exception:
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
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    def action_masks(self):
        if hasattr(self.venv, "par_env"):
            masks_dict = self.venv.par_env.action_mask()
            return [masks_dict[agent] for agent in self.venv.par_env.possible_agents]
        raise NotImplementedError("Could not retrieve action masks from Supersuit wrapper")


