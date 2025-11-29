import gymnasium as gym
import numpy as np
from mahjong.pettingzoo_env import MahjongPettingZooEnv
from mahjong.agents.heuristic_agent import HeuristicAgent

class MahjongVsBotEnv(gym.Env):
    """
    A Single-Agent Gymnasium Environment where the agent plays against 3 Heuristic Bots.
    Target Agent is always 'player_0'.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super().__init__()
        self.env = MahjongPettingZooEnv()
        self.learner_agent = "player_0"
        self.bots = {
            "player_1": HeuristicAgent(),
            "player_2": HeuristicAgent(),
            "player_3": HeuristicAgent()
        }
        
        # Expose spaces from the internal env
        self.observation_space = self.env.observation_space(self.learner_agent)
        self.action_space = self.env.action_space(self.learner_agent)
        
        self.render_mode = None

    def reset(self, seed=None, options=None):
        # Reset internal env
        observations, infos = self.env.reset(seed=seed, options=options)
        
        # Fast-forward until it is the learner's turn
        return self._process_turn(observations, infos)

    def step(self, action):
        # Step the environment with the learner's action
        actions = {self.learner_agent: action}
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Check if immediate termination (rare in Mahjong step, usually after full game)
        if terminations[self.learner_agent] or truncations[self.learner_agent]:
            return (
                observations[self.learner_agent], 
                rewards[self.learner_agent], 
                True, 
                False, 
                infos[self.learner_agent]
            )

        # If game continues, we need to play out the bots' turns until it's learner's turn again
        obs, info = self._process_turn(observations, infos)
        
        # We need to accumulate rewards? 
        # The internal env accumulates rewards in self.env.rewards but only returns them on step.
        # Since we might call env.step() multiple times, we need to sum the rewards for the learner.
        # However, _process_turn returns (obs, info). We need to change it to return (obs, reward, done, truncated, info).
        
        # Re-thinking: _process_turn should handle the loop and accumulation.
        return self._process_turns_loop(observations, rewards, terminations, truncations, infos)

    def _process_turns_loop(self, observations, step_rewards, terminations, truncations, infos):
        """
        Loop that lets bots play until it is the learner's turn or the episode ends.
        Accumulates rewards for the learner.
        """
        total_reward = step_rewards[self.learner_agent]
        
        while True:
            # Check termination
            if terminations[self.learner_agent] or truncations[self.learner_agent]:
                return (
                    observations[self.learner_agent],
                    total_reward,
                    True,
                    truncations[self.learner_agent],
                    infos[self.learner_agent]
                )
            
            # Identify current player
            current_idx = self.env.game.current_player_idx
            current_agent = self.env.idx_to_agent_name[current_idx]
            
            # If it's the learner's turn, we return control to the agent
            if current_agent == self.learner_agent:
                return (
                    observations[self.learner_agent],
                    total_reward,
                    False,
                    False,
                    infos[self.learner_agent]
                )
            
            # Otherwise, it's a Bot's turn
            bot = self.bots[current_agent]
            mask = infos[current_agent]["action_mask"]
            
            # Bot selects action
            bot_action = bot.select_action(self.env.game, current_idx, mask)
            
            # Step environment
            actions = {current_agent: bot_action}
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Accumulate learner's reward (e.g. from winning/losing hands during bot turns)
            total_reward += rewards[self.learner_agent]

    def _process_turn(self, observations, infos):
        """
        Helper for Reset to just find the first state.
        """
        # We treat Reset as a special case where reward is 0.
        # We reuse _process_turns_loop logic but initialize vars.
        
        # Dummy initial values
        rewards = {a: 0 for a in self.env.possible_agents}
        terminations = {a: False for a in self.env.possible_agents}
        truncations = {a: False for a in self.env.possible_agents}
        
        obs, reward, done, truncated, info = self._process_turns_loop(
            observations, rewards, terminations, truncations, infos
        )
        
        return obs, info
        
    def action_masks(self):
        """
        Required by MaskablePPO.
        """
        # Get the mask for the learner
        # We need to peek at the environment state
        # The mask is stored in infos by the env step/reset, but we might not have easy access to the *current* info 
        # if we are outside step().
        # Fortunately, our Env wrapper logic ensures we only return when it IS the learner's turn.
        # So we can ask the env directly.
        masks = self.env.action_mask()
        return masks[self.learner_agent]

