import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pettingzoo import ParallelEnv

from .game import Game
from .phase import Phase
from .action_space import ActionSpace
from .observation import ObservationBuilder
from .game_history import GameOutcome
from .tile import Tile

class MahjongPettingZooEnv(ParallelEnv):
    """
    A PettingZoo Parallel Environment for 4-player Mahjong.
    
    The episode continues for a full rotation of the table wind (East -> North).
    Rewards are sparse: distributed only at the end of the full rotation based on final scores.
    """
    
    metadata = {"render_modes": ["human"], "name": "mahjong_v1"}

    def __init__(self):
        super().__init__()
        
        self.game = Game()
        self.observation_builder = ObservationBuilder()
        
        self.possible_agents = ["player_0", "player_1", "player_2", "player_3"]
        self.agents = self.possible_agents[:]
        
        self.action_spaces = {
            agent: spaces.Discrete(ActionSpace.SIZE) for agent in self.possible_agents
        }
        
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, 
                high=1, 
                shape=(self.observation_builder.num_channels, self.observation_builder.num_tiles), 
                dtype=np.float32
            ) for agent in self.possible_agents
        }
        
        # Mapping for "player_x" -> int index
        self.agent_name_to_idx = {a: i for i, a in enumerate(self.possible_agents)}
        self.idx_to_agent_name = {i: a for i, a in enumerate(self.possible_agents)}
        
        self.rewards = {a: 0 for a in self.agents}
        self.render_mode = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.game = Game() 
        self.game.initialize_game()
        self.rewards = {a: 0 for a in self.agents}
        
        # Fast-forward to the first decision point
        self._advance_to_next_decision()
        
        observations = {a: self.observe(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions):
        """
        actions: dict of {agent: action}
        """
        # 1. Identify the active agent who NEEDS to act
        # We assume self.game.current_player_idx points to them
        # AND we are in a decision phase (DISCARD)
        # If we are not in a decision phase, we shouldn't be here (handled by _advance)
        
        current_idx = self.game.current_player_idx
        current_agent = self.idx_to_agent_name[current_idx]
        
        # Validation: Is it actually a decision step?
        if self.game.phase == Phase.DISCARD:
            # Extract the action for the ACTIVE agent
            # We ignore actions from other agents
            action = actions[current_agent]
            tile_to_discard = self._decode_action(action, current_idx)
            
            # Apply to Game
            self.game.step(external_action=(action, tile_to_discard))
        
        # 2. Advance Game until next decision or end
        self._advance_to_next_decision()
        
        # 3. Check Termination
        terminated = (self.game.phase == Phase.GAME_OVER)
        truncated = False
        
        if terminated:
            self._assign_final_rewards()
            observations = {a: self.observe(a) for a in self.agents}
            rewards = self.rewards
            terminations = {a: True for a in self.agents}
            truncations = {a: False for a in self.agents}
            infos = {a: {} for a in self.agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos
            
        # 4. Return step info
        observations = {a: self.observe(a) for a in self.agents}
        # Rewards are 0 until end (sparse)
        rewards = {a: 0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        # Add Action Mask to info?
        # Usually useful for RL.
        # Only the current agent has a valid mask. Others are all False?
        # Or just always return mask based on hand.
        
        return observations, rewards, terminations, truncations, infos

    def _advance_to_next_decision(self):
        """
        Runs the game loop until:
        1. A player needs to DISCARD (Phase.DISCARD) -> Returns.
        2. The Full Rotation Ends -> Returns (Phase.GAME_OVER).
        """
        while True:
            if self.game.phase == Phase.GAME_OVER:
                # Check for Rotation End logic (Same as before)
                last_game_entry = self.game.history[-1] if self.game.history else None
                self.game.initialize_game()
                
                if self.game.table_wind == self.game.table_wind.EAST and \
                   last_game_entry and last_game_entry.table_wind == self.game.table_wind.NORTH:
                    self.game.phase = Phase.GAME_OVER
                    return
                continue

            if self.game.phase == Phase.DISCARD:
                return
                
            self.game.step()

    def _assign_final_rewards(self):
        scores = [(i, p.score) for i, p in enumerate(self.game.players)]
        scores.sort(key=lambda x: x[1], reverse=True)
        rank_rewards = [100, 50, -50, -100]
        for rank, (agent_idx, score) in enumerate(scores):
            agent_name = self.idx_to_agent_name[agent_idx]
            self.rewards[agent_name] = rank_rewards[rank]

    def _decode_action(self, action_idx: int, agent_idx: int) -> Tile | None:
        if action_idx < 34:
            agent = self.game.players[agent_idx]
            for tile in agent.hand:
                if tile.get_index_34() == action_idx:
                    return tile
        return None

    def observe(self, agent):
        agent_idx = self.agent_name_to_idx[agent]
        return self.observation_builder.build_observation(self.game, agent_idx)

    def render(self):
        self.game.display_player_hands()
        
    def close(self):
        pass
