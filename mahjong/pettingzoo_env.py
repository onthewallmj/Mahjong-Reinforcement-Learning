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
        
        self.game = Game(debug=False)
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
        self.game = Game(debug=False) 
        self.game.initialize_game()
        self.rewards = {a: 0 for a in self.agents}
        
        # Fast-forward to the first decision point
        self._advance_to_next_decision()
        
        observations = {a: self.observe(a) for a in self.agents}
        
        # Populate infos with action masks immediately
        infos = {a: {} for a in self.agents}
        action_masks = self.action_mask()
        for agent in self.agents:
             infos[agent]["action_mask"] = action_masks[agent]
             
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
        # Return accumulated rewards (from intermediate games) and reset
        step_rewards = self.rewards.copy()
        self.rewards = {a: 0 for a in self.agents}
        
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}
        
        # Add Action Mask to info
        action_masks = self.action_mask()
        for agent in self.agents:
             infos[agent]["action_mask"] = action_masks[agent]
        
        return observations, step_rewards, terminations, truncations, infos

    def _assign_intermediate_rewards(self, winner_idx: int | None, loser_idx: int | None):
        """
        Assigns intermediate rewards for winning/losing a hand.
        This provides dense feedback to the agent.
        """
        # Default penalty/reward
        win_reward = 10.0
        deal_in_penalty = -10.0
        
        if winner_idx is not None:
            winner_agent = self.idx_to_agent_name[winner_idx]
            self.rewards[winner_agent] += win_reward
            
            if loser_idx is not None:
                # Ron (Win on discard)
                loser_agent = self.idx_to_agent_name[loser_idx]
                self.rewards[loser_agent] += deal_in_penalty
            else:
                # Tsumo (Self-draw) - All others pay?
                # For simplicity, maybe just reward winner?
                # Or small penalty for everyone else?
                pass

    def _advance_to_next_decision(self):
        """
        Runs the game loop until:
        1. A player needs to DISCARD (Phase.DISCARD) -> Returns.
        2. The Full Rotation Ends -> Returns (Phase.GAME_OVER).
        """
        while True:
            if self.game.phase == Phase.GAME_OVER:
                # IMPORTANT: We must finalize the game to update history and dealer rotation
                # before checking for episode end or re-initializing.
                
                # Assign intermediate rewards for the hand just finished
                self._assign_intermediate_rewards(self.game.winner_seat_index, self.game.loser_seat_index)
                
                self.game.finalize_game(self.game.winner_seat_index)

                # Check for Rotation End logic (Same as before)
                last_game_entry = self.game.history[-1] if self.game.history else None
                
                # DEBUG: Print state before re-init
                # print(f"Game Over. History Len: {len(self.game.history)}. Last Wind: {last_game_entry.table_wind if last_game_entry else 'None'}")

                self.game.initialize_game()
                
                # DEBUG: Print state after re-init
                # print(f"New Wind: {self.game.table_wind}")

                # Use .value for safer comparison of Enums
                if self.game.table_wind.value == self.game.table_wind.EAST.value and \
                   last_game_entry and last_game_entry.table_wind.value == self.game.table_wind.NORTH.value:
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

    def action_mask(self):
        """
        Returns a dictionary of boolean masks for each agent.
        Only valid discard actions are True.
        """
        masks = {}
        for agent in self.agents:
            agent_idx = self.agent_name_to_idx[agent]
            # Get player object
            player = self.game.players[agent_idx]
            
            # Mask is size 34 (ActionSpace.SIZE)
            mask = np.zeros(ActionSpace.SIZE, dtype=bool)
            
            # If player is active and needs to discard, mark tiles in hand as True
            # But we generate mask for ALL agents always? Or just active?
            # SB3 usually needs mask for the agent being queried.
            # In vectorized env, we return masks for all?
            
            # For now, just mark tiles present in hand as valid.
            # Even if it's not their turn, this is the "valid action space" given their state.
            # If it IS their turn, they must pick one of these.
            for tile in player.hand:
                idx = tile.get_index_34()
                if idx < 34:
                    mask[idx] = True
            
            masks[agent] = mask
            
        return masks

    def observe(self, agent):
        agent_idx = self.agent_name_to_idx[agent]
        return self.observation_builder.build_observation(self.game, agent_idx)

    def render(self):
        self.game.display_player_hands()
        
    def close(self):
        pass
