import functools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from .game import Game
from .phase import Phase
from .action_space import ActionSpace
from .observation import ObservationBuilder
from .game_history import GameOutcome
from .tile import Tile

class MahjongPettingZooEnv(AECEnv):
    """
    A PettingZoo AEC Environment for 4-player Mahjong.
    
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
        
        self._agent_selector = agent_selector.agent_selector(self.agents)
        self.agent_selection = None
        
        # Track if we are in the first game or subsequent games of the rotation
        self.games_played_in_rotation = 0
        
        # Store cumulative rewards if needed, but for this specific request, 
        # rewards are 0 until the very end.
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        
        # Mapping for "player_x" -> int index
        self.agent_name_to_idx = {a: i for i, a in enumerate(self.agents)}
        self.idx_to_agent_name = {i: a for i, a in enumerate(self.agents)}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        # Reset the underlying game completely
        # This clears history, effectively starting a new Rotation.
        self.game = Game() 
        self.game.initialize_game()
        
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Determine starting agent
        # Game starts with dealer (index 0 initially, but game.current_player_idx tracks it)
        self._agent_selector = agent_selector.agent_selector(self.agents)
        
        # Fast-forward to the first decision point
        self._advance_to_next_decision()
        
        # Set the agent_selection to the player who needs to act
        # self.game.current_player_idx should point to the active player
        self.agent_selection = self.idx_to_agent_name[self.game.current_player_idx]
        
        # We need to sync the selector to this agent so next() works correctly
        # agent_selector usually cycles 0-1-2-3. But Mahjong turn order can jump (winner becomes dealer).
        # So we might not use _agent_selector strictly for turn order if game dictates it.
        # We will manually set agent_selection based on game state.
        
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_idx = self.agent_name_to_idx[agent]
        
        # 1. Decode and Apply Action
        # Only if it's actually this player's turn in the Game (validation)
        if self.game.current_player_idx == agent_idx and self.game.phase == Phase.DISCARD:
             # Decode Action
             tile_to_discard = self._decode_action(action, agent_idx)
             
             # Apply to Game
             self.game.step(external_action=(action, tile_to_discard))
        else:
            # If agent acted out of turn or in wrong phase, treat as no-op or penalty?
            # For AEC, we assume the caller respects agent_selection.
            # If we are here, it IS the agent's turn.
            # But if Game thinks it's Reaction phase, and we don't support Agent Reaction yet,
            # we shouldn't be here. _advance_to_next_decision ensures we only stop at DISCARD.
            pass
            
        # 2. Advance Game
        self._advance_to_next_decision()
        
        # 3. Check for Rotation End
        # _advance_to_next_decision handles moving to next hand if one finishes.
        # If it sets Phase.GAME_OVER, it means the WHOLE ROTATION is done.
        
        if self.game.phase == Phase.GAME_OVER:
            self._assign_final_rewards()
            for a in self.agents:
                self.terminations[a] = True
        
        # 4. Update agent_selection
        # If not over, set selection to current player
        if not all(self.terminations.values()):
            self.agent_selection = self.idx_to_agent_name[self.game.current_player_idx]
            
        self._accumulate_rewards()

    def _advance_to_next_decision(self):
        """
        Runs the game loop until:
        1. A player needs to DISCARD (Phase.DISCARD) -> Returns, sets agent_selection.
        2. The Full Rotation Ends -> Returns, sets terminations.
        """
        while True:
            if self.game.phase == Phase.GAME_OVER:
                # Check if Rotation is complete
                # Logic: East -> South -> West -> North -> Done
                # self.game.table_wind updates automatically in game.py logic.
                # But game.py resets table_wind to EAST in initialize_game IF history is empty.
                # We are preserving history.
                
                # Check termination condition:
                # If we just finished a hand, check table wind.
                # We need to peek at the NEXT game's wind.
                # self.game.determine_table_wind() calculates based on history.
                
                # Actually, game.finalize_game() appends to history and rotates dealer.
                # We need to simulate "starting" the next game to see what the wind IS.
                
                # Hack: We check the history.
                # If the last game was NORTH wind and dealer rotated back to 0?
                # The logic in main.py was:
                # if game.table_wind == EAST and last_game.table_wind == NORTH: break
                
                last_game_entry = self.game.history[-1] if self.game.history else None
                
                # We need to determine the wind for the *next* game without fully starting it?
                # Or we just start it.
                self.game.initialize_game() # This calculates new table_wind
                
                if self.game.table_wind == self.game.table_wind.EAST and \
                   last_game_entry and last_game_entry.table_wind == self.game.table_wind.NORTH:
                    # Rotation Complete!
                    # Revert phase to GAME_OVER to signal termination
                    self.game.phase = Phase.GAME_OVER
                    return
                
                # Otherwise, continue loop (new hand started)
                # initialize_game sets phase to DRAW
                continue

            # Stop if we are in DISCARD phase (Decision point)
            if self.game.phase == Phase.DISCARD:
                return
                
            # Advance Game Step (Bots / System logic)
            self.game.step()

    def _assign_final_rewards(self):
        """
        Assign rewards based on final scores.
        Rank 1: +100
        Rank 2: +50
        Rank 3: -50
        Rank 4: -100
        """
        scores = [(i, p.score) for i, p in enumerate(self.game.players)]
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Map rank to reward
        rank_rewards = [100, 50, -50, -100]
        
        for rank, (agent_idx, score) in enumerate(scores):
            agent_name = self.idx_to_agent_name[agent_idx]
            self.rewards[agent_name] = rank_rewards[rank]

    def _decode_action(self, action_idx: int, agent_idx: int) -> Tile | None:
        # Same logic as MahjongEnv
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

