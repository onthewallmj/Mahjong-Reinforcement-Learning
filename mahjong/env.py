import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .game import Game
from .phase import Phase
from .action_space import ActionSpace, ActionType
from .observation import ObservationBuilder
from .game_history import GameOutcome
from .player import Player
from .tile import Tile

class MahjongEnv(gym.Env):
    """
    A Gymnasium environment for Mahjong (Hong Kong style).
    
    This environment wraps the Mahjong game logic for a single RL agent 
    playing as Player 0 against 3 bot opponents (random/heuristic).
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self):
        super().__init__()
        
        self.game = Game()
        self.observation_builder = ObservationBuilder()
        
        # Define Action Space (Discrete 42)
        self.action_space = spaces.Discrete(ActionSpace.SIZE)
        
        # Define Observation Space (Box 33x34)
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(self.observation_builder.num_channels, self.observation_builder.num_tiles), 
            dtype=np.float32
        )
        
        self.agent_index = 0 # The RL agent is always Player 0 in this wrapper
        self.previous_score = 0
        
    def reset(self, seed=None, options=None):
        """
        Resets the game to a new state.
        """
        super().reset(seed=seed)
        
        self.game.initialize_game()
        self.previous_score = 0
        
        # Advance the game until the Agent (Player 0) needs to act
        self._advance_until_agent_turn()
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info

    def step(self, action):
        """
        Executes the given action for the agent.
        Then simulates the game until it is the agent's turn again.
        """
        # Convert RL action to Game action
        game_action = self._decode_action(action)
        
        # Apply Agent's Action
        # We pass the external action to the game.step() call
        # Note: game.step() advances ONE phase.
        # If we are in a phase requiring input (DISCARD or REACTION), this step should CONSUME the action.
        
        # Validation: Is it actually the agent's turn?
        if self.game.phase == Phase.GAME_OVER:
            return self._get_obs(), 0, True, False, self._get_info()
            
        if self.game.current_player_idx == self.agent_index:
            # Apply the action
            self.game.step(external_action=game_action)
        else:
            # Agent tried to act out of turn? Or maybe this is a reaction?
            # In reaction phase, ANY player can act.
            pass
            
        # Check for game end immediately after agent's move
        terminated = self.game.phase == Phase.GAME_OVER
        if terminated:
             return self._get_obs(), self._calculate_reward(), True, False, self._get_info()
        
        # Run game loop for other players until Agent needs to act again
        self._advance_until_agent_turn()
        
        terminated = self.game.phase == Phase.GAME_OVER
        truncated = False
        reward = self._calculate_reward() # Calculate reward (delta)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _advance_until_agent_turn(self):
        """
        Loops game.step() until:
        1. It is the Agent's turn to DISCARD.
        2. It is the Agent's turn to REACT (not implemented deeply yet, usually any player can react).
        3. The game ends.
        """
        while self.game.phase != Phase.GAME_OVER:
            # Check if we need to stop for Agent Input
            
            # Case A: Agent needs to DISCARD
            if self.game.phase == Phase.DISCARD and self.game.current_player_idx == self.agent_index:
                break
                
            # Case B: Agent can REACT (TODO: Implement pausing for reactions)
            # Currently, reaction logic iterates all players. To support RL, Game.step would need
            # to pause specifically for the agent's reaction opportunity.
            # For v1, we might skip agent reactions or implement them simply.
            
            # Step the game (bot turns)
            self.game.step()

    def _decode_action(self, action_idx: int) -> tuple[int, Tile | None]:
        """
        Converts the discrete action index (0-41) into a tuple understood by Game.step.
        Returns (ActionType_Int, Target_Tile or None).
        
        The Game logic mostly needs the Tile to discard.
        """
        # Check Discard (0-33)
        if action_idx < 34:
            # We need to map index 0-33 back to a Tile object.
            # But which tile? The index represents a Tile TYPE (e.g. 1 Bamboo).
            # We need to find a matching tile in the player's hand.
            tile_type_idx = action_idx
            return (action_idx, self._find_tile_in_hand(tile_type_idx))
            
        # TODO: Decode other actions (Skip, Chow, Pong, Kong, Win)
        return (action_idx, None)

    def _find_tile_in_hand(self, tile_type_idx: int) -> Tile | None:
        """
        Finds a tile in the agent's hand that matches the tile_type_idx (0-33).
        """
        agent = self.game.players[self.agent_index]
        for tile in agent.hand:
            if tile.get_index_34() == tile_type_idx:
                return tile
        return None

    def _calculate_reward(self) -> float:
        """
        Calculates the reward for the agent as the change in score.
        """
        agent = self.game.players[self.agent_index]
        current_score = agent.score
        reward = float(current_score - self.previous_score)
        self.previous_score = current_score
        return reward

    def render(self):
        self.game.display_player_hands()

    def _get_obs(self):
        return self.observation_builder.build_observation(self.game, self.agent_index)

    def _get_info(self):
        return {
            "valid_actions": self._get_action_mask()
        }

    def _get_action_mask(self) -> np.ndarray:
        """
        Returns a boolean mask of valid actions for the agent.
        Shape: (42,)
        """
        mask = np.zeros(ActionSpace.SIZE, dtype=bool)
        agent = self.game.players[self.agent_index]
        
        # 1. Discard Actions (0-33)
        # Valid if the agent is in DISCARD phase and has the tile in hand
        if self.game.phase == Phase.DISCARD and self.game.current_player_idx == self.agent_index:
            hand_indices = {tile.get_index_34() for tile in agent.hand}
            for idx in hand_indices:
                if 0 <= idx < 34:
                    mask[idx] = True
                    
        # 2. Skip (34), Chow (35-37), Pong (38), Kong (39)
        # These are Reaction actions. Currently, our Env only pauses for DISCARD.
        # So these are always False for now.
        
        # 3. Self-Kong (40)
        # Currently auto-handled by game logic before agent gets control.
        # So False.
        
        # 4. Win (41)
        # Self-draw win is auto-handled. Discard win is a reaction.
        # So False.
        
        return mask
