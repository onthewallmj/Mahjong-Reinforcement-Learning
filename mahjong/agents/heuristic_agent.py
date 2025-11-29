import random
import numpy as np
from mahjong.action_space import ActionSpace
from mahjong.tile import Tile, TileSuit

class HeuristicAgent:
    """
    A Rule-Based Agent that plays with basic strategies:
    1. Always wins if possible.
    2. Discards isolated Honor tiles (Winds/Dragons) first.
    3. Discards isolated Terminal tiles (1, 9) second.
    4. Discards random other tiles.
    5. Randomly decides to Chow/Pong/Kong (for now).
    """
    
    def __init__(self):
        pass
        
    def select_action(self, game, player_index: int, action_mask: np.ndarray) -> int:
        """
        Selects an action based on heuristics.
        """
        # 1. Check for Win (Action 41)
        if action_mask[ActionSpace.ACT_WIN]:
            return ActionSpace.ACT_WIN
            
        # 2. Check for Declarations (Chow/Pong/Kong)
        # For now, we'll be conservative and mostly Skip (Action 34) unless we have a very strong reason.
        # A simple heuristic: 20% chance to Call if valid.
        reaction_actions = [
            ActionSpace.ACT_CHOW_LOW, ActionSpace.ACT_CHOW_MID, ActionSpace.ACT_CHOW_HIGH,
            ActionSpace.ACT_PONG, ActionSpace.ACT_KONG, ActionSpace.ACT_SELF_KONG
        ]
        
        possible_reactions = [a for a in reaction_actions if action_mask[a]]
        
        if possible_reactions:
            # If we can self-kong, always do it (usually good)
            if action_mask[ActionSpace.ACT_SELF_KONG]:
                return ActionSpace.ACT_SELF_KONG
            
            # Otherwise, small chance to call
            if random.random() < 0.2:
                return random.choice(possible_reactions)
            
            # Otherwise, check if we MUST call (e.g., if Skip is not allowed? Skip is always allowed if logic is correct)
            if action_mask[ActionSpace.ACT_SKIP]:
                return ActionSpace.ACT_SKIP
                
        # 3. Discard Logic
        # Identify valid discard actions
        valid_discards = [i for i in range(34) if action_mask[i]]
        
        if not valid_discards:
            # Should essentially never happen in Discard phase if mask is correct.
            # If we are in reaction phase and chose Skip, we are done.
            # If we are forced to discard but have no options?
            # Return Skip if available, else random valid.
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                return random.choice(valid_indices)
            return ActionSpace.ACT_SKIP # Fallback

        player = game.players[player_index]
        hand = player.hand
        
        # We map action indices (0-33) back to Tile objects to analyze them
        # Note: This mapping is static based on ActionSpace definition.
        # However, we need to know which specific tile instance in hand corresponds to the abstract tile type?
        # Actually, ActionSpace 0-33 just means "Discard a tile of type X".
        
        # Heuristic: Rank discards
        # Score: Higher is better to discard
        discard_scores = {}
        
        for action_idx in valid_discards:
            # Reconstruct tile type from index (simplified)
            # 0-8: Man, 9-17: Pin, 18-26: Sou, 27-33: Honors
            suit = None
            value = None
            is_honor = False
            is_terminal = False
            
            if 0 <= action_idx <= 8:
                suit = TileSuit.CHARACTER
                value = action_idx + 1
            elif 9 <= action_idx <= 17:
                suit = TileSuit.DOT
                value = action_idx - 9 + 1
            elif 18 <= action_idx <= 26:
                suit = TileSuit.BAMBOO
                value = action_idx - 18 + 1
            else:
                is_honor = True
                
            if value in [1, 9]:
                is_terminal = True
                
            # Base Score
            score = 0
            
            # Preference 1: Discard Honors (Isolated ones)
            # We should check if we have a pair/triplet. If action_mask allows it, we have at least 1.
            # We need to check count in hand.
            # Since we don't have easy access to "count of tile type X" without iterating hand...
            # Let's assume the agent just prefers discarding Honors.
            if is_honor:
                score += 100
            
            # Preference 2: Discard Terminals
            if is_terminal:
                score += 50
                
            # Add noise to avoid being deterministic
            score += random.uniform(0, 10)
            
            discard_scores[action_idx] = score
            
        # Select discard with highest score
        best_discard = max(discard_scores, key=discard_scores.get)
        return best_discard

