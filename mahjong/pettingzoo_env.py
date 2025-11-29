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
from .win import WinCondition

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
        # Handle actions based on current phase
        if self.game.phase == Phase.DISCARD:
            # Discard phase: only the active player acts
            current_idx = self.game.current_player_idx
            current_agent = self.idx_to_agent_name[current_idx]
            action = actions[current_agent]
            
            # Decode discard action
            if action < 34:
                tile_to_discard = self._decode_action(action, current_idx)
                self.game.step(external_action=(action, tile_to_discard))
            else:
                # Invalid action for discard phase (shouldn't happen with proper masking)
                # Fallback: just step without action
                self.game.step()
                
        elif self.game.phase == Phase.REACTION:
            # Reaction phase: multiple players can react
            # Process reactions in priority order: WIN > Kong > Pong > Chow
            # We manually process reactions based on agent actions, then advance game state
            most_recent_discard = self.game.discards[-1] if self.game.discards else None
            discarder_idx = self.game.current_player_idx
            
            # Check for WIN first (highest priority)
            win_claimed = False
            for offset in range(1, 4):
                seat_idx = (discarder_idx + offset) % 4
                agent_name = self.idx_to_agent_name[seat_idx]
                player = self.game.players[seat_idx]
                
                # Check if this agent selected WIN action
                if actions.get(agent_name) == ActionSpace.ACT_WIN:
                    is_last_tile = len(self.game.tiles) == 0
                    if player.can_win(
                        min_points_to_win=self.game.min_points_to_win,
                        table_wind=self.game.table_wind,
                        win_condition=WinCondition.WIN_FROM_DISCARD,
                        is_last_tile=is_last_tile,
                        new_tile=most_recent_discard
                    ):
                        player.declare_discard_win(most_recent_discard, discarder_idx)
                        from .action_log import ActionLogType
                        self.game.log_action(ActionLogType.WIN, player.seat_index, most_recent_discard)
                        self.game.winner_seat_index = seat_idx
                        self.game.loser_seat_index = discarder_idx
                        self.game.phase = Phase.GAME_OVER
                        win_claimed = True
                        break
            
            if not win_claimed:
                # Check for Kong/Pong (any player except discarder)
                reaction_claimed = False
                for seat_idx in range(4):
                    if seat_idx == discarder_idx:
                        continue
                    agent_name = self.idx_to_agent_name[seat_idx]
                    player = self.game.players[seat_idx]
                    action = actions.get(agent_name, ActionSpace.ACT_SKIP)
                    
                    if action == ActionSpace.ACT_KONG and player.can_kong(most_recent_discard):
                        player.declare_kong(most_recent_discard)
                        self.game.discards.pop()
                        from .action_log import ActionLogType
                        self.game.log_action(ActionLogType.KONG, player.seat_index, most_recent_discard)
                        reaction_claimed = True
                        self.game.current_player_idx = seat_idx
                        self.game.should_draw = True  # Kong needs replacement draw
                        self.game.phase = Phase.DRAW
                        break
                    elif action == ActionSpace.ACT_PONG and player.can_pong(most_recent_discard):
                        player.declare_pong(most_recent_discard)
                        self.game.discards.pop()
                        from .action_log import ActionLogType
                        self.game.log_action(ActionLogType.PONG, player.seat_index, most_recent_discard)
                        reaction_claimed = True
                        self.game.current_player_idx = seat_idx
                        self.game.should_draw = False
                        self.game.phase = Phase.DISCARD
                        break
                
                if not reaction_claimed:
                    # Check for Chow (only next player in turn order)
                    next_player_idx = (discarder_idx + 1) % 4
                    next_agent_name = self.idx_to_agent_name[next_player_idx]
                    next_player = self.game.players[next_player_idx]
                    action = actions.get(next_agent_name, ActionSpace.ACT_SKIP)
                    
                    if action in [ActionSpace.ACT_CHOW_LOW, ActionSpace.ACT_CHOW_MID, ActionSpace.ACT_CHOW_HIGH] and \
                       next_player.can_chow(most_recent_discard, discarder_idx):
                        # Determine which chow combination to use
                        chow_combos = next_player.get_possible_chow_combinations(most_recent_discard)
                        if chow_combos:
                            # Use the first available combo (simplified - should match chow_type)
                            tile1, tile2 = chow_combos[0]
                            next_player.chow(tile1, tile2, most_recent_discard)
                            self.game.discards.pop()
                            from .action_log import ActionLogType
                            self.game.log_action(ActionLogType.CHOW, next_player.seat_index, most_recent_discard)
                            self.game.current_player_idx = next_player_idx
                            self.game.should_draw = False
                            self.game.phase = Phase.DISCARD
                        else:
                            # No chow possible, proceed to next player
                            self.game.current_player_idx = (discarder_idx + 1) % 4
                            self.game.should_draw = True
                            self.game.phase = Phase.DRAW
                    else:
                        # No reactions, proceed to next player
                        self.game.current_player_idx = (discarder_idx + 1) % 4
                        self.game.should_draw = True
                        self.game.phase = Phase.DRAW
                
                # Check for wall exhaustion
                if self.game.are_tiles_exhausted():
                    self.game.phase = Phase.GAME_OVER
        
        # 2. Advance Game until next decision or end
        self._advance_to_next_decision()
        
        # 3. Check Termination
        terminated = (self.game.phase == Phase.GAME_OVER)
        truncated = False
        
        if terminated:
            self._assign_final_rewards()
            # Return info for ALL possible agents to ensure wrappers don't crash
            observations = {a: self.observe(a) for a in self.possible_agents}
            rewards = self.rewards
            terminations = {a: True for a in self.possible_agents}
            truncations = {a: False for a in self.possible_agents}
            infos = {a: {} for a in self.possible_agents}
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
        """
        Assign final tournament-style rewards based on table scores.

        Note: If all players have the same score (e.g. 0 for a full-draw rotation),
        we treat it as a true tie and give everyone 0 reward instead of arbitrarily
        assigning +100 to player_0, +50 to player_1, etc.
        """
        scores = [(i, p.score) for i, p in enumerate(self.game.players)]

        # If all scores are equal, treat as full tie → zero reward for everyone
        unique_scores = {s for _, s in scores}
        if len(unique_scores) == 1:
            for agent_name in self.possible_agents:
                self.rewards[agent_name] = 0.0
            return

        # Otherwise, rank by score and assign tournament rewards
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
        Marks valid actions based on current game phase:
        - DISCARD phase: Only discard actions (0-33) for the active player
        - REACTION phase: WIN, reactions (Chow/Pong/Kong), and SKIP for eligible players
        """
        masks = {}
        for agent in self.possible_agents:
            agent_idx = self.agent_name_to_idx[agent]
            player = self.game.players[agent_idx]
            
            mask = np.zeros(ActionSpace.SIZE, dtype=bool)
            
            if self.game.phase == Phase.DISCARD:
                # Only the active player can discard
                if agent_idx == self.game.current_player_idx:
                    for tile in player.hand:
                        idx = tile.get_index_34()
                        if idx < 34:
                            mask[idx] = True
                    
                    # Check for self-draw win (after drawing, before discarding)
                    # This happens during DRAW phase, but we check here for the mask
                    # Actually, self-draw wins are handled automatically in _handle_draw_phase
                    # So we don't need to expose WIN action during DISCARD phase
                    
            elif self.game.phase == Phase.REACTION:
                # Multiple players can react to the most recent discard
                most_recent_discard = self.game.discards[-1] if self.game.discards else None
                if not most_recent_discard:
                    # No discard to react to, skip is always valid
                    mask[ActionSpace.ACT_SKIP] = True
                else:
                    discarder_idx = self.game.current_player_idx
                    
                    # Check WIN (highest priority)
                    is_last_tile = len(self.game.tiles) == 0
                    if player.can_win(
                        min_points_to_win=self.game.min_points_to_win,
                        table_wind=self.game.table_wind,
                        win_condition=WinCondition.WIN_FROM_DISCARD,
                        is_last_tile=is_last_tile,
                        new_tile=most_recent_discard
                    ):
                        mask[ActionSpace.ACT_WIN] = True
                    
                    # Check Kong (only if not the discarder)
                    if agent_idx != discarder_idx and player.can_kong(most_recent_discard):
                        mask[ActionSpace.ACT_KONG] = True
                    
                    # Check Pong (only if not the discarder)
                    if agent_idx != discarder_idx and player.can_pong(most_recent_discard):
                        mask[ActionSpace.ACT_PONG] = True
                    
                    # Check Chow (only for next player in turn order)
                    if player.can_chow(most_recent_discard, discarder_idx):
                        chow_combos = player.get_possible_chow_combinations(most_recent_discard)
                        # Mark which chow types are possible
                        for combo in chow_combos:
                            # Determine which chow action this corresponds to
                            # This is simplified - we'd need to check the actual combo structure
                            # For now, mark all chow actions if any chow is possible
                            mask[ActionSpace.ACT_CHOW_LOW] = True
                            mask[ActionSpace.ACT_CHOW_MID] = True
                            mask[ActionSpace.ACT_CHOW_HIGH] = True
                            break  # If any chow is possible, mark all (simplified)
                    
                    # SKIP is always valid during reaction
                    mask[ActionSpace.ACT_SKIP] = True
            
            masks[agent] = mask
            
        return masks

    def observe(self, agent):
        agent_idx = self.agent_name_to_idx[agent]
        return self.observation_builder.build_observation(self.game, agent_idx)

    def render(self):
        self.game.display_player_hands()
        
    def close(self):
        pass
