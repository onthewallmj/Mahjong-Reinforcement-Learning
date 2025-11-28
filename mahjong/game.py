from collections import deque
import random

from .common import Wind
from .game_history import GameHistory, GameResult
from .meld import MeldType
from .player import Player
from .tile import DragonValue, FlowerValue, SeasonValue, Tile, TileSuit, WindValue
from .win import WinCondition
from .action_log import ActionLog, ActionType


class Game:

    def __init__(self, min_points_to_win: int = 3, max_point_limit: int = 13):
        # Wind of the table.
        self.table_wind: Wind = Wind.EAST
        # Index of the dealer player.
        self.dealer_index: int = 0
        # Wall tiles; acts as a double-ended queue.
        self.tiles: deque[Tile] = deque[Tile]()
        # List of discarded tiles.
        self.discards: list[Tile] = []
        # Log of actions taken during the game.
        self.action_log: list[ActionLog] = []
        # Tracks the current move index, incremented for each new action sequence.
        self.move_index: int = 0

        # Minimum points required to declare a win.
        self.min_points_to_win: int = min_points_to_win
        # Maximum points limit for scoring.
        self.max_point_limit: int = max_point_limit
        # History of game results.
        self.history: list[GameHistory] = []
        # List of players.
        self.players: list[Player] = self.initialize_players()

    # -------------------------------------------------------------------------
    # Initialization & Setup
    # -------------------------------------------------------------------------

    def initialize_players(self) -> list[Player]:
        """
        Initializes and returns a list of four Player objects for the game.
        Each player is assigned a unique seat index from 0 to 3.
        """
        return [Player(seat_index=seat_idx) for seat_idx in range(4)]

    def initialize_game(self) -> None:
        """
        Sets up a new game: determines wind, shuffles, resets states, and deals hands.
        """
        self.action_log = []
        self.move_index = 0
        self.determine_table_wind()
        self.shuffle_tiles()
        self.reset_player_game_states()
        self.deal_initial_player_hands()

        # Repeatedly replace bonus tiles (Flowers/Seasons) from players' hands until none remain.
        while any(tile.suit in [TileSuit.FLOWER, TileSuit.SEASON] for player in self.players for tile in player.hand):
            self.replace_bonus_tiles_in_player_hands()

        # Sort hands after initial deal and replacements.
        for player in self.players:
            player.sort_hand()

        print(f"Game #{self.get_game_count() + 1}")
        self.display_player_hands()

    def reset_player_game_states(self):
        """
        Resets the game-specific state for all players.
        This includes clearing their hands, discards, melds, and bonus tiles,
        preparing them for a new round or game.
        """
        for player in self.players:
            # Calculate the player's seat wind based on the current dealer.
            # The dealer always has the East wind for the round.
            # Players are ordered clockwise: East (dealer), South, West, North.
            # We use modulo 4 arithmetic to determine the relative position from the dealer.
            relative_position_from_dealer: int = (
                player.seat_index - self.dealer_index + 4) % 4
            seat_wind: Wind = Wind(relative_position_from_dealer)
            player.reset_game_state(seat_wind=seat_wind)

    def shuffle_tiles(self) -> None:
        """
        Initializes and shuffles the full set of Mahjong tiles.
        Each tile type (except Flowers and Seasons) has 4 copies.
        Flowers and Seasons have 1 copy each.
        """
        # Clear the discard pile to start a new game or round
        self.discards = []

        # Build the complete tile set, including all suited, honor, and bonus tiles
        all_tiles: list[Tile] = self._build_complete_tile_set()

        # Shuffle the tiles randomly to ensure game fairness
        random.shuffle(all_tiles)

        # Place the shuffled tiles into a deque to simulate the tile wall
        self.tiles = deque[Tile](all_tiles)

    @staticmethod
    def _build_complete_tile_set() -> list[Tile]:
        """
        Generates the complete set of tiles before shuffling.
        """
        tiles: list[Tile] = []

        # Suited tiles: Characters, Bamboo, Dots (1-9, four copies each)
        suited_suits = (TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT)
        for suit in suited_suits:
            for value in range(1, 10):
                tiles.extend(Tile(suit, value) for _ in range(4))

        # Honor tiles: Winds and Dragons (four copies each)
        for wind in (WindValue.EAST, WindValue.SOUTH, WindValue.WEST, WindValue.NORTH):
            tiles.extend(Tile(TileSuit.HONOR, wind) for _ in range(4))

        for dragon in (DragonValue.RED, DragonValue.GREEN, DragonValue.WHITE):
            tiles.extend(Tile(TileSuit.HONOR, dragon) for _ in range(4))

        # Bonus tiles: Flowers and Seasons (single copy each)
        for flower in (
            FlowerValue.PLUM,
            FlowerValue.ORCHID,
            FlowerValue.CHRYSANTHEMUM,
            FlowerValue.BAMBOO,
        ):
            tiles.append(Tile(TileSuit.FLOWER, flower))

        for season in (
            SeasonValue.SPRING,
            SeasonValue.SUMMER,
            SeasonValue.AUTUMN,
            SeasonValue.WINTER,
        ):
            tiles.append(Tile(TileSuit.SEASON, season))

        return tiles

    def deal_initial_player_hands(self):
        """
        Deals 13 initial tiles to each player from the shuffled tile stack.
        """
        for player in self.players:
            player.hand = [self.tiles.pop() for _ in range(13)]

    def determine_table_wind(self) -> Wind:
        """
        Determines the table wind by analyzing the game history.
        The table wind changes (increments) every time the dealership passes from the last player (North/3) back to the first (East/0).

        Note: If a dealer wins, they remain the dealer for the next game. In this case,
        the dealer index does not change, and no rotation is counted.
        """
        rounds = 0
        # Start with the initial dealer index (0).
        previous_dealer = 0

        # Iterate through past games to track dealer rotations.
        for game in self.history:
            current_dealer = game.dealer_index
            if previous_dealer == 3 and current_dealer == 0:
                rounds += 1
            previous_dealer = current_dealer

        # Check the transition to the current (upcoming) game's dealer.
        if previous_dealer == 3 and self.dealer_index == 0:
            rounds += 1

        self.table_wind = Wind(rounds % 4)
        return self.table_wind

    # -------------------------------------------------------------------------
    # Game Loop
    # -------------------------------------------------------------------------

    def log_action(self, action_type: ActionType, player_index: int, tile: Tile, discard_tile: Tile | None = None):
        """
        Logs a game action to the action log.
        """
        log_entry = ActionLog(self.move_index, action_type,
                              player_index, tile, discard_tile)
        self.action_log.append(log_entry)

    def play(self):
        """
        Main game loop to manage turns, player actions, and win conditions.
        """
        self.initialize_game()

        # Game always starts with the dealer seat.
        active_player_idx = self.dealer_index
        winner_seat_index = None

        # Dealer starts with 14 tiles (effectively already drew), or we deal 13 and they draw.
        # Our initialize_game deals 13. So dealer needs to draw.
        should_draw = True

        while not self.are_tiles_exhausted():
            self.move_index += 1
            current_player = self.players[active_player_idx]

            # --- Draw Phase ---
            if should_draw:
                turn_ended, winner_idx = self._handle_draw_phase(
                    current_player)
                if turn_ended:
                    if winner_idx is not None:
                        winner_seat_index = winner_idx
                        # Log win on self-draw
                        if self.action_log:
                            self.log_action(
                                ActionType.WIN, winner_idx, self.action_log[-1].inserted_tile)
                            print(self.action_log[-1])
                        else:
                            # Edge case: win on first draw
                            # Use last tile in hand? self-draw tile is already in hand.
                            # We need to know *which* tile was drawn to log it properly,
                            # but _handle_draw_phase already logs DRAW.
                            pass

                    break  # Game over (Win or Draw)

                # If _handle_draw_phase returns False, it means player drew (and possibly Kong'd) and now needs to discard.
                # BUT, if they Kong'd, _handle_draw_phase loops internally until no more Kongs.
                # So we are definitely ready to discard.

            # --- Discard Phase ---
            if not current_player.hand:
                # Should not happen if logic is correct.
                raise ValueError(
                    f"Player {current_player.seat_index} has no tiles to discard!")

            tile_to_discard = current_player.determine_tile_to_discard()
            discarded_tile = current_player.discard_tile(tile_to_discard)
            self.discards.append(discarded_tile)

            # Log discard by attaching to previous action (Draw/Pong/Chow/Kong)
            if self.action_log:
                self.action_log[-1].discarded_tile = discarded_tile
                # Print the log entry now that it is complete with the discard
                print(self.action_log[-1])

            # --- Reaction Phase ---
            # Check reactions from OTHER players to this discard.
            # Priority: Win > Kong/Pong > Chow (only next player)

            # 1. Check Win
            discarder_idx = active_player_idx
            win_claimed = False

            for offset in range(1, 4):
                seat_idx = (discarder_idx + offset) % 4
                player = self.players[seat_idx]

                is_last_tile = len(self.tiles) == 0
                can_win = player.can_win(
                    min_points_to_win=self.min_points_to_win,
                    table_wind=self.table_wind,
                    win_condition=WinCondition.WIN_FROM_DISCARD,
                    is_last_tile=is_last_tile,
                    new_tile=discarded_tile
                )

                if can_win and player.wants_to_win(self.min_points_to_win):
                    player.declare_discard_win(discarded_tile, discarder_idx)
                    self.log_action(
                        ActionType.WIN, player.seat_index, discarded_tile)
                    print(self.action_log[-1])
                    win_claimed = True
                    winner_seat_index = player.seat_index
                    break  # Head Bump priority

            if win_claimed:
                break

            # 2. Check Kong/Pong (any player)
            # We use the existing helper, but note it modifies self.discards!
            # It pops the tile if claimed.
            reaction_claimed, reactor_idx = self._handle_discard_reactions(
                active_player_idx, discarded_tile)

            if reaction_claimed:
                active_player_idx = reactor_idx
                # If Kong, need to draw replacement -> Discard.
                # If Pong, skip draw -> Discard.
                # My _handle_discard_reactions returns True for both.
                # We need to know WHICH one it was to set should_draw.
                # We can peek at the log.
                if self.action_log:
                    last_action = self.action_log[-1]
                    if last_action.action == ActionType.KONG:
                        should_draw = True
                    else:  # PONG
                        should_draw = False
                else:
                    # Should not happen if reaction_claimed is True
                    should_draw = True
                continue  # Start loop for reactor

            # 3. Check Chow (next player only)
            next_player_idx = (active_player_idx + 1) % 4
            next_player = self.players[next_player_idx]

            # _handle_chow_reaction also pops discards if claimed.
            if self._handle_chow_reaction(next_player, discarded_tile, active_player_idx):
                active_player_idx = next_player_idx
                should_draw = False  # Chow skips draw
                continue

            # --- Next Turn (Normal) ---
            active_player_idx = next_player_idx
            should_draw = True

        self.finalize_game(winner_seat_index)

    def finalize_game(self, winner_seat_index: int | None):
        """
        Handles the end of a game, updates history, and determines dealer rotation.
        """
        if winner_seat_index is not None:
            # A player won
            self.append_to_history(
                GameResult.WIN, winner_index=winner_seat_index)
            if winner_seat_index == self.dealer_index:
                # Dealer won, so they stay dealer. Round count does NOT increase.
                pass
            else:
                # Dealer lost, rotate dealer.
                self.dealer_index = (self.dealer_index + 1) % 4
        else:
            # Draw (Wall exhausted)
            self.append_to_history(GameResult.DRAW)
            # On draw, typically dealer rotates (unless special rules apply).
            # We assume rotation here.
            self.dealer_index = (self.dealer_index + 1) % 4

        self.update_player_scores()

    def _handle_draw_phase(self, current_player: Player) -> tuple[bool, int | None]:
        """
        Handles the draw phase for the current player, including:
        - Drawing a tile (and replacing bonus tiles).
        - Checking for Self-Draw Win.
        - Checking for and executing Self-Kong (with re-draws).

        Returns (True, winner_seat_index) if the turn ends immediately (due to Win or Draw),
        otherwise (False, None) if the player should proceed to discard.
        """
        consecutive_kongs = 0
        while True:
            # 1. Draw a tile (handling bonus tiles automatically).
            drawn_tile = self._draw_and_replace_bonus_tiles(current_player)

            # 2. Check for Game End (Draw) if wall is empty.
            if drawn_tile is None:
                return True, None

            # If this is a replacement draw after a Kong, we suppress the separate DRAW log
            # but we update the KONG log? No, user wants KONG to have discard.
            # So we essentially merge this DRAW into the KONG flow.
            # But we need to record the tile in the player's hand.
            current_player.draw_tile(drawn_tile)

            # If the previous action was KONG, we don't log a new DRAW.
            # The subsequent discard will be attached to the KONG action.
            is_replacement_draw = False
            if consecutive_kongs > 0:
                is_replacement_draw = True
            else:
                self.log_action(ActionType.DRAW,
                                current_player.seat_index, drawn_tile)

            # Determine potential win condition based on consecutive kongs
            current_win_condition = WinCondition.WIN_FROM_SELF_DRAW
            if consecutive_kongs == 1:
                current_win_condition = WinCondition.WIN_FROM_KONG
            elif consecutive_kongs >= 2:
                current_win_condition = WinCondition.WIN_FROM_DOUBLE_KONG

            # 3. Check for Self-Draw Win.
            is_last_tile = len(self.tiles) == 0
            if current_player.can_win(min_points_to_win=self.min_points_to_win, table_wind=self.table_wind, win_condition=current_win_condition, is_last_tile=is_last_tile) and current_player.wants_to_win(
                    min_points_to_win=self.min_points_to_win):

                current_player.declare_self_draw_win(
                    drawn_tile, win_condition=current_win_condition)
                return True, current_player.seat_index

            # 4. Check for Self-Kong.
            # If a kong is declared, the loop repeats to draw a replacement tile.
            # This handles chained kongs (Draw -> Kong -> Draw -> Kong).
            if current_player.can_self_kong(drawn_tile) and current_player.wants_to_self_kong(drawn_tile):
                # Determine if this is a promoted Kong (from a previous Pong)
                is_promoted_kong = any(m.meld_type == MeldType.PONG and m.first_tile ==
                                       drawn_tile for m in current_player.gameState.melds)

                # For now, we assume promoted kongs or concealed kongs during self-draw phase count towards self-draw wins
                # unless we track the source of the original Pong.
                # The `win_from_player_id` logic in `update_player_scores` handles the payout.
                # But we need to ensure that if this Kong leads to a win, we know if it was originally from a discard.
                # Since we don't track original Pong source yet, we'll assume self-draw payment for now.

                # If we have a previous action (likely DRAW or previous KONG) that hasn't been printed,
                # we should print it now because the turn is continuing into a new Kong.
                if self.action_log:
                    print(self.action_log[-1])

                current_player.declare_self_kong(drawn_tile)
                self.log_action(ActionType.KONG,
                                current_player.seat_index, drawn_tile)
                consecutive_kongs += 1
                continue

            # 5. Proceed to Discard Phase (no win or kong).
            return False, None

    def _handle_discard_reactions(self, player_who_would_draw_next_idx: int, most_recent_discard: Tile) -> tuple[bool, int | None]:
        """
        Checks for Kong/Pong reactions from any player (except the discarder) to the most recent discard.

        Returns (True, reacting_player_seat_index) if a reaction occurred and the discard was taken, otherwise (False, None).

        Priority: Kong > Pong, and first player encountered.
        """
        if not most_recent_discard:
            return False, None

        # Determine the index of the player who made the discard.
        discarder_idx = (player_who_would_draw_next_idx - 1 + 4) % 4

        for player in self.players:
            # The player who just discarded cannot react with a Pong or Kong to their own discard.
            if player.seat_index == discarder_idx:
                continue

            # Check for Kong
            if player.can_kong(most_recent_discard) and player.wants_to_kong(most_recent_discard):
                player.declare_kong(most_recent_discard)
                self.discards.pop()  # The discarded tile is taken by the player who Kong'd
                self.log_action(ActionType.KONG,
                                player.seat_index, most_recent_discard)
                return True, player.seat_index

            # If no Kong, check for Pong
            elif player.can_pong(most_recent_discard) and player.wants_to_pong(most_recent_discard):
                player.declare_pong(most_recent_discard)
                self.discards.pop()  # The discarded tile is taken by the player who Pong'd
                self.log_action(ActionType.PONG,
                                player.seat_index, most_recent_discard)
                return True, player.seat_index
        return False, None

    def _handle_chow_reaction(self, current_player: Player,  most_recent_discard: Tile, discarding_player_index: (0 | 1 | 2 | 3)) -> bool:
        """
        Checks if the `current_player` (who would normally draw next) can and wants to Chow the most recent discard.
        If so, performs the Chow and removes the discard. Returns True if Chow occurred, False otherwise.
        Chow is typically only possible for the player immediately following the discarder.
        """

        if current_player.can_chow(most_recent_discard, discarding_player_index) and current_player.wants_to_chow(most_recent_discard):
            current_player.declare_chow(most_recent_discard)
            self.discards.pop()  # The chowed tile is removed from discards.
            self.log_action(ActionType.CHOW,
                            current_player.seat_index, most_recent_discard)
            return True
        return False

    # -------------------------------------------------------------------------
    # Game Helper Methods
    # -------------------------------------------------------------------------

    def replace_bonus_tiles_in_player_hands(self):
        """
        Processes each player's initial hand to identify and replace bonus tiles (Flowers and Seasons).
        Bonus tiles are moved from the player's hand to their bonus_tiles collection,
        and replacement tiles are drawn from the main tile stack to maintain hand size.
        """
        # Define which suits are considered bonus tiles (Flowers and Seasons).
        bonus_suits = (TileSuit.FLOWER, TileSuit.SEASON)

        # Iterate through all players to process their hands for bonus tiles.
        for player in self.players:
            # Identify all bonus tiles in the player's hand.
            bonus_tiles = [
                tile for tile in player.hand if tile.suit in bonus_suits]

            # If this player has no bonus tiles, skip them.
            if not bonus_tiles:
                continue

            # Move identified bonus tiles from hand to the player's bonus_tiles collection.
            player.bonus_tiles.extend(bonus_tiles)
            # Remove bonus tiles from the player's hand, retaining only non-bonus tiles.
            player.hand = [
                tile for tile in player.hand if tile.suit not in bonus_suits]

            # For each removed bonus tile, draw a replacement tile from the wall.
            self._draw_replacements_from_wall(player, len(bonus_tiles))

    def _draw_replacements_from_wall(self, player: Player, count: int) -> None:
        """
        Draws up to `count` replacement tiles for the specified player from the wall.
        """
        for _ in range(count):
            if not self.tiles:
                break
            player.hand.append(self.tiles.popleft())

    def _draw_and_replace_bonus_tiles(self, player: Player) -> Tile | None:
        """
        Player draws a tile, replacing bonus tiles (Flowers/Seasons) until a non-bonus tile is drawn
        or the main tile stack (wall) is empty.
        Returns the last non-bonus tile drawn, or None if the wall is exhausted before drawing one.
        """
        latest_non_bonus_tile_drawn = None
        while True:
            if not self.tiles:
                # No more tiles left in the wall to draw from.
                break

            drawn_tile = self.tiles.pop()

            if drawn_tile.suit in [TileSuit.FLOWER, TileSuit.SEASON]:
                # If the drawn tile is a bonus tile, add it to bonus_tiles and draw again.
                player.bonus_tiles.append(drawn_tile)
            else:
                # Non-bonus tile drawn, so this is the final tile drawn for this turn.
                latest_non_bonus_tile_drawn = drawn_tile
                break
        return latest_non_bonus_tile_drawn

    def are_tiles_exhausted(self) -> bool:
        """
        Determines if the tile wall is exhausted (no tiles left).
        """
        return len(self.tiles) == 0

    def append_to_history(self, result: GameResult, winner_index: int | None = None) -> GameHistory:
        """
        Appends a new game result to the game's history.
        """
        new_history_entry = GameHistory(
            index=self.get_game_count(),
            result=result,
            table_wind=self.table_wind,
            dealer_index=self.dealer_index,
            winner_index=winner_index
        )
        self.history.append(new_history_entry)
        return new_history_entry

    def get_game_count(self) -> int:
        return len(self.history)

    def update_player_scores(self):
        """
        Updates the scores of all players based on their wins.
        If the win is a self-draw, all other non-winning players pay the winner.
        If it's a regular win (discard), only the discarder pays the winner.
        """
        # Reset scores to 0 or carry over? Assuming cumulative or round-based.
        # Since this method is called potentially multiple times or at end of game,
        # let's assume it calculates based on self.wins history or just the latest state.
        # However, `Player.wins` stores a list of wins.
        # Let's iterate through all players and their wins to calculate score transfers.

        # Reset scores first if recalculating from scratch, or handle incrementally.
        # Given `Player.score` is a simple integer, let's recalculate from scratch for safety.
        for player in self.players:
            player.score = 0

        for player_idx, player in enumerate(self.players):
            for win in player.wins:
                win_score = win.calculate_score(self.max_point_limit)

                if win.win_from_player_id is not None:
                    # Specific player pays (Discard win, or Kong win initiated by discard)
                    self.players[win.win_from_player_id].score -= win_score
                    player.score += win_score
                else:
                    # All others pay (Self-draw win, or Kong win initiated by self-draw)
                    for other_idx, other_player in enumerate(self.players):
                        if other_idx != player_idx:
                            other_player.score -= win_score
                            player.score += win_score

    def display_player_hands(self):
        for idx, player in enumerate(self.players, 1):
            print(f"Player {idx}'s Hand:")
            player.display_hand()
            print(f"Player {idx}'s Bonus Tiles:")
            player.display_bonus_tiles()
            print()
