import random

from common import Wind
from game_history import GameHistory, GameResult
from player import Player
from tile import DragonValue, FlowerValue, SeasonValue, Tile, TileSuit, WindValue


class MahjongGame:

    def __init__(self, min_points_to_win: int = 3, max_point_limit: int = 13):
        self.tableWind = Wind.EAST
        self.dealer_index = 0  # Index of the dealer player
        self.tiles = []  # List of all tiles in the game
        self.discards = []  # List of discarded tiles

        # Minimum points required to declare a win.
        self.min_points_to_win = min_points_to_win
        # Maximum points limit for scoring.
        self.max_point_limit = max_point_limit
        self.history = []
        self.players = self.create_players()

    def create_players(self) -> list[Player]:
        """
        Initializes and returns a list of four Player objects for the game.
        Each player is assigned a unique seat index from 0 to 3.
        """
        players = []
        for seat_idx in range(4):
            player = Player(seat_index=seat_idx)
            players.append(player)
        return players

    def initialize_game(self) -> None:
        self.shuffle_tiles()
        self.reset_player_game_states()
        self.deal_initial_player_hands()

        # Repeatedly replace bonus tiles (Flowers/Seasons) from players' hands until none remain.
        while any(tile.suit in [TileSuit.FLOWER, TileSuit.SEASON] for player in self.players for tile in player.hand):
            self.replace_bonus_tiles_in_player_hands()

        print(f"Game #{self.get_game_count()}")

        for idx, player in enumerate(self.players, 1):
            player.sort_hand()
            print(f"Player {idx}'s Hand:")
            player.display_hand()
            print(f"Player {idx}'s Bonus Tiles:")
            player.display_bonus_tiles()
            print()

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
            relative_position_from_dealer = (
                player.seat_index - self.dealer_index + 4) % 4
            seat_wind = Wind(relative_position_from_dealer)
            player.reset_game_state(seat_wind=seat_wind)

    def shuffle_tiles(self) -> list[Tile]:
        """
        Initializes and shuffles the full set of Mahjong tiles.
        Each tile type (except Flowers and Seasons) has 4 copies.
        Flowers and Seasons have 1 copy each.
        """
        # Reset tiles and discards arrays.
        self.tiles = []
        self.discards = []

        all_tiles = [
            # Standard Suit Tiles (4 copies each of 1-9 for each suit)

            # Characters (Wan)
            *[Tile(TileSuit.CHARACTER, i) for i in range(1, 10)] * 4,
            # Bamboo (Sou)
            *[Tile(TileSuit.BAMBOO, i) for i in range(1, 10)] * 4,
            # Dots (Pin)
            *[Tile(TileSuit.DOT, i) for i in range(1, 10)] * 4,

            # Honor Tiles (4 copies each)

            # Winds
            *[Tile(TileSuit.HONOR, WindValue.EAST) for _ in range(4)],
            *[Tile(TileSuit.HONOR, WindValue.SOUTH) for _ in range(4)],
            *[Tile(TileSuit.HONOR, WindValue.WEST) for _ in range(4)],
            *[Tile(TileSuit.HONOR, WindValue.NORTH) for _ in range(4)],
            # Dragons
            *[Tile(TileSuit.HONOR, DragonValue.RED) for _ in range(4)],
            *[Tile(TileSuit.HONOR, DragonValue.GREEN) for _ in range(4)],
            *[Tile(TileSuit.HONOR, DragonValue.WHITE) for _ in range(4)],

            # Bonus Tiles (1 copy each)

            # Flowers
            *[Tile(TileSuit.FLOWER, FlowerValue.PLUM)],
            *[Tile(TileSuit.FLOWER, FlowerValue.ORCHID)],
            *[Tile(TileSuit.FLOWER, FlowerValue.CHRYSANTHEMUM)],
            *[Tile(TileSuit.FLOWER, FlowerValue.BAMBOO)],
            # Seasons
            *[Tile(TileSuit.SEASON, SeasonValue.SPRING)],
            *[Tile(TileSuit.SEASON, SeasonValue.SUMMER)],
            *[Tile(TileSuit.SEASON, SeasonValue.AUTUMN)],
            *[Tile(TileSuit.SEASON, SeasonValue.WINTER)],
        ]

        # Randomly shuffle all generated tiles
        random.shuffle(all_tiles)
        # Assign the shuffled tiles to the game's tile list
        self.tiles = all_tiles

    def deal_initial_player_hands(self):
        """
        Deals 13 initial tiles to each player from the shuffled tile stack.
        """
        for player in self.players:
            player.hand = [self.tiles.pop() for _ in range(13)]

    def replace_bonus_tiles_in_player_hands(self):
        """
        Processes each player's initial hand to identify and replace bonus tiles (Flowers and Seasons).
        Bonus tiles are moved from the player's hand to their bonus_tiles collection,
        and replacement tiles are drawn from the main tile stack to maintain hand size.
        """
        for player in self.players:
            initial_hand_size = len(player.hand)
            new_hand = []

            # Extract Flower and Season tiles from the player's hand and move them to bonus_tiles.
            for tile in player.hand:
                if tile.suit in [TileSuit.FLOWER, TileSuit.SEASON]:
                    player.bonus_tiles.append(tile)
                else:
                    new_hand.append(tile)
            player.hand = new_hand

            # Draw replacement tiles for the removed bonus tiles.
            num_tile_replacements_needed = initial_hand_size - len(player.hand)

            for _ in range(num_tile_replacements_needed):
                if self.tiles:
                    player.hand.append(self.tiles.pop())

    def play(self):
        """
        Main game loop to manage turns, player actions, and win conditions.
        """
        self.initialize_game()

        # Game always starts with the dealer seat.
        active_player_idx = self.dealer_index

        while not self.is_game_over():
            current_player = self.players[active_player_idx]
            most_recent_discard = self.discards[-1] if self.discards else None

            # Flag to track if a discard was taken by a meld (Kong/Pong/Chow).
            discard_taken_by_meld = False
            new_active_player_idx = None

            # 1. Check for high-priority reactions (Kong/Pong) from other players to the most recent discard.
            # This only happens if a discard has actually occurred.
            if most_recent_discard is not None:
                # `_handle_discard_reactions` determines if a Kong/Pong occurred and returns the reacting player's index.
                discard_taken_by_meld, new_active_player_idx = self._handle_discard_reactions(
                    player_who_would_draw_next_idx=active_player_idx,
                    most_recent_discard=most_recent_discard
                )

            if discard_taken_by_meld:
                # If a Kong or Pong occurred, the turn immediately passes to the reacting player.
                active_player_idx = new_active_player_idx
                # Skip remaining actions for this 'original' turn and start the new player's turn.
                continue

            # If no Kong/Pong, it's `current_player`'s turn.
            # 2. Check for Chow reaction from the `current_player` (only if a discard exists).
            if most_recent_discard is not None:
                # `_handle_chow_reaction` checks if the current player can and wants to Chow.
                discarding_player_index = (active_player_idx - 1 + 4) % 4
                if self._handle_chow_reaction(current_player, most_recent_discard, discarding_player_index):
                    discard_taken_by_meld = True

            # 3. If no meld reaction occurred (Kong/Pong/Chow), the current player draws a tile from the wall.
            if not discard_taken_by_meld:
                drawn_tile = self._draw_and_replace_bonus_tiles(current_player)
                current_player.draw_tile(drawn_tile)

                if current_player.can_win(min_points_to_win=self.min_points_to_win) and current_player.wants_to_win(
                        min_points_to_win=self.min_points_to_win):

                    current_player.declare_self_draw_win(drawn_tile)

                    # If a player declares a win, they do not make a discard.
                    continue

            # 4. Player discards a tile.
            # A player always discards a tile if their hand is not empty after drawing or completing a meld.
            if current_player.hand:
                # Retain the original logic of discarding the first tile as a placeholder for player strategy.
                discarded_tile = current_player.discard_tile(
                    current_player.hand[0])

                self.discards.append(discarded_tile)

            # 5. Move to the next player's turn.
            active_player_idx = (active_player_idx + 1) % 4

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
                # If the drawn tile is a bonus tile, move it from player's hand to bonus_tiles.
                player.bonus_tiles.append(player.hand.pop())
            else:
                # Non-bonus tile drawn, so this is the final tile drawn for this turn.
                latest_non_bonus_tile_drawn = drawn_tile
                break
        return latest_non_bonus_tile_drawn

    def _handle_discard_reactions(self, player_who_would_draw_next_idx: int, most_recent_discard: Tile) -> tuple[bool, int | None]:
        """
        Checks for Kong/Pong reactions from any player (except the discarder) to the most recent discard.
        Returns (True, reacting_player_seat_index) if a reaction occurred and the discard was taken,
        otherwise (False, None). Priority: Kong > Pong, and first player encountered (simplification).
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
                return True, player.seat_index

            # If no Kong, check for Pong
            elif player.can_pong(most_recent_discard) and player.wants_to_pong(most_recent_discard):
                player.declare_pong(most_recent_discard)
                self.discards.pop()  # The discarded tile is taken by the player who Pong'd
                return True, player.seat_index
        return False, None

    def _handle_chow_reaction(self, current_player: Player,  most_recent_discard: Tile, discarding_player_index: (0 | 1 | 2 | 3)) -> bool:
        """
        Checks if the `current_player` (who would normally draw next) can and wants to Chow the most recent discard.
        If so, performs the Chow and removes the discard. Returns True if Chow occurred, False otherwise.
        Chow is typically only possible for the player immediately following the discarder.
        """
        if not most_recent_discard:
            return False

        if current_player.can_chow(most_recent_discard, discarding_player_index) and current_player.wants_to_chow(most_recent_discard):
            current_player.declare_chow(most_recent_discard)
            self.discards.pop()  # The chowed tile is removed from discards
            return True
        return False

    def is_game_over(self) -> bool:
        """
        Determines if the game is over based on win conditions or tile exhaustion.
        """
        # Placeholder logic: game ends when there are no tiles left to draw.
        return len(self.tiles) == 0

    def append_to_history(self, result: GameResult) -> GameHistory:
        """
        Appends a new game result to the game's history.

        Args:
            result (GameResult): The result of the completed game round.

        Returns:
            GameHistory: The newly created game history entry.
        """
        new_history_entry = GameHistory(
            index=self.get_game_count(), result=result)
        self.history.append(new_history_entry)
        return new_history_entry

    def get_game_count(self) -> int:
        return len(self.history)
