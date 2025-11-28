from common import Wind
from meld import Meld
from tile import Tile, TileSuit
from win import Win


class PlayerGameState:
    """
    Represents the current game state pertinent to a single player.
    This includes their hand, melds, bonus tiles, and points.
    """

    def __init__(self, seatWind: Wind):
        self.seatWind = seatWind
        self.hand = []
        self.bonus_tiles = []  # Flower and Season Tiles
        self.melds: list[Meld] = []
        self.point_sources = []  # List of PointSource objects


class Player:

    def __init__(self, seat_index: (0 | 1 | 2 | 3)):
        """
        Initialize a player with a fixed seat index and empty game state.

        The actual seat wind (East/South/West/North) is computed by the
        MahjongGame and passed into reset_game_state; here we just store the
        stable seat index 0-3.
        """
        self.starting_index = seat_index
        self.wins = []  # List of Win objects.
        self.score = 0  # Player's current score.
        self.gameState = PlayerGameState(Wind(seat_index))

    def reset_game_state(self, seat_wind: Wind):
        """
        Resets the player's game state for a new round.

        `seatWind` is typically a Wind enum instance provided by MahjongGame.
        We store it on the PlayerGameState to avoid importing the enum here
        and creating circular dependencies.
        """
        self.gameState = PlayerGameState(seat_wind)

    @property
    def seat_index(self) -> int:
        """
        Return this player's fixed seat index (0-3) at the table.
        """
        return self.starting_index

    @property
    def hand(self):
        """
        Shortcut accessor for the player's hand tiles stored in gameState.
        """
        return self.gameState.hand

    @hand.setter
    def hand(self, value):
        self.gameState.hand = value

    @property
    def bonus_tiles(self):
        """
        Shortcut accessor for the player's bonus tiles (flowers/seasons).
        """
        return self.gameState.bonus_tiles

    @bonus_tiles.setter
    def bonus_tiles(self, value):
        self.gameState.bonus_tiles = value

    def sort_hand(self):
        """
        Sort the player's hand by suit and value for easier management.
        """
        self.gameState.hand.sort(
            key=lambda tile: (
                tile.suit.value,
                tile.value if isinstance(
                    tile.value, int) else tile.value.value,
            )
        )

    def determine_tile_to_discard(self) -> Tile:
        """
        Decide which tile to discard.
        """
        # TODO: Implement strategy via PyTorch / policy network.
        pass

    def discard_tile(self, tile: Tile) -> Tile:
        """
        Remove a tile from the player's hand and return it.
        """
        # Iterate through the hand to find and remove the matching tile based on suit and value.
        for idx, hand_tile in enumerate(self.gameState.hand):
            if hand_tile.suit == tile.suit and hand_tile.value == tile.value:
                self.hand.pop(idx)
                break
        return tile

    def can_chow(
        self,
        discarded_tile: Tile,
        discarding_player_index: (0 | 1 | 2 | 3),
    ) -> bool:
        """
        Return True if the player can declare a Chow with the discarded tile.

        A Chow can only be declared by the next player in turn order, and only
        on a suited numbered tile (Character, Bamboo, or Dot).
        """
        # Ensure this player is the next in turn order relative to the discarder.
        if (self.gameState.seatWind.value - discarding_player_index) % 4 != 1:
            return False

        # Chow is only possible on suited numbered tiles.
        if discarded_tile.suit not in [TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT]:
            return False

        tile_value = discarded_tile.value
        if not isinstance(tile_value, int):
            return False

        needed_tiles = [
            [Tile(discarded_tile.suit, tile_value - 2),
             Tile(discarded_tile.suit, tile_value - 1)],
            [Tile(discarded_tile.suit, tile_value - 1),
             Tile(discarded_tile.suit, tile_value + 1)],
            [Tile(discarded_tile.suit, tile_value + 1),
             Tile(discarded_tile.suit, tile_value + 2)],
        ]

        for tile_pair in needed_tiles:
            if all(tile in self.gameState.hand for tile in tile_pair):
                return True

        return False

    def can_pong(self, discarded_tile: Tile) -> bool:
        """
        Return True if the player can declare a Pong.

        A Pong can be declared if the player has two identical tiles in hand.
        """
        return self.gameState.hand.count(discarded_tile) >= 2

    def can_kong(self, discarded_tile: Tile) -> bool:
        """
        Return True if the player can declare a Kong.

        A Kong can be declared if the player has three identical tiles in hand.
        """
        return self.gameState.hand.count(discarded_tile) >= 3

    def can_win(self, min_points_to_win: int) -> bool:
        """
        Return True if the player can declare a win.
        """
        pass

    def wants_to_chow(self, discarded_tile: Tile) -> bool:
        """
        Decision hook: whether the player *chooses* to Chow.
        """
        # TODO: Implement strategy via PyTorch / policy network.
        return True

    def wants_to_pong(self, discarded_tile: Tile) -> bool:
        """
        Decision hook: whether the player *chooses* to Pong.
        """
        # TODO: Implement strategy via PyTorch / policy network.
        return True

    def wants_to_kong(self, discarded_tile: Tile) -> bool:
        """
        Decision hook: whether the player *chooses* to Kong.
        """
        # TODO: Implement strategy via PyTorch / policy network.
        return True

    def wants_to_win(self, min_points_to_win: int) -> bool:
        """
        Decision hook: whether the player *chooses* to declare a win.
        """
        # TODO: Implement strategy via PyTorch / policy network.
        return True

    def chow(self, tile1: Tile, tile2: Tile, discarded_tile: Tile):
        """
        Declare a Chow meld with the given three tiles.

        Removes tile1 and tile2 from the player's hand and adds a Chow meld to
        the player's meld list.
        """
        if not (tile1 in self.gameState.hand and tile2 in self.gameState.hand):
            raise ValueError(
                "Cannot declare Chow: one or more tiles not in hand.")

        discarded_tile_value = discarded_tile.value
        if not isinstance(discarded_tile_value, int) or \
           discarded_tile.suit not in [TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT]:
            raise ValueError(
                "Cannot declare Chow: discarded tile is not a numbered suit tile."
            )

        all_chow_tiles = sorted(
            [tile1, tile2, discarded_tile], key=lambda t: t.value)

        # Ensure all tiles are of the same suit.
        if not (all_chow_tiles[0].suit == all_chow_tiles[1].suit == all_chow_tiles[2].suit):
            raise ValueError(
                "Cannot declare Chow: tiles are not of the same suit.")

        # Ensure all tiles are consecutive.
        if not (
            all_chow_tiles[0].value == all_chow_tiles[1].value - 1
            and all_chow_tiles[1].value == all_chow_tiles[2].value - 1
        ):
            raise ValueError(
                "Cannot declare Chow: tiles do not form a consecutive sequence."
            )

        self.gameState.hand.remove(tile1)
        self.gameState.hand.remove(tile2)
        chow_meld = Meld.create_chow(tile1, tile2, discarded_tile)
        self.gameState.melds.append(chow_meld)

    def pong(self, discarded_tile: Tile):
        """
        Declare a Pong meld with the given tile.

        Removes two identical tiles from the player's hand and adds a Pong meld.
        """
        self.gameState.hand.remove(discarded_tile)
        self.gameState.hand.remove(discarded_tile)
        pong_meld = Meld.create_pong(
            discarded_tile, discarded_tile, discarded_tile)
        self.gameState.melds.append(pong_meld)

    def kong(self, discarded_tile: Tile):
        """
        Declare a Kong meld with the given tile.

        Removes three identical tiles from the player's hand and adds a Kong meld.
        """
        self.gameState.hand.remove(discarded_tile)
        self.gameState.hand.remove(discarded_tile)
        self.gameState.hand.remove(discarded_tile)
        kong_meld = Meld.create_kong(
            discarded_tile, discarded_tile, discarded_tile, discarded_tile
        )
        self.gameState.melds.append(kong_meld)

    def draw_tile(self, drawn_tile: Tile) -> Tile:
        self.gameState.hand.append(drawn_tile)

    def declare_chow(self, discarded_tile: Tile) -> Tile | None:
        """
        Attempt to declare a Chow using the discarded tile, then discard.

        Returns the tile discarded after forming the Chow, or None if no Chow
        could be formed.
        """
        if not isinstance(discarded_tile.value, int) or \
                discarded_tile.suit not in [TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT]:
            return None

        tile_value = discarded_tile.value
        tile_suit = discarded_tile.suit
        chosen_tile1, chosen_tile2 = None, None

        # Case 1: Discarded tile is the middle tile (X-1, X, X+1).
        if 1 < tile_value < 9:
            needed_tile1 = Tile(tile_suit, tile_value - 1)
            needed_tile2 = Tile(tile_suit, tile_value + 1)
            if (
                needed_tile1 in self.gameState.hand
                and needed_tile2 in self.gameState.hand
            ):
                chosen_tile1, chosen_tile2 = needed_tile1, needed_tile2

        # Case 2: Discarded tile is the lowest tile (X, X+1, X+2).
        if chosen_tile1 is None and tile_value < 8:
            needed_tile1 = Tile(tile_suit, tile_value + 1)
            needed_tile2 = Tile(tile_suit, tile_value + 2)
            if (
                needed_tile1 in self.gameState.hand
                and needed_tile2 in self.gameState.hand
            ):
                chosen_tile1, chosen_tile2 = needed_tile1, needed_tile2

        # Case 3: Discarded tile is the highest tile (X-2, X-1, X).
        if chosen_tile1 is None and tile_value > 2:
            needed_tile1 = Tile(tile_suit, tile_value - 2)
            needed_tile2 = Tile(tile_suit, tile_value - 1)
            if (
                needed_tile1 in self.gameState.hand
                and needed_tile2 in self.gameState.hand
            ):
                chosen_tile1, chosen_tile2 = needed_tile1, needed_tile2

        if chosen_tile1 is None:
            return None

        self.chow(chosen_tile1, chosen_tile2, discarded_tile)
        tile_to_discard = self.determine_tile_to_discard()
        self.discard_tile(tile_to_discard)
        self.sort_hand()
        return tile_to_discard

    def declare_pong(self, discarded_tile: Tile) -> Tile:
        """
        Declare a Pong meld with the discarded tile and then discard a tile.
        """
        self.pong(discarded_tile)
        tile_to_discard = self.determine_tile_to_discard()
        self.discard_tile(tile_to_discard)
        self.sort_hand()
        return tile_to_discard

    def declare_kong(self, discarded_tile: Tile) -> Tile:
        """
        Declare a Kong meld with the given tile and then discard a tile.
        """
        self.kong(discarded_tile)
        tile_to_discard = self.determine_tile_to_discard()
        self.discard_tile(tile_to_discard)
        self.sort_hand()
        return tile_to_discard

    def declare_self_draw_win(self, winning_tile: Tile):
        """
        Record a self-drawn win for the player.
        """
        self.wins.append(
            Win.create_self_draw_win(
                winning_tile=winning_tile,
                hand_tiles=self.gameState.hand,
                bonus_tiles=self.gameState.bonus_tiles,
                point_sources=[],
            )
        )

    def declare_discard_win(self, winning_tile: Tile, win_from_player_id: int):
        """
        Record a win declared on a tile discarded by another player.
        """
        self.wins.append(
            Win.create_discard_win(
                winning_tile=winning_tile,
                hand_tiles=self.gameState.hand,
                bonus_tiles=self.gameState.bonus_tiles,
                point_sources=[],
                win_from_player_id=win_from_player_id,
            )
        )

    def update_score(self, point_limit: int) -> int:
        """
        Calculate and update the player's total score from recorded wins.
        """
        total_score = 0
        for win in self.wins:
            total_score += win.get_score(max_point_limit=point_limit)
        self.score = total_score
        return total_score

    def display_hand(self) -> None:
        """
        Display the player's hand tiles in a readable string format.
        """
        output = []
        for tile in self.hand:
            output.append(str(tile))
        print(", ".join(output))

    def display_bonus_tiles(self) -> None:
        """
        Display the player's bonus tiles in a readable string format.
        """
        output = []
        for tile in self.bonus_tiles:
            output.append(str(tile))
        if not output:
            print("None")
        else:
            print(", ".join(output))
