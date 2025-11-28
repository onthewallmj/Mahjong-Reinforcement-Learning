from enum import Enum
from types import MappingProxyType

from point_source import PointSource
from tile import Tile

# Maps points to base score values according to standard rules.
POINT_TO_SCORE_MAP: MappingProxyType[int, int] = MappingProxyType[int, int]({
    3: 8,
    4: 16,
    5: 24,
    6: 32,
    7: 48,
    8: 64,
    9: 96,
    10: 128,
    11: 192,
    12: 256,
    13: 384,
})


class WinCondition(Enum):
    """
    Represents the condition under which a player wins a game.
    """
    # Win by drawing the winning tile yourself from the wall.
    WIN_FROM_SELF_DRAW = 0
    # Win by claiming a tile discarded by another player.
    WIN_FROM_DISCARD = 1
    # Win by drawing a replacement tile after a Kong.
    WIN_FROM_KONG = 2
    # Win by drawing a replacement tile after two consecutive Kongs.
    WIN_FROM_DOUBLE_KONG = 3


class Win:
    """
    Represents a win in Mahjong.
    """

    def __init__(self, winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource], win_condition: WinCondition):
        self.winning_tile = winning_tile
        self.hand_tiles = hand_tiles
        self.bonus_tiles = bonus_tiles
        self.point_sources = point_sources
        self.win_condition = win_condition
        # ID of the player who discarded the winning tile (if applicable).
        self.win_from_player_id = None
        # Score for the win.
        self.score = self.calculate_score()
        # Total points for the win.
        self.points = self.get_points()

    @staticmethod
    def create_discard_win(winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource], win_from_player_id: int):
        """
        Creates a Win object representing a win from a discarded tile.
        """
        win = Win(winning_tile, hand_tiles,
                  bonus_tiles, point_sources, WinCondition.WIN_FROM_DISCARD)
        win.win_from_player_id = win_from_player_id
        return win

    @staticmethod
    def create_self_draw_win(winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource], win_condition: WinCondition = WinCondition.WIN_FROM_SELF_DRAW):
        """
        Creates a Win object representing a self-draw win.
        """
        win = Win(winning_tile, hand_tiles,
                  bonus_tiles, point_sources, win_condition)
        return win

    def get_points(self) -> int:
        """
        Calculates the total points for the win.
        """
        return sum(point_source.value for point_source in self.point_sources)

    def calculate_score(self, max_point_limit: int = 13) -> int:
        """
        Calculates the score for the win.
        """
        total_points = self.get_total_points()
        points_to_lookup = min(total_points, max_point_limit)
        # Default to 0 if not found, though all standard points should be in the map up to 13.
        base_score = POINT_TO_SCORE_MAP.get(points_to_lookup, 0)

        # Check if it is any type of self-draw win
        is_self_draw = self.win_condition in [
            WinCondition.WIN_FROM_SELF_DRAW,
            WinCondition.WIN_FROM_KONG,
            WinCondition.WIN_FROM_DOUBLE_KONG
        ]

        return int(base_score * 1.5) if is_self_draw else base_score

    def get_total_points(self) -> int:
        return sum(p.value for p in self.point_sources)
