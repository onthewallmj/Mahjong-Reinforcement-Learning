from enum import Enum
from tile import Tile
from point_source import PointSource

# Maps points to base score values according to standard rules.
POINT_TO_SCORE_MAP = {
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
    13: 384
}


class WinCondition(Enum):
    SELF_DRAW = 0
    DISCARD = 1


class Win:

    def __init__(self, winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource], win_condition: WinCondition):
        self.winning_tile = winning_tile
        self.hand_tiles = hand_tiles
        self.bonus_tiles = bonus_tiles
        self.point_sources = point_sources
        self.win_condition = win_condition
        # ID of the player who discarded the winning tile (if applicable).
        self.win_from_player_id = None

    @staticmethod
    def create_discard_win(winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource], win_from_player_id: int):
        """Creates a Win object representing a win from a discarded tile."""
        win = Win(winning_tile, hand_tiles,
                  bonus_tiles, point_sources, WinCondition.DISCARD)
        win.win_from_player_id = win_from_player_id
        return win

    @staticmethod
    def create_self_draw_win(winning_tile: Tile, hand_tiles: list[Tile], bonus_tiles: list[Tile], point_sources: list[PointSource]):
        """Creates a Win object representing a self-draw win."""
        win = Win(winning_tile, hand_tiles,
                  bonus_tiles, point_sources, WinCondition.SELF_DRAW)
        return win

    def get_total_points(self) -> int:
        """Calculates the total points for the win."""
        return sum(point_source.value for point_source in self.point_sources)

    def get_score(self, max_point_limit: int = 13) -> int:
        """Calculates the score for the win."""
        total_points = self.get_total_points()
        points_to_lookup = min(total_points, max_point_limit)
        # Default to 0 if not found, though all standard points should be in the map up to 13.
        base_score = POINT_TO_SCORE_MAP.get(points_to_lookup, 0)

        if self.win_condition == WinCondition.SELF_DRAW:
            return int(base_score * 1.5)
        return base_score
