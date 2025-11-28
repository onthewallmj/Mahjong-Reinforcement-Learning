from enum import Enum


class GameResult(Enum):
    WIN = "Win"
    DRAW = "Draw"


class GameHistory:
    """Represents the history of a single game."""

    def __init__(self, index: int, result: GameResult):
        self.index = index
        self.result = result
