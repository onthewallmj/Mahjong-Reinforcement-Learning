from enum import Enum
from .common import Wind


class GameResult(Enum):
    """
    Represents the result of a game.
    """

    # The game ended in a draw, e.g., all tiles depleted with no winner
    DRAW = 0
    # The game ended with a win by one or more players.
    WIN = 1


class GameHistory:
    """
    Represents the history of a single game.
    """

    def __init__(self, index: int, result: GameResult, dealer_index: int, table_wind: Wind, winner_index: int | None = None):
        # Index of the game.
        self.index = index
        # Result of the game.
        self.result = result
        # Wind of the table.
        self.table_wind = table_wind
        # Index of the dealer player.
        self.dealer_index = dealer_index
        # Index of the winner (if any).
        self.winner_index = winner_index
