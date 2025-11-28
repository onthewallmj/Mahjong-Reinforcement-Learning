from enum import Enum, auto


class Phase(Enum):
    """
    Represents the current phase of the game loop.
    """
    DRAW = auto()
    DISCARD = auto()
    REACTION = auto()
    GAME_OVER = auto()
