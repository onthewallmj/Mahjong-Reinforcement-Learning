from enum import Enum
from .tile import Tile


class ActionType(Enum):
    """
    Represents the type of action taken by a player during the Mahjong game.
    Each action type is associated with a specific Mahjong game action.
    """
    DRAW = "draw"
    DISCARD = "discard"
    CHOW = "chow"
    PONG = "pong"
    KONG = "kong"
    WIN = "win"


class ActionLog:
    """
    Represents a log entry for an action taken by a player during the Mahjong game.
    Each entry records the type of action (e.g., draw, discard, chow, pong, kong, win),
    the index of the player performing the action, the relevant tile involved,
    and (if applicable) the tile that was discarded as part of the action.
    """

    def __init__(self, action: ActionType, player_index: (0 | 1 | 2 | 3), tile: Tile, discard_tile: Tile | None = None):
        # The type of action taken (draw, discard, chow, pong, kong, win).
        self.action = action
        # Index of the player who performed the action (0–3).
        self.player_index = player_index
        # The tile involved in the action (e.g., drawn or discarded tile).
        self.tile = tile
        # The tile that was discarded (if relevant to the action, e.g., for chow/pong), otherwise None.
        self.discard_tile = discard_tile
