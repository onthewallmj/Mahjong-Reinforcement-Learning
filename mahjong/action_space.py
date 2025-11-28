from enum import Enum, auto

class ActionType(Enum):
    DISCARD = auto()
    SKIP = auto()
    CHOW = auto()  # Generic Chow, specific type handled by detail
    PONG = auto()
    KONG = auto()
    SELF_KONG = auto()
    WIN = auto()

class ActionSpace:
    """
    Maps RL agent discrete actions (0-N) to semantic Mahjong moves.
    
    Action Mapping:
    0-33:  Discard Tile (Index 0-33)
    34:    Skip (Pass)
    35:    Chow (Sequence: Left / X-2, X-1)
    36:    Chow (Sequence: Middle / X-1, X+1)
    37:    Chow (Sequence: Right / X+1, X+2)
    38:    Pong
    39:    Kong (from discard)
    40:    Self-Kong (from hand)
    41:    Win (Hu)
    
    Total Action Space Size: 42
    """
    
    SIZE = 42
    
    # Action Indices
    ACT_SKIP = 34
    ACT_CHOW_LOW = 35    # Uses X-2, X-1 with discarded X
    ACT_CHOW_MID = 36    # Uses X-1, X+1 with discarded X
    ACT_CHOW_HIGH = 37   # Uses X+1, X+2 with discarded X
    ACT_PONG = 38
    ACT_KONG = 39
    ACT_SELF_KONG = 40
    ACT_WIN = 41

    @staticmethod
    def get_action_description(action_idx: int) -> str:
        if 0 <= action_idx <= 33:
            return f"Discard Tile {action_idx}"
        if action_idx == ActionSpace.ACT_SKIP:
            return "Skip"
        if action_idx == ActionSpace.ACT_CHOW_LOW:
            return "Chow (Low)"
        if action_idx == ActionSpace.ACT_CHOW_MID:
            return "Chow (Mid)"
        if action_idx == ActionSpace.ACT_CHOW_HIGH:
            return "Chow (High)"
        if action_idx == ActionSpace.ACT_PONG:
            return "Pong"
        if action_idx == ActionSpace.ACT_KONG:
            return "Kong"
        if action_idx == ActionSpace.ACT_SELF_KONG:
            return "Self-Kong"
        if action_idx == ActionSpace.ACT_WIN:
            return "Win"
        return "Unknown"

    @staticmethod
    def is_discard(action_idx: int) -> bool:
        return 0 <= action_idx <= 33

