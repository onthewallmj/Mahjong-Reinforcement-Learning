from enum import Enum


class MeldType(Enum):
    CHOW = 0
    PONG = 1
    KONG = 2


class Meld:
    """
    Represents a meld (set, sequence, or quadruplet) in Mahjong.
    """

    def __init__(self, tiles: list[str], meld_type: MeldType):
        self.meld_type = meld_type
        self.tiles = tiles

    @staticmethod
    def create_chow(tile1: str, tile2: str, tile3: str) -> 'Meld':
        """
        Creates a Chow meld from three sequential tiles of the same suit.
        """
        return Meld([tile1, tile2, tile3], MeldType.CHOW)

    @staticmethod
    def create_pong(tile1: str, tile2: str, tile3: str) -> 'Meld':
        """
        Creates a Pong meld from three identical tiles.
        """
        if not (tile1 == tile2 == tile3):
            raise ValueError("Tiles for a pong meld must be identical.")
        return Meld([tile1, tile2, tile3], MeldType.PONG)

    @staticmethod
    def create_kong(tile1: str, tile2: str, tile3: str, tile4: str) -> 'Meld':
        """
        Creates a Kong meld from four identical tiles.
        """
        if not (tile1 == tile2 == tile3 == tile4):
            raise ValueError("Tiles for a kong meld must be identical.")
        return Meld([tile1, tile2, tile3, tile4], MeldType.KONG)
