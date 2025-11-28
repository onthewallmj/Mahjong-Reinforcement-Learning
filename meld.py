from enum import Enum

from tile import Tile


class MeldType(Enum):
    CHOW = 0
    PONG = 1
    KONG = 2


class Meld:
    """
    Represents a meld (set, sequence, or quadruplet) in Mahjong.
    """

    def __init__(self, tiles: list[Tile], meld_type: MeldType):
        self.meld_type = meld_type
        self.tiles = tiles

    @property
    def first_tile(self) -> Tile:
        return self.tiles[0]

    @property
    def second_tile(self) -> Tile:
        return self.tiles[1]

    @property
    def third_tile(self) -> Tile:
        return self.tiles[2]

    @property
    def fourth_tile(self) -> Tile:
        return self.tiles[3]

    @staticmethod
    def create_chow(tile1: Tile, tile2: Tile, tile3: Tile) -> 'Meld':
        """
        Creates a Chow meld from three sequential tiles of the same suit.
        """
        return Meld([tile1, tile2, tile3], MeldType.CHOW)

    @staticmethod
    def create_pong(tile1: Tile, tile2: Tile, tile3: Tile) -> 'Meld':
        """
        Creates a Pong meld from three identical tiles.
        """
        if tile1 != tile2 or tile1 != tile3:
            raise ValueError("Tiles for a pong meld must be identical.")
        return Meld([tile1, tile2, tile3], MeldType.PONG)

    @staticmethod
    def create_kong(tile1: Tile, tile2: Tile, tile3: Tile, tile4: Tile) -> 'Meld':
        """
        Creates a Kong meld from four identical tiles.
        """
        if tile1 != tile2 or tile1 != tile3 or tile1 != tile4:
            raise ValueError("Tiles for a kong meld must be identical.")
        return Meld([tile1, tile2, tile3, tile4], MeldType.KONG)
