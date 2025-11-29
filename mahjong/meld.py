from enum import Enum

from .tile import Tile


class MeldType(Enum):
    CHOW = "Chow"
    PONG = "Pong"
    KONG = "Kong"


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
        # Validate that all tiles are suited (not honors/bonus) and of the same suit
        tiles = [tile1, tile2, tile3]
        suits = {t.suit for t in tiles}
        if len(suits) != 1:
            raise ValueError("Chow tiles must all have the same suit.")

        # Chow is only allowed for numbered suits (Characters, Bamboo, Dots)
        from .tile import TileSuit  # local import to avoid circulars at module import time
        suit = suits.pop()
        if suit not in {TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT}:
            raise ValueError("Cannot create Chow from honor or bonus tiles.")

        # All values must be integers and consecutive (e.g. 4,5,6)
        try:
            values = sorted(int(t.value) for t in tiles)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Chow tiles must have integer values.") from exc

        if not (values[0] + 1 == values[1] and values[1] + 1 == values[2]):
            raise ValueError("Chow tiles must form a consecutive sequence.")

        return Meld(tiles, MeldType.CHOW)

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

    def __repr__(self) -> str:
        tile_strs = [str(tile) for tile in self.tiles]
        return f"{self.meld_type.value}({' '.join(tile_strs)})"
