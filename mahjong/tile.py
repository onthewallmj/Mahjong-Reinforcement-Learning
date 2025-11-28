from dataclasses import dataclass
from enum import Enum


class TileSuit(Enum):
    CHARACTER = 'Characters'
    BAMBOO = 'Bamboo'
    DOT = 'Dots'
    HONOR = "Honor"
    FLOWER = "Flower"
    SEASON = "Season"


class WindValue:
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


class DragonValue(Enum):
    RED = 0
    GREEN = 1
    WHITE = 2


class FlowerValue(Enum):
    PLUM = 0
    ORCHID = 1
    CHRYSANTHEMUM = 2
    BAMBOO = 3


class SeasonValue(Enum):
    SPRING = 0
    SUMMER = 1
    AUTUMN = 2
    WINTER = 3


@dataclass(frozen=True)
class Tile:
    """
    Represents a Mahjong tile with a suit and a value.
    """
    suit: TileSuit
    value: int | FlowerValue | SeasonValue | WindValue | DragonValue

    def is_honor(self) -> bool:
        return self.suit == TileSuit.HONOR

    def is_terminal(self) -> bool:
        return not self.is_bonus_tile() and self.value in [1, 9]

    def is_bonus_tile(self) -> bool:
        return self.suit in [TileSuit.FLOWER, TileSuit.SEASON]

    def is_mixed_orphan(self) -> bool:
        """
        Checks if the tile is a 'mixed orphan' tile.
        These are terminal tiles (1s and 9s) or honor tiles (winds and dragons).
        """
        if self.is_bonus_tile():
            return False

        if self.is_honor():
            return True

        if self.value in [1, 9]:
            return True

        return False

    def get_index_34(self) -> int:
        """
        Returns a unique index (0-33) for the tile type.
        Suitable for vector representations (34 unique tiles).
        Returns -1 for bonus tiles.
        """
        if self.is_bonus_tile():
            return -1

        if self.suit == TileSuit.CHARACTER:
            return int(self.value) - 1
        
        if self.suit == TileSuit.BAMBOO:
            return int(self.value) - 1 + 9
        
        if self.suit == TileSuit.DOT:
            return int(self.value) - 1 + 18
        
        if self.suit == TileSuit.HONOR:
            if isinstance(self.value, int): # WindValue constant
                return 27 + self.value
            if isinstance(self.value, DragonValue):
                return 31 + self.value.value
            # If value is already mapped or unknown
            pass
            
        raise ValueError(f"Cannot map tile {self} to index 34")

    def __eq__(self, otherTile):
        """
        Checks equality between this tile and another.
        Two tiles are considered equal if they share the same suit and value.
        """
        if not isinstance(otherTile, Tile):
            return False
        return self.suit == otherTile.suit and self.value == otherTile.value

    def __repr__(self) -> str:
        honor_repr = {
            DragonValue.RED: "RD",
            DragonValue.GREEN: "GD",
            DragonValue.WHITE: "WD",
            WindValue.EAST: "EW",
            WindValue.SOUTH: "SW",
            WindValue.WEST: "WW",
            WindValue.NORTH: "NW",
            FlowerValue.PLUM: "1F",
            FlowerValue.ORCHID: "2F",
            FlowerValue.CHRYSANTHEMUM: "3F",
            FlowerValue.BAMBOO: "4F",
            SeasonValue.SPRING: "1S",
            SeasonValue.SUMMER: "2S",
            SeasonValue.AUTUMN: "3S",
            SeasonValue.WINTER: "4S",
        }

        if self.value in honor_repr:
            return honor_repr[self.value]

        suit_suffix = {
            TileSuit.BAMBOO: "B",
            TileSuit.CHARACTER: "C",
            TileSuit.DOT: "D",
        }.get(self.suit)

        if suit_suffix:
            return f"{self.value}{suit_suffix}"

        return f"{self.suit.name}:{self.value}"
