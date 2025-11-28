from enum import Enum


class TileSuit(Enum):
    CHARACTER = 'Characters'
    BAMBOO = 'Bamboo'
    DOT = 'Dots'
    HONOR = "Honor"
    FLOWER = "Flower"
    SEASON = "Season"


class WindValue(Enum):
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


class Tile:
    """
    Represents a Mahjong tile with a suit and a value.
    """

    def __init__(self, suit: TileSuit, value: (int | FlowerValue | SeasonValue | WindValue | DragonValue)):
        self.suit = suit
        self.value = value

    def __eq__(self, otherTile):
        return self.suit == otherTile.suit and self.value == otherTile.value

    def __repr__(self) -> str:
        if self.value == DragonValue.RED:
            return "RD"
        elif self.value == DragonValue.GREEN:
            return "GD"
        elif self.value == DragonValue.WHITE:
            return "WD"
        elif self.value == WindValue.EAST:
            return "EW"
        elif self.value == WindValue.SOUTH:
            return "SW"
        elif self.value == WindValue.WEST:
            return "WW"
        elif self.value == WindValue.NORTH:
            return "NW"
        elif self.value == FlowerValue.PLUM:
            return "1F"
        elif self.value == FlowerValue.ORCHID:
            return "2F"
        elif self.value == FlowerValue.CHRYSANTHEMUM:
            return "3F"
        elif self.value == FlowerValue.BAMBOO:
            return "4F"
        elif self.value == SeasonValue.SPRING:
            return "1S"
        elif self.value == SeasonValue.SUMMER:
            return "2S"
        elif self.value == SeasonValue.AUTUMN:
            return "3S"
        elif self.value == SeasonValue.WINTER:
            return "4S"
        elif self.suit == TileSuit.BAMBOO:
            return f"{self.value}B"
        elif self.suit == TileSuit.CHARACTER:
            return f"{self.value}C"
        elif self.suit == TileSuit.DOT:
            return f"{self.value}D"
