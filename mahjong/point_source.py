from enum import Enum


class PointType(Enum):
    """
    Represents different types of point-scoring conditions or special hands in Mahjong.
    """
    COMMON_HAND = "Common Hand"
    ALL_IN_TRIPLETS = "All in Triplets"
    SEVER_PAIRS = "Sever Pairs"
    MIXED_ONE_SUIT = "Mixed One Suit"
    ALL_ONE_SUIT = "All One Suit"
    ALL_HONOR_TILES = "All Honor Tiles"
    SMALL_DRAGONS = "Small Dragons"
    GREAT_DRAGONS = "Great Dragons"
    SMALL_WINDS = "Small Winds"
    GREAT_WINDS = "Great Winds"
    THIRTEEN_ORPHANS = "Thirteen Orphans"
    ALL_KONGS = "All Kongs"
    SELF_TRIPLETS = "Self Triplets"
    ORPHANS = "Orphans"
    NINE_GATES = "Nine Gates"
    SEAT_WIND = "Seat Wind"
    PREVAILING_WIND = "Prevailing Wind"
    RED_DRAGON = "Red Dragon"
    GREEN_DRAGON = "Green Dragon"
    WHITE_DRAGON = "White Dragon"
    MIXED_ORPHANS = "Mixed Orphans"
    SELF_DRAW = "Self Draw"
    WIN_FROM_WALL = "Win From Wall"
    ROBBING_KONG = "Robbing Kong"
    WIN_BY_LAST_CATCH = "Win By Last Catch"
    WIN_BY_KONG = "Win By Kong"
    WIN_BY_DOUBLE_KONG = "Win By Double Kong"
    HEAVENLY_HAND = "Heavenly Hand"
    EARTHLY_HAND = "Earthly Hand"
    NO_FLOWERS_OR_SEASONS = "No Flowers Or Seasons"
    FLOWER_OF_OWN_WIND = "Flower Of Own Wind"
    SEASON_OF_OWN_WIND = "Season Of Own Wind"
    ALL_FLOWERS = "All Flowers"
    ALL_SEASONS = "All Seasons"


class PointSource:
    """
    Represents a source of points in Mahjong, encapsulating a specific point type
    and its associated value.
    """

    def __init__(self, point_type: PointType, value: int):
        # Enum describing the specific scoring criteria this instance represents.
        self.point_type = point_type
        # Numeric fan/point value awarded when this source is present.
        self.value = value

    @staticmethod
    def common_hand() -> "PointSource":
        return PointSource(PointType.COMMON_HAND, 1)

    @staticmethod
    def all_in_triplets() -> "PointSource":
        return PointSource(PointType.ALL_IN_TRIPLETS, 3)

    @staticmethod
    def sever_pairs() -> "PointSource":
        return PointSource(PointType.SEVER_PAIRS, 4)

    @staticmethod
    def mixed_one_suit() -> "PointSource":
        return PointSource(PointType.MIXED_ONE_SUIT, 3)

    @staticmethod
    def all_one_suit() -> "PointSource":
        return PointSource(PointType.ALL_ONE_SUIT, 7)

    @staticmethod
    def all_honor_tiles() -> "PointSource":
        return PointSource(PointType.ALL_HONOR_TILES, 10)

    @staticmethod
    def small_dragons() -> "PointSource":
        return PointSource(PointType.SMALL_DRAGONS, 5)

    @staticmethod
    def great_dragons() -> "PointSource":
        return PointSource(PointType.GREAT_DRAGONS, 8)

    @staticmethod
    def small_winds() -> "PointSource":
        return PointSource(PointType.SMALL_WINDS, 6)

    @staticmethod
    def great_winds() -> "PointSource":
        return PointSource(PointType.GREAT_WINDS, 13)

    @staticmethod
    def thirteen_orphans() -> "PointSource":
        return PointSource(PointType.THIRTEEN_ORPHANS, 13)

    @staticmethod
    def all_kongs() -> "PointSource":
        return PointSource(PointType.ALL_KONGS, 13)

    @staticmethod
    def self_triplets() -> "PointSource":
        return PointSource(PointType.SELF_TRIPLETS, 10)

    @staticmethod
    def orphans() -> "PointSource":
        return PointSource(PointType.ORPHANS, 10)

    @staticmethod
    def nine_gates() -> "PointSource":
        return PointSource(PointType.NINE_GATES, 10)

    @staticmethod
    def seat_wind() -> "PointSource":
        return PointSource(PointType.SEAT_WIND, 1)

    @staticmethod
    def prevailing_wind() -> "PointSource":
        return PointSource(PointType.PREVAILING_WIND, 1)

    @staticmethod
    def red_dragon() -> "PointSource":
        return PointSource(PointType.RED_DRAGON, 1)

    @staticmethod
    def green_dragon() -> "PointSource":
        return PointSource(PointType.GREEN_DRAGON, 1)

    @staticmethod
    def white_dragon() -> "PointSource":
        return PointSource(PointType.WHITE_DRAGON, 1)

    @staticmethod
    def mixed_orphans() -> "PointSource":
        return PointSource(PointType.MIXED_ORPHANS, 1)

    @staticmethod
    def self_draw() -> "PointSource":
        return PointSource(PointType.SELF_DRAW, 1)

    @staticmethod
    def win_from_wall() -> "PointSource":
        return PointSource(PointType.WIN_FROM_WALL, 1)

    @staticmethod
    def robbing_kong() -> "PointSource":
        return PointSource(PointType.ROBBING_KONG, 1)

    @staticmethod
    def win_by_last_catch() -> "PointSource":
        return PointSource(PointType.WIN_BY_LAST_CATCH, 1)

    @staticmethod
    def win_by_kong() -> "PointSource":
        return PointSource(PointType.WIN_BY_KONG, 1)

    @staticmethod
    def win_by_double_kong() -> "PointSource":
        return PointSource(PointType.WIN_BY_DOUBLE_KONG, 8)

    @staticmethod
    def heavenly_hand() -> "PointSource":
        return PointSource(PointType.HEAVENLY_HAND, 13)

    @staticmethod
    def earthly_hand() -> "PointSource":
        return PointSource(PointType.EARTHLY_HAND, 13)

    @staticmethod
    def no_flowers_or_seasons() -> "PointSource":
        return PointSource(PointType.NO_FLOWERS_OR_SEASONS, 1)

    @staticmethod
    def flower_of_own_wind() -> "PointSource":
        return PointSource(PointType.FLOWER_OF_OWN_WIND, 1)

    @staticmethod
    def season_of_own_wind() -> "PointSource":
        return PointSource(PointType.SEASON_OF_OWN_WIND, 1)

    @staticmethod
    def all_flowers() -> "PointSource":
        return PointSource(PointType.ALL_FLOWERS, 2)

    @staticmethod
    def all_seasons() -> "PointSource":
        return PointSource(PointType.ALL_SEASONS, 2)

    def __repr__(self) -> str:
        return self.point_type.value
