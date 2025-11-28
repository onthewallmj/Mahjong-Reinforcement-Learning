from __future__ import annotations

from collections import Counter
from typing import Any

from .common import Wind
from .meld import Meld, MeldType
from .point_source import PointSource, PointType
from .tile import DragonValue, Tile, TileSuit, WindValue
from .win import WinCondition


class HandScorer:
    """
    Scores Mahjong hands to determine the PointSources they earn.
    """

    # Suits eligible for sequences/flushes (excludes honors/bonus suits).
    _SUITED_SUITS: set[TileSuit] = frozenset[TileSuit]({
        TileSuit.CHARACTER,
        TileSuit.BAMBOO,
        TileSuit.DOT,
    })

    # Faan types that represent mutually exclusive high-level hand shapes.
    _HAND_TYPE_POINT_TYPES: set[PointType] = frozenset[PointType]({
        PointType.ALL_IN_TRIPLETS,
        PointType.SEVER_PAIRS,
        PointType.MIXED_ONE_SUIT,
        PointType.ALL_ONE_SUIT,
        PointType.ALL_HONOR_TILES,
        PointType.SMALL_DRAGONS,
        PointType.GREAT_DRAGONS,
        PointType.SMALL_WINDS,
        PointType.GREAT_WINDS,
        PointType.THIRTEEN_ORPHANS,
        PointType.ALL_KONGS,
        PointType.SELF_TRIPLETS,
        PointType.ORPHANS,
        PointType.MIXED_ORPHANS,
        PointType.NINE_GATES,
    })

    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------

    @staticmethod
    def can_win(hand: list[Tile], melds: list[Meld]) -> bool:
        """
        Returns True if the hand consists of 4 valid melds + 1 pair or is a special hand.
        """
        tiles = HandScorer._collect_tiles(hand, melds)

        # Calculate expected tile count: 14 + number of Kongs
        # Each Kong adds 1 tile to the standard 14 (since a Kong has 4 tiles but counts as 1 set)
        num_kongs = sum(1 for m in melds if m.meld_type == MeldType.KONG)
        expected_count = 14 + num_kongs

        if len(tiles) != expected_count:
            return False

        # Check special hands first
        if HandScorer._is_thirteen_orphans(tiles):
            return True
        if HandScorer._is_seven_pairs(melds, tiles):
            return True
        if HandScorer._is_nine_gates(tiles):
            return True

        # Check standard hand (4 sets + 1 pair)
        # This requires a recursive backtracking algorithm or similar check.
        # Basic algorithm:
        # 1. Find a pair.
        # 2. Check if the rest can be formed into sets (triplets or sequences).
        return HandScorer._is_standard_hand(hand, melds)

    @classmethod
    def evaluate(
        cls,
        hand: list[Tile],
        melds: list[Meld],
        bonus_tiles: list[Tile],
        seat_wind: Wind,
        table_wind: Wind,
        win_condition: WinCondition,
        is_last_tile: bool,
        consecutive_kongs: int = 0,
    ) -> list[PointSource]:
        """
        Evaluates a Mahjong hand and returns a list of all point sources (fan) achieved.

        This is the main entry point for scoring. It aggregates points from multiple categories:
        - Bonus tiles (flowers/seasons)
        - Suit patterns (flushes, mixed one suit)
        - Shape patterns (all triplets, seven pairs, all kongs)
        - Dragon tile patterns
        - Wind tile patterns
        - Special orphan patterns (thirteen orphans, etc.)
        - Win condition specifics (self-draw, etc.)
        """
        tiles = cls._collect_tiles(hand, melds)

        points: list[PointSource] = []
        points.extend(cls._score_bonus_tiles(bonus_tiles, seat_wind))
        points.extend(cls._score_flushes(tiles))
        points.extend(cls._score_shapes(melds, tiles))
        points.extend(cls._score_dragons(tiles))
        points.extend(cls._score_winds(tiles, seat_wind, table_wind))
        points.extend(cls._score_orphan_patterns(tiles))

        # Determine if special hands (Seven Pairs or Thirteen Orphans) are present
        points.extend(cls._score_win_conditions(
            win_condition=win_condition,
            has_no_melds=len(melds) == 0,
            is_last_tile=is_last_tile,
            consecutive_kongs=consecutive_kongs,
            is_ineligible_for_win_from_wall=cls._is_ineligible_for_win_from_wall(
                melds, tiles)
        ))
        return points

    # -------------------------------------------------------------------------
    # High-Level Helper
    # -------------------------------------------------------------------------

    @classmethod
    def _is_ineligible_for_win_from_wall(cls, melds: list[Meld], tiles: list[Tile]) -> bool:
        """
        Checks if the hand matches any pattern that typically excludes standard "Win from Wall" points.
        """
        return (
            cls._is_seven_pairs(melds, tiles) or
            cls._is_thirteen_orphans(tiles) or
            cls._is_nine_gates(tiles) or
            cls._is_self_triplets(melds, tiles)
        )

    @staticmethod
    def _collect_tiles(hand: list[Tile], melds: list[Meld]) -> list[Tile]:
        """
        Combines tiles from the standing hand and any exposed melds into a single list.
        """
        tiles = list[Tile](hand)
        for meld in melds:
            tiles.extend(meld.tiles)
        return tiles

    # -------------------------------------------------------------------------
    # Scoring Logic (Grouped by category)
    # -------------------------------------------------------------------------

    @staticmethod
    def _score_bonus_tiles(bonus_tiles: list[Tile], seat_wind: Wind) -> list[PointSource]:
        """
        Scores points for Flowers and Seasons.
        """
        if not bonus_tiles:
            return [PointSource.no_flowers_or_seasons()]

        flower_points = HandScorer._score_flowers(bonus_tiles, seat_wind)
        season_points = HandScorer._score_seasons(bonus_tiles, seat_wind)

        return flower_points + season_points

    @staticmethod
    def _score_flowers(bonus_tiles: list[Tile], seat_wind: Wind) -> list[PointSource]:
        """
        Scores flower tiles based on owning the complete set or the flower matching the seat wind.
        """
        points: list[PointSource] = []
        flower_tiles_present = [
            tile for tile in bonus_tiles if tile.suit == TileSuit.FLOWER
        ]
        flower_values_present = {tile.value for tile in flower_tiles_present}

        if len(flower_values_present) == 4:
            points.append(PointSource.all_flowers())
        else:
            desired_flower_index = seat_wind.value
            for flower_value in flower_values_present:
                if getattr(flower_value, "value", None) == desired_flower_index:
                    points.append(PointSource.flower_of_own_wind())

        return points

    @staticmethod
    def _score_seasons(bonus_tiles: list[Tile], seat_wind: Wind) -> list[PointSource]:
        """
        Scores season tiles based on owning the complete set or the season matching the seat wind.
        """
        points: list[PointSource] = []
        season_tiles_present = [
            tile for tile in bonus_tiles if tile.suit == TileSuit.SEASON
        ]
        season_values_present = {tile.value for tile in season_tiles_present}

        if len(season_values_present) == 4:
            points.append(PointSource.all_seasons())
        else:
            desired_season_index = seat_wind.value
            for season_value in season_values_present:
                if getattr(season_value, "value", None) == desired_season_index:
                    points.append(PointSource.season_of_own_wind())

        return points

    @classmethod
    def _score_flushes(cls, tiles: list[Tile]) -> list[PointSource]:
        """
        Scores hands based on suit uniformity (e.g., All One Suit, Mixed One Suit, All Honors).
        """
        if not tiles:
            return []

        non_bonus_tiles = [tile for tile in tiles if not tile.is_bonus_tile()]
        if not non_bonus_tiles:
            return []

        if all(tile.is_honor() for tile in non_bonus_tiles):
            return [PointSource.all_honor_tiles()]

        suited_suits = {
            tile.suit for tile in non_bonus_tiles if tile.suit in cls._SUITED_SUITS
        }
        has_honor_tiles = any(tile.is_honor() for tile in non_bonus_tiles)

        if len(suited_suits) == 1:
            if has_honor_tiles:
                return [PointSource.mixed_one_suit()]
            return [PointSource.all_one_suit()]

        return []

    @classmethod
    def _score_shapes(cls, melds: list[Meld], tiles: list[Tile]) -> list[PointSource]:
        """
        Scores hands based on specific structural shapes like All Triplets, Seven Pairs, or All Kongs.
        """
        points: list[PointSource] = []

        if cls._has_all_kongs(melds):
            points.append(PointSource.all_kongs())
            return points

        if cls._is_self_triplets(melds, tiles):
            points.append(PointSource.self_triplets())
            return points

        if cls._is_all_triplets(melds, tiles):
            points.append(PointSource.all_in_triplets())

        if cls._is_seven_pairs(melds, tiles):
            points.append(PointSource.sever_pairs())

        return points

    @classmethod
    def _score_orphan_patterns(cls, tiles: list[Tile]) -> list[PointSource]:
        """
        Scores hands that consist of terminal and honor tiles (orphans).
        Includes Thirteen Orphans, Nine Gates, Pure Orphans, and Mixed Orphans.
        """
        if cls._is_thirteen_orphans(tiles):
            return [PointSource.thirteen_orphans()]
        if cls._is_nine_gates(tiles):
            return [PointSource.nine_gates()]
        if cls._is_pure_orphans(tiles):
            return [PointSource.orphans()]
        if cls._is_mixed_orphans_hand(tiles):
            return [PointSource.mixed_orphans()]
        return []

    @staticmethod
    def _score_dragons(tiles: list[Tile]) -> list[PointSource]:
        """
        Scores Dragon triplets (Pungs/Kongs).
        """
        counts = HandScorer._dragon_counts(tiles)
        pung_dragons = {dragon for dragon,
                        count in counts.items() if count >= 3}

        if len(pung_dragons) == 3:
            return [PointSource.great_dragons()]

        if len(pung_dragons) == 2:
            missing = ({DragonValue.RED, DragonValue.GREEN,
                        DragonValue.WHITE} - pung_dragons).pop()
            if counts.get(missing, 0) >= 2:
                return [PointSource.small_dragons()]

        points: list[PointSource] = []
        if DragonValue.RED in pung_dragons:
            points.append(PointSource.red_dragon())
        if DragonValue.GREEN in pung_dragons:
            points.append(PointSource.green_dragon())
        if DragonValue.WHITE in pung_dragons:
            points.append(PointSource.white_dragon())
        return points

    @staticmethod
    def _score_winds(tiles: list[Tile], seat_wind: Wind, table_wind: Wind) -> list[PointSource]:
        """
        Scores Wind triplets (Pungs/Kongs), accounting for seat wind and prevailing wind.
        """
        counts = HandScorer._wind_counts(tiles)
        pung_winds = {wind for wind, count in counts.items() if count >= 3}
        all_winds = {WindValue.EAST, WindValue.SOUTH,
                     WindValue.WEST, WindValue.NORTH}

        if len(pung_winds) == 4:
            return [PointSource.great_winds()]

        if len(pung_winds) == 3:
            missing = (all_winds - pung_winds).pop()
            if counts.get(missing, 0) >= 2:
                return [PointSource.small_winds()]

        points: list[PointSource] = []
        # Use values directly since WindValue is a class, not an Enum, or cast if needed.
        # Ideally, WindValue constants match Wind enum values.
        seat_value = seat_wind.value
        table_value = table_wind.value

        if counts.get(seat_value, 0) >= 3:
            points.append(PointSource.seat_wind())
        if counts.get(table_value, 0) >= 3:
            points.append(PointSource.prevailing_wind())

        return points

    @staticmethod
    def _score_win_conditions(win_condition: WinCondition, has_no_melds: bool, is_last_tile: bool, consecutive_kongs: int, is_ineligible_for_win_from_wall: bool) -> list[PointSource]:
        """
        Calculates points derived specifically from how the hand was won (e.g., self-draw).
        """
        points = []

        if win_condition == WinCondition.WIN_FROM_SELF_DRAW:
            points.append(PointSource.self_draw())

        # Win From Wall (Concealed Hand) is mutually exclusive with special hands like Seven Pairs or Thirteen Orphans
        if has_no_melds and not is_ineligible_for_win_from_wall:
            points.append(PointSource.win_from_wall())

        if is_last_tile:
            points.append(PointSource.win_by_last_catch())

        if consecutive_kongs == 1:
            points.append(PointSource.win_by_kong())
        elif consecutive_kongs == 2:
            points.append(PointSource.win_by_double_kong())

        return points

    # -------------------------------------------------------------------------
    # Pattern Checkers (Used by Scoring & can_win)
    # -------------------------------------------------------------------------

    @classmethod
    def _is_standard_hand(cls, hand: list[Tile], melds: list[Meld]) -> bool:
        """
        Checks if the hand and melds form a standard 4-set + 1-pair structure.
        Melds are assumed to be valid pre-formed sets.
        We need to check if the remaining 'hand' tiles can be formed into (4 - len(melds)) sets + 1 pair.

        Note on Kongs: A Kong (4 tiles) counts as 1 set. If a Kong is in 'melds',
        it contributes 1 to len(melds) but 4 to the total tile count.
        The remaining tiles in 'hand' must still form the remaining sets (3 tiles each) + 1 pair (2 tiles).
        Thus, the tile count of 'hand' (excluding bonus tiles) must be exactly:
        (sets_needed * 3) + 2.
        """
        # Filter out bonus tiles just in case, though they shouldn't be in the main hand list for structure check.
        non_bonus_hand = [t for t in hand if not t.is_bonus_tile()]

        # We need to form this many more sets
        sets_needed = 4 - len(melds)

        # Basic validation: Ensure we have the exact number of tiles needed for (sets + pair)
        if len(non_bonus_hand) != (sets_needed * 3) + 2:
            return False

        # Sort tiles to make finding sequences easier
        sorted_tiles = sorted(non_bonus_hand, key=lambda t: (
            t.suit.value, t.value.value if hasattr(t.value, 'value') else t.value))

        # Convert to a simpler representation for backtracking: Counter or list of integers per suit
        # Actually, Counter is good for triplets, sorted list good for sequences.
        # Let's use recursion on the sorted list.

        # Try to find the pair first
        counts = cls._tile_counts(sorted_tiles)
        possible_pairs = [tile for tile, count in counts.items() if count >= 2]

        for pair_tile_key in possible_pairs:
            # Create a copy of tiles without the pair
            remaining_tiles = list(sorted_tiles)
            # Remove 2 of the pair tile
            pair_suit, pair_val = pair_tile_key
            removed_count = 0
            for i in range(len(remaining_tiles) - 1, -1, -1):
                if remaining_tiles[i].suit == pair_suit and remaining_tiles[i].value == pair_val:
                    remaining_tiles.pop(i)
                    removed_count += 1
                    if removed_count == 2:
                        break

            if cls._can_form_sets(remaining_tiles, sets_needed):
                return True

        return False

    @staticmethod
    def _can_form_sets(tiles: list[Tile], sets_needed: int) -> bool:
        """
        Recursive backtracking to check if 'tiles' can be partitioned into 'sets_needed' valid sets (triplets or sequences).
        'tiles' is expected to be sorted.
        """
        if sets_needed == 0:
            return len(tiles) == 0

        if not tiles:
            return False

        first = tiles[0]

        # Try to form a Triplet (Pung)
        # Check if we have at least 3 of 'first'
        # Since tiles is sorted, they should be adjacent
        if len(tiles) >= 3 and tiles[1] == first and tiles[2] == first:
            if HandScorer._can_form_sets(tiles[3:], sets_needed - 1):
                return True

        # Try to form a Sequence (Chow)
        # Only for suited tiles (Character, Bamboo, Dot)
        if isinstance(first.value, int) and first.suit in [TileSuit.CHARACTER, TileSuit.BAMBOO, TileSuit.DOT]:
            # Look for first.value + 1 and first.value + 2 in the same suit
            val = first.value
            second_idx = -1
            third_idx = -1

            for i in range(1, len(tiles)):
                if tiles[i].suit == first.suit:
                    if second_idx == -1 and tiles[i].value == val + 1:
                        second_idx = i
                    elif third_idx == -1 and tiles[i].value == val + 2:
                        third_idx = i
                        break  # Found both

            if second_idx != -1 and third_idx != -1:
                # Create new list removing these 3 tiles
                next_tiles = list(tiles)
                # Remove in reverse index order to keep indices valid
                next_tiles.pop(third_idx)
                next_tiles.pop(second_idx)
                next_tiles.pop(0)

                if HandScorer._can_form_sets(next_tiles, sets_needed - 1):
                    return True

        return False

    @staticmethod
    def _is_thirteen_orphans(tiles: list[Tile]) -> bool:
        """
        Determines if the hand forms the Thirteen Orphans special pattern.
        """
        if len(tiles) != 14:
            return False

        required_keys = {
            (TileSuit.CHARACTER, 1),
            (TileSuit.CHARACTER, 9),
            (TileSuit.BAMBOO, 1),
            (TileSuit.BAMBOO, 9),
            (TileSuit.DOT, 1),
            (TileSuit.DOT, 9),
            (TileSuit.HONOR, WindValue.EAST),
            (TileSuit.HONOR, WindValue.SOUTH),
            (TileSuit.HONOR, WindValue.WEST),
            (TileSuit.HONOR, WindValue.NORTH),
            (TileSuit.HONOR, DragonValue.RED),
            (TileSuit.HONOR, DragonValue.GREEN),
            (TileSuit.HONOR, DragonValue.WHITE),
        }

        counts = HandScorer._tile_counts(tiles)
        if not required_keys.issubset(counts.keys()):
            return False

        duplicates = sum(count - 1 for count in counts.values())
        return duplicates == 1 and len(counts) == len(required_keys)

    @staticmethod
    def _is_seven_pairs(melds: list[Meld], tiles: list[Tile]) -> bool:
        """
        Determines if the hand forms the Seven Pairs special pattern.
        """
        if melds or len(tiles) != 14:
            return False
        counts = Counter[tuple[TileSuit, object]](
            HandScorer._tile_key(tile) for tile in tiles)
        return len(counts) == 7 and all(count == 2 for count in counts.values())

    @classmethod
    def _is_nine_gates(cls, tiles: list[Tile]) -> bool:
        """
        Determines if the hand forms the Nine Gates special pattern.
        """
        if len(tiles) != 14:
            return False
        if any(tile.is_bonus_tile() or tile.is_honor() for tile in tiles):
            return False

        suit_set = {
            tile.suit for tile in tiles if tile.suit in cls._SUITED_SUITS}
        if len(suit_set) != 1:
            return False

        suit = suit_set.pop()
        value_counts = Counter(
            tile.value for tile in tiles if tile.suit == suit)

        required = {1: 3, 9: 3}
        for value in range(2, 9):
            required[value] = 1

        return all(value_counts.get(value, 0) >= needed for value, needed in required.items())

    @staticmethod
    def _is_pure_orphans(tiles: list[Tile]) -> bool:
        """
        Determines if the hand consists exclusively of terminal tiles (1s and 9s).
        """
        if not tiles:
            return False
        if any(tile.is_bonus_tile() or tile.is_honor() for tile in tiles):
            return False

        suits = {tile.suit for tile in tiles}
        if len(suits) != 1:
            return False

        return all(tile.is_terminal() for tile in tiles)

    @staticmethod
    def _is_mixed_orphans_hand(tiles: list[Tile]) -> bool:
        """
        Determines if the hand consists of a mix of terminal tiles and honor tiles.
        """
        if not tiles:
            return False
        if any(tile.is_bonus_tile() for tile in tiles):
            return False
        return all(tile.is_mixed_orphan() for tile in tiles)

    @staticmethod
    def _has_all_kongs(melds: list[Meld]) -> bool:
        """
        Determines if the hand consists of four Kongs.
        """
        return len([meld for meld in melds if meld.meld_type == MeldType.KONG]) == 4

    @classmethod
    def _is_self_triplets(cls, melds: list[Meld], tiles: list[Tile]) -> bool:
        """
        Determines if the hand qualifies as Self Triplets (Concealed Triplets).
        """
        return not melds and cls._is_all_triplets(melds, tiles)

    @classmethod
    def _is_all_triplets(cls, melds: list[Meld], tiles: list[Tile]) -> bool:
        """
        Determines if the hand consists entirely of triplets (Pungs/Kongs) and one pair.
        """
        if not tiles:
            return False

        if any(meld.meld_type == MeldType.CHOW for meld in melds):
            return False

        counts = cls._tile_counts(tiles)
        pair_counts = sum(1 for count in counts.values() if count == 2)
        if pair_counts != 1:
            return False

        if any(count == 1 for count in counts.values()):
            return False

        triplet_units = sum(count // 3 for count in counts.values())
        return triplet_units == 4

    # -------------------------------------------------------------------------
    # Low-Level Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _tile_key(tile: Tile) -> tuple[TileSuit, object]:
        """
        Generates a consistent key for counting tiles based on suit and value.
        """
        return tile.suit, tile.value

    @classmethod
    def _tile_counts(cls, tiles: list[Tile]) -> Counter:
        """
        Counts the occurrences of each unique tile (by suit and value) in the provided list.
        """
        return Counter(cls._tile_key(tile) for tile in tiles)

    @staticmethod
    def _dragon_counts(tiles: list[Tile]) -> Counter:
        """
        Counts the occurrences of each dragon tile value in the provided list.
        """
        counts = Counter()
        for tile in tiles:
            if tile.suit == TileSuit.HONOR and isinstance(tile.value, DragonValue):
                counts[tile.value] += 1
        return counts

    @staticmethod
    def _wind_counts(tiles: list[Tile]) -> Counter:
        """
        Counts the occurrences of each wind tile value in the provided list.
        """
        counts = Counter()
        for tile in tiles:
            if tile.suit == TileSuit.HONOR and isinstance(tile.value, WindValue):
                counts[tile.value] += 1
        return counts

    @classmethod
    def _has_hand_type_point(cls, points: list[PointSource]) -> bool:
        """
        Checks if the current list of points includes any major hand shape types.
        """
        return any(point.point_type in cls._HAND_TYPE_POINT_TYPES for point in points)
