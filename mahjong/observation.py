import numpy as np
from .game import Game
from .player import Player
from .tile import Tile

class ObservationBuilder:
    """
    Builds the observation vector (tensor) for a specific player in the game.
    
    The observation is a multi-channel 2D matrix of shape (C, 34), where:
    - C is the number of feature channels (planes).
    - 34 is the number of unique tile types.
    
    Channels mapping (Total 33 channels):
    0-3:   My Hand (Hidden) - Counts [>=1, >=2, >=3, ==4]
    4-7:   My Melds (Exposed) - Counts [>=1, >=2, >=3, ==4]
    8-11:  Player+1 Melds - Counts [>=1, >=2, >=3, ==4]
    12-15: Player+2 Melds - Counts [>=1, >=2, >=3, ==4]
    16-19: Player+3 Melds - Counts [>=1, >=2, >=3, ==4]
    20-23: Discards (Global) - Counts [>=1, >=2, >=3, ==4]
    24-27: My Seat Wind (One-hot: East, South, West, North)
    28-31: Table Wind (One-hot: East, South, West, North)
    32:    Am I Dealer? (All 1s if true, else 0s)
    """

    def __init__(self):
        self.num_channels = 33
        self.num_tiles = 34

    def build_observation(self, game: Game, player_index: int) -> np.ndarray:
        """
        Constructs the observation tensor for the player at 'player_index'.
        Returns a float32 numpy array of shape (33, 34).
        """
        obs = np.zeros((self.num_channels, self.num_tiles), dtype=np.float32)
        
        me = game.players[player_index]
        
        # 1. Encode My Hand (Hidden)
        # We count tiles in hand (excluding melds, which are stored in me.melds)
        hand_counts = self._count_tiles(me.hand)
        self._encode_counts(obs, 0, hand_counts)
        
        # 2. Encode My Melds
        my_meld_tiles = self._get_meld_tiles(me)
        my_meld_counts = self._count_tiles(my_meld_tiles)
        self._encode_counts(obs, 4, my_meld_counts)
        
        # 3. Encode Other Players' Melds (Relative order: Right, Across, Left)
        for i in range(1, 4):
            relative_idx = (player_index + i) % 4
            other_player = game.players[relative_idx]
            other_meld_tiles = self._get_meld_tiles(other_player)
            other_meld_counts = self._count_tiles(other_meld_tiles)
            
            start_channel = 8 + (i - 1) * 4
            self._encode_counts(obs, start_channel, other_meld_counts)
            
        # 4. Encode Discards (Global for now)
        # In a more advanced version, we might want discard history or per-player discards.
        discard_counts = self._count_tiles(game.discards)
        self._encode_counts(obs, 20, discard_counts)
        
        # 5. Encode My Seat Wind
        # seat_index is fixed 0-3. Seat Wind depends on dealer.
        # However, PlayerGameState stores 'seatWind' (common.Wind).
        # We assume standard order: East=0, South=1, West=2, North=3.
        my_seat_wind_val = me.gameState.seatWind.value
        obs[24 + my_seat_wind_val, :] = 1.0
        
        # 6. Encode Table Wind
        table_wind_val = game.table_wind.value
        obs[28 + table_wind_val, :] = 1.0
        
        # 7. Encode Dealer Status
        if game.dealer_index == player_index:
            obs[32, :] = 1.0
            
        return obs

    def _count_tiles(self, tiles: list[Tile]) -> np.ndarray:
        """
        Returns an array of shape (34,) with counts of each tile type.
        """
        counts = np.zeros(self.num_tiles, dtype=int)
        for tile in tiles:
            if tile.is_bonus_tile():
                continue
            idx = tile.get_index_34()
            counts[idx] += 1
        return counts

    def _encode_counts(self, obs: np.ndarray, start_channel: int, counts: np.ndarray):
        """
        Encodes counts into 4 channels [>=1, >=2, >=3, ==4].
        """
        obs[start_channel + 0, :] = (counts >= 1).astype(np.float32)
        obs[start_channel + 1, :] = (counts >= 2).astype(np.float32)
        obs[start_channel + 2, :] = (counts >= 3).astype(np.float32)
        obs[start_channel + 3, :] = (counts >= 4).astype(np.float32)

    def _get_meld_tiles(self, player: Player) -> list[Tile]:
        """
        Extracts all tiles involved in a player's melds.
        """
        tiles = []
        for meld in player.gameState.melds:
            tiles.extend(meld.tiles)
        return tiles

