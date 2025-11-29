import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MahjongFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Mahjong using 1D Convolutions.
    
    The observation space is (33, 34):
    - 33 Channels: Hand, Melds, Discards, Winds, etc.
    - 34 Tile Types: Man, Pin, Sou, Honors.
    
    We use 1D Convolutions with kernel_size=3 to efficiently detect:
    - Sequences (Chows): 3 consecutive tiles (e.g., 1-2-3 Man).
    - Triplets (Pongs): 3 identical tiles (captured by channel depth).
    """
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Observation shape is (33, 34)
        # Conv1d expects input of shape (Batch, Channels, Length)
        # So we treat 33 as channels and 34 as the sequence length.
        n_input_channels = observation_space.shape[0] # 33
        
        self.cnn = nn.Sequential(
            # Layer 1: Detect local patterns (sequences)
            nn.Conv1d(n_input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            # Layer 2: Combine low-level features
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            # Layer 3: Higher-level abstraction
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Flatten(),
        )
        
        # Compute the size of the output from the CNN to connect to the linear layer
        with th.no_grad():
            # Create a dummy input with batch_size=1
            # Shape: (1, 33, 34)
            sample_obs = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        # Final projection to features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

