"""
CyclePatch Framework for battery cycle data tokenization.
"""

import numpy as np
import torch
from torch import nn
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CyclePatchConfig:
    """Configuration for CyclePatch tokenization."""
    patch_size: int = 10  # Number of cycles per patch
    stride: int = 5  # Stride between patches
    embed_dim: int = 128  # Embedding dimension
    max_cycles: int = 2000  # Maximum number of cycles
    features: List[str] = None  # Features to include in patches
    
    def __post_init__(self):
        if self.features is None:
            self.features = ['capacity', 'voltage_mean', 'current_mean', 
                           'temperature_mean', 'soh', 'capacity_fade']


class CyclePatchTokenizer:
    """Tokenize battery cycle data into patches."""
    
    def __init__(self, config: CyclePatchConfig):
        self.config = config
        
    def create_patches(self, cycle_data: np.ndarray) -> np.ndarray:
        """
        Create patches from cycle data.
        
        Args:
            cycle_data: Shape (num_cycles, num_features)
            
        Returns:
            patches: Shape (num_patches, patch_size, num_features)
        """
        num_cycles, num_features = cycle_data.shape
        patches = []
        
        # Create overlapping patches
        for i in range(0, num_cycles - self.config.patch_size + 1, self.config.stride):
            patch = cycle_data[i:i + self.config.patch_size]
            patches.append(patch)
        
        if len(patches) == 0:
            # If sequence is too short, pad and create single patch
            padded = np.pad(cycle_data, 
                          ((0, self.config.patch_size - num_cycles), (0, 0)), 
                          mode='edge')
            patches.append(padded)
        
        return np.array(patches)
    
    def create_positional_encoding(self, num_patches: int) -> np.ndarray:
        """Create positional encoding for patches."""
        position = np.arange(num_patches)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.config.embed_dim, 2) * 
                         -(np.log(10000.0) / self.config.embed_dim))
        
        pos_encoding = np.zeros((num_patches, self.config.embed_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding


class CyclePatchEncoder(nn.Module):
    """Encode cycle patches into embeddings."""
    
    def __init__(self, config: CyclePatchConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        input_dim = config.patch_size * len(config.features)
        self.patch_embed = nn.Linear(input_dim, config.embed_dim)
        
        # Learnable cycle type embedding
        self.cycle_type_embed = nn.Embedding(3, config.embed_dim)  # charge, discharge, rest
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.max_cycles // config.stride, config.embed_dim)
        )
        
        # Normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, patches: torch.Tensor, cycle_types: Optional[torch.Tensor] = None):
        """
        Encode patches into embeddings.
        
        Args:
            patches: Shape (batch_size, num_patches, patch_size, num_features)
            cycle_types: Shape (batch_size, num_patches)
            
        Returns:
            embeddings: Shape (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, patch_size, num_features = patches.shape
        
        # Flatten patches
        patches_flat = patches.reshape(batch_size, num_patches, -1)
        
        # Embed patches
        patch_embeds = self.patch_embed(patches_flat)
        
        # Add cycle type embeddings if provided
        if cycle_types is not None:
            type_embeds = self.cycle_type_embed(cycle_types)
            patch_embeds = patch_embeds + type_embeds
        
        # Add positional embeddings
        patch_embeds = patch_embeds + self.pos_embed[:, :num_patches, :]
        
        # Normalize and dropout
        embeddings = self.dropout(self.norm(patch_embeds))
        
        return embeddings


class IntraCycleEncoder(nn.Module):
    """Encode intra-cycle patterns within patches."""
    
    def __init__(self, config: CyclePatchConfig):
        super().__init__()
        self.config = config
        
        # 1D CNN for intra-cycle patterns
        self.conv1 = nn.Conv1d(len(config.features), 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, config.embed_dim, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, patches: torch.Tensor):
        """
        Extract intra-cycle features from patches.
        
        Args:
            patches: Shape (batch_size, num_patches, patch_size, num_features)
            
        Returns:
            features: Shape (batch_size, num_patches, embed_dim)
        """
        batch_size, num_patches, patch_size, num_features = patches.shape
        
        # Reshape for CNN
        x = patches.reshape(-1, num_features, patch_size)
        x = x.transpose(1, 2)  # (batch*patches, features, patch_size)
        
        # Apply convolutions
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.activation(self.conv2(x)))
        x = self.conv3(x)
        
        # Global pooling
        x = self.pool(x).squeeze(-1)
        
        # Reshape back
        features = x.reshape(batch_size, num_patches, -1)
        
        return features


class CyclePatchFramework(nn.Module):
    """Complete CyclePatch framework combining tokenization and encoding."""
    
    def __init__(self, config: CyclePatchConfig):
        super().__init__()
        self.config = config
        self.tokenizer = CyclePatchTokenizer(config)
        self.patch_encoder = CyclePatchEncoder(config)
        self.intra_cycle_encoder = IntraCycleEncoder(config)
        
        # Fusion layer
        self.fusion = nn.Linear(config.embed_dim * 2, config.embed_dim)
        self.fusion_norm = nn.LayerNorm(config.embed_dim)
        
    def forward(self, cycle_data: torch.Tensor, cycle_types: Optional[torch.Tensor] = None):
        """
        Process cycle data through CyclePatch framework.
        
        Args:
            cycle_data: Shape (batch_size, num_cycles, num_features)
            cycle_types: Optional shape (batch_size, num_cycles)
            
        Returns:
            embeddings: Shape (batch_size, num_patches, embed_dim)
        """
        batch_size = cycle_data.shape[0]
        
        # Create patches for each battery in batch
        all_patches = []
        for i in range(batch_size):
            patches = self.tokenizer.create_patches(cycle_data[i].cpu().numpy())
            all_patches.append(torch.from_numpy(patches))
        
        # Pad to same number of patches
        max_patches = max(p.shape[0] for p in all_patches)
        padded_patches = []
        for patches in all_patches:
            if patches.shape[0] < max_patches:
                pad_size = max_patches - patches.shape[0]
                padding = patches[-1:].repeat(pad_size, 1, 1)
                patches = torch.cat([patches, padding], dim=0)
            padded_patches.append(patches)
        
        patches_tensor = torch.stack(padded_patches).to(cycle_data.device)
        
        # Encode patches
        patch_embeds = self.patch_encoder(patches_tensor, cycle_types)
        intra_embeds = self.intra_cycle_encoder(patches_tensor)
        
        # Fuse embeddings
        fused = torch.cat([patch_embeds, intra_embeds], dim=-1)
        embeddings = self.fusion_norm(self.fusion(fused))
        
        return embeddings


def create_cyclepatch_features(cycle_data: np.ndarray, 
                              config: CyclePatchConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create CyclePatch features from raw cycle data.
    
    Args:
        cycle_data: Raw cycle data array
        config: CyclePatch configuration
        
    Returns:
        patches: Tokenized patches
        positions: Positional encodings
    """
    tokenizer = CyclePatchTokenizer(config)
    
    # Create patches
    patches = tokenizer.create_patches(cycle_data)
    
    # Create positional encoding
    positions = tokenizer.create_positional_encoding(len(patches))
    
    return patches, positions