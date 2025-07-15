"""
CyclePatch-GRU model for battery performance prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from src.models.base import BatteryPredictionModel
from src.data.cyclepatch import CyclePatchFramework, CyclePatchConfig


class CPGRU(BatteryPredictionModel):
    """CyclePatch-GRU model combining CyclePatch tokenization with GRU."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        patch_size: int = 10,
        patch_stride: int = 5,
        patch_embed_dim: int = 128,
        bidirectional: bool = True,
        **kwargs
    ):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
        
        # CyclePatch configuration
        self.cp_config = CyclePatchConfig(
            patch_size=patch_size,
            stride=patch_stride,
            embed_dim=patch_embed_dim,
            features=['capacity', 'voltage_mean', 'current_mean', 
                     'temperature_mean', 'soh', 'capacity_fade']
        )
        
        # CyclePatch framework
        self.cyclepatch = CyclePatchFramework(self.cp_config)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=patch_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Adjust hidden dim for bidirectional
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=gru_output_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(gru_output_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.rul_head = self._create_task_head(hidden_dim // 2, 'rul')
        self.soh_head = self._create_task_head(hidden_dim // 2, 'soh')
        self.soc_head = self._create_task_head(hidden_dim // 2, 'soc')
        self.capacity_head = self._create_task_head(hidden_dim // 2, 'capacity')
        
    def _create_task_head(self, input_dim: int, task: str) -> nn.Module:
        """Create task-specific prediction head."""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CP-GRU model.
        
        Args:
            x: Input features of shape (batch_size, sequence_length, input_dim)
               or (batch_size, input_dim) for single time step
        
        Returns:
            Dictionary of predictions for each task
        """
        # Handle both sequence and single time step inputs
        if len(x.shape) == 2:
            # Single time step - use only feature-based prediction
            return self._forward_single(x)
        
        # Extract cycle sequence features
        cycle_features = x[:, :, :6]  # First 6 features are cycle data
        
        # Apply CyclePatch tokenization
        patch_embeddings = self.cyclepatch(cycle_features)
        
        # Process through GRU
        gru_out, _ = self.gru(patch_embeddings)
        
        # Apply attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Global pooling over sequence
        if hasattr(self, 'pooling_method') and self.pooling_method == 'attention':
            pooled = self._attention_pooling(attn_out)
        else:
            # Default: take last hidden state
            pooled = attn_out[:, -1, :]
        
        # Get current features (last time step)
        current_features = x[:, -1, :]
        
        # Fuse temporal and current features
        fused = torch.cat([pooled, current_features], dim=-1)
        features = self.fusion(fused)
        
        # Task-specific predictions
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions
    
    def _forward_single(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for single time step input."""
        # Use a simple MLP for single time step
        features = self.fusion(torch.cat([
            torch.zeros(x.size(0), self.hidden_dim * 2).to(x.device),
            x
        ], dim=-1))
        
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions
    
    def _attention_pooling(self, sequence: torch.Tensor) -> torch.Tensor:
        """Attention-based pooling over sequence."""
        # Compute attention weights
        attention_weights = torch.softmax(
            self.attention_pool(sequence).squeeze(-1), dim=1
        )
        
        # Weighted sum
        pooled = torch.sum(
            sequence * attention_weights.unsqueeze(-1), dim=1
        )
        
        return pooled


class EnhancedCPGRU(CPGRU):
    """Enhanced CP-GRU with additional features."""
    
    def __init__(self, use_residual: bool = True, use_layer_norm: bool = True, **kwargs):
        super().__init__(**kwargs)
        
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim * 2)
        
        # Residual connections
        if use_residual:
            self.residual_proj = nn.Linear(self.cp_config.embed_dim, self.hidden_dim * 2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with residual connections and layer norm."""
        if len(x.shape) == 2:
            return self._forward_single(x)
        
        # Extract cycle sequence features
        cycle_features = x[:, :, :6]
        
        # Apply CyclePatch tokenization
        patch_embeddings = self.cyclepatch(cycle_features)
        
        # Process through GRU with residual
        gru_out, _ = self.gru(patch_embeddings)
        
        if self.use_residual:
            # Project patch embeddings to match GRU output dim
            residual = self.residual_proj(patch_embeddings)
            gru_out = gru_out + residual
        
        if self.use_layer_norm:
            gru_out = self.layer_norm(gru_out)
        
        # Apply attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Global pooling
        pooled = attn_out[:, -1, :]
        
        # Get current features
        current_features = x[:, -1, :]
        
        # Fuse features
        fused = torch.cat([pooled, current_features], dim=-1)
        features = self.fusion(fused)
        
        # Task-specific predictions
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions