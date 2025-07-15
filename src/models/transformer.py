"""
CyclePatch-Transformer model for battery performance prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math
from src.models.base import BatteryPredictionModel
from src.data.cyclepatch import CyclePatchFramework, CyclePatchConfig


class CPTransformer(BatteryPredictionModel):
    """CyclePatch-Transformer model combining CyclePatch tokenization with Transformer."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        patch_size: int = 10,
        patch_stride: int = 5,
        patch_embed_dim: int = 128,
        feedforward_dim: int = 1024,
        max_seq_length: int = 1000,
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
        
        # Projection layer to match transformer dimension
        self.input_projection = nn.Linear(patch_embed_dim, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Cross-attention for feature integration
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature fusion
        fusion_input_dim = hidden_dim * 2 + input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific transformer heads
        self.rul_head = TransformerTaskHead(hidden_dim // 2, num_heads=4, task='rul')
        self.soh_head = TransformerTaskHead(hidden_dim // 2, num_heads=4, task='soh')
        self.soc_head = TransformerTaskHead(hidden_dim // 2, num_heads=4, task='soc')
        self.capacity_head = TransformerTaskHead(hidden_dim // 2, num_heads=4, task='capacity')
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CP-Transformer model.
        
        Args:
            x: Input features of shape (batch_size, sequence_length, input_dim)
               or (batch_size, input_dim) for single time step
        
        Returns:
            Dictionary of predictions for each task
        """
        # Handle both sequence and single time step inputs
        if len(x.shape) == 2:
            return self._forward_single(x)
        
        batch_size = x.size(0)
        
        # Extract cycle sequence features
        cycle_features = x[:, :, :6]  # First 6 features are cycle data
        
        # Apply CyclePatch tokenization
        patch_embeddings = self.cyclepatch(cycle_features)
        
        # Project to transformer dimension
        patch_embeddings = self.input_projection(patch_embeddings)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(embeddings)
        
        # Extract CLS token representation
        cls_representation = transformer_out[:, 0]
        
        # Apply cross-attention with original features
        current_features = x[:, -1:, :]  # Last time step features
        current_proj = self.input_projection(
            torch.zeros(batch_size, 1, self.hidden_dim).to(x.device)
        )
        
        # Cross-attention between CLS and current features
        cross_attn_out, _ = self.cross_attention(
            current_proj,
            transformer_out[:, 1:],  # Skip CLS token
            transformer_out[:, 1:]
        )
        cross_attn_out = cross_attn_out.squeeze(1)
        
        # Get current features (last time step)
        current_features_flat = x[:, -1, :]
        
        # Fuse all representations
        fused = torch.cat([cls_representation, cross_attn_out, current_features_flat], dim=-1)
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
        batch_size = x.size(0)
        
        # Create dummy transformer input
        dummy_input = self.cls_token.expand(batch_size, -1, -1)
        dummy_input = self.positional_encoding(dummy_input)
        
        # Process through transformer
        transformer_out = self.transformer_encoder(dummy_input)
        cls_representation = transformer_out[:, 0]
        
        # Fuse with input features
        fused = torch.cat([
            cls_representation,
            torch.zeros(batch_size, self.hidden_dim).to(x.device),
            x
        ], dim=-1)
        features = self.fusion(fused)
        
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TransformerTaskHead(nn.Module):
    """Task-specific transformer head with self-attention."""
    
    def __init__(self, input_dim: int, num_heads: int = 4, task: str = 'rul'):
        super().__init__()
        self.task = task
        
        # Task-specific attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        # Output projection
        self.output = nn.Linear(input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with self-attention."""
        # Reshape for attention (add sequence dimension)
        x = x.unsqueeze(1)
        
        # Self-attention with residual
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Squeeze sequence dimension and project
        x = x.squeeze(1)
        return self.output(x)


class HierarchicalCPTransformer(CPTransformer):
    """Hierarchical CP-Transformer with multi-scale attention."""
    
    def __init__(self, scales: list = [5, 10, 20], **kwargs):
        super().__init__(**kwargs)
        self.scales = scales
        
        # Multi-scale patch encoders
        self.multi_scale_encoders = nn.ModuleList([
            CyclePatchFramework(
                CyclePatchConfig(
                    patch_size=scale,
                    stride=scale // 2,
                    embed_dim=self.cp_config.embed_dim,
                    features=self.cp_config.features
                )
            )
            for scale in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(
            len(scales) * self.cp_config.embed_dim,
            self.cp_config.embed_dim
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale processing."""
        if len(x.shape) == 2:
            return self._forward_single(x)
        
        batch_size = x.size(0)
        cycle_features = x[:, :, :6]
        
        # Extract multi-scale patches
        multi_scale_embeddings = []
        for encoder in self.multi_scale_encoders:
            embeddings = encoder(cycle_features)
            # Global average pooling
            pooled = embeddings.mean(dim=1)
            multi_scale_embeddings.append(pooled)
        
        # Fuse multi-scale representations
        fused_scales = torch.cat(multi_scale_embeddings, dim=-1)
        scale_representation = self.scale_fusion(fused_scales)
        
        # Continue with standard transformer processing
        patch_embeddings = self.cyclepatch(cycle_features)
        patch_embeddings = self.input_projection(patch_embeddings)
        
        # Add scale representation to CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + scale_representation.unsqueeze(1)
        
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)
        embeddings = self.positional_encoding(embeddings)
        
        transformer_out = self.transformer_encoder(embeddings)
        cls_representation = transformer_out[:, 0]
        
        # Cross-attention
        current_features = x[:, -1:, :]
        current_proj = self.input_projection(
            torch.zeros(batch_size, 1, self.hidden_dim).to(x.device)
        )
        
        cross_attn_out, _ = self.cross_attention(
            current_proj,
            transformer_out[:, 1:],
            transformer_out[:, 1:]
        )
        cross_attn_out = cross_attn_out.squeeze(1)
        
        current_features_flat = x[:, -1, :]
        
        fused = torch.cat([cls_representation, cross_attn_out, current_features_flat], dim=-1)
        features = self.fusion(fused)
        
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }