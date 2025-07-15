"""
CyclePatch-LSTM model for battery performance prediction.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from src.models.base import BatteryPredictionModel
from src.data.cyclepatch import CyclePatchFramework, CyclePatchConfig


class CPLSTM(BatteryPredictionModel):
    """CyclePatch-LSTM model combining CyclePatch tokenization with LSTM."""
    
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
        use_attention: bool = True,
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
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=patch_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Adjust hidden dim for bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Optional attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = BahdanauAttention(lstm_output_dim)
        
        # Temporal convolution for local patterns
        self.temporal_conv = nn.Conv1d(
            in_channels=lstm_output_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1
        )
        
        # Feature fusion
        fusion_input_dim = hidden_dim + lstm_output_dim + input_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads with skip connections
        self.rul_head = ResidualTaskHead(hidden_dim // 2, 'rul')
        self.soh_head = ResidualTaskHead(hidden_dim // 2, 'soh')
        self.soc_head = ResidualTaskHead(hidden_dim // 2, 'soc')
        self.capacity_head = ResidualTaskHead(hidden_dim // 2, 'capacity')
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CP-LSTM model.
        
        Args:
            x: Input features of shape (batch_size, sequence_length, input_dim)
               or (batch_size, input_dim) for single time step
        
        Returns:
            Dictionary of predictions for each task
        """
        # Handle both sequence and single time step inputs
        if len(x.shape) == 2:
            return self._forward_single(x)
        
        # Extract cycle sequence features
        cycle_features = x[:, :, :6]  # First 6 features are cycle data
        
        # Apply CyclePatch tokenization
        patch_embeddings = self.cyclepatch(cycle_features)
        
        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(patch_embeddings)
        
        # Apply temporal convolution
        conv_out = self.temporal_conv(lstm_out.transpose(1, 2)).transpose(1, 2)
        conv_out = torch.relu(conv_out)
        
        # Apply attention if enabled
        if self.use_attention:
            # Use last hidden state as query
            if self.lstm.bidirectional:
                query = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                query = h_n[-1]
            context, attention_weights = self.attention(query, lstm_out)
        else:
            # Use last output
            context = lstm_out[:, -1, :]
        
        # Global max pooling on convolution output
        conv_pooled = torch.max(conv_out, dim=1)[0]
        
        # Get current features (last time step)
        current_features = x[:, -1, :]
        
        # Fuse all features
        fused = torch.cat([context, conv_pooled, current_features], dim=-1)
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
        # Initialize hidden states
        h_0 = torch.zeros(
            self.lstm.num_layers * (2 if self.lstm.bidirectional else 1),
            x.size(0),
            self.hidden_dim
        ).to(x.device)
        c_0 = torch.zeros_like(h_0)
        
        # Create dummy sequence
        dummy_features = torch.zeros(x.size(0), 1, self.cp_config.embed_dim).to(x.device)
        
        # Process through LSTM
        _, (h_n, c_n) = self.lstm(dummy_features, (h_0, c_0))
        
        # Get final hidden state
        if self.lstm.bidirectional:
            context = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            context = h_n[-1]
        
        # Fuse with input features
        fused = torch.cat([
            context,
            torch.zeros(x.size(0), self.hidden_dim).to(x.device),
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


class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / (hidden_size ** 0.5)
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Bahdanau attention.
        
        Args:
            query: Shape (batch_size, hidden_size)
            keys: Shape (batch_size, seq_len, hidden_size)
            
        Returns:
            context: Weighted sum of values (batch_size, hidden_size)
            attention_weights: Attention weights (batch_size, seq_len)
        """
        # Project query and keys
        q = self.W_q(query).unsqueeze(1)  # (batch_size, 1, hidden_size)
        k = self.W_k(keys)  # (batch_size, seq_len, hidden_size)
        v = self.W_v(keys)  # (batch_size, seq_len, hidden_size)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (batch_size, 1, seq_len)
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute weighted sum
        context = torch.bmm(attention_weights, v).squeeze(1)  # (batch_size, hidden_size)
        
        # Output projection
        context = self.W_o(context)
        
        return context, attention_weights.squeeze(1)


class ResidualTaskHead(nn.Module):
    """Task-specific head with residual connection."""
    
    def __init__(self, input_dim: int, task: str):
        super().__init__()
        self.task = task
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Skip connection
        self.skip = nn.Linear(input_dim, input_dim // 4)
        
        # Final projection
        self.output = nn.Linear(input_dim // 4, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = self.skip(x)
        features = self.layers(x) + residual
        return self.output(features)


class StackedCPLSTM(CPLSTM):
    """Stacked CP-LSTM with multiple LSTM blocks."""
    
    def __init__(self, num_blocks: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        
        # Additional LSTM blocks
        self.lstm_blocks = nn.ModuleList([
            nn.LSTM(
                input_size=self.hidden_dim * 2 if self.lstm.bidirectional else self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                dropout=kwargs.get('dropout', 0.2),
                batch_first=True,
                bidirectional=self.lstm.bidirectional
            )
            for _ in range(num_blocks - 1)
        ])
        
        # Block fusion layers
        self.block_fusion = nn.ModuleList([
            nn.Linear(self.hidden_dim * 4 if self.lstm.bidirectional else self.hidden_dim * 2, 
                     self.hidden_dim * 2 if self.lstm.bidirectional else self.hidden_dim)
            for _ in range(num_blocks - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through stacked LSTM blocks."""
        if len(x.shape) == 2:
            return self._forward_single(x)
        
        # Initial processing
        cycle_features = x[:, :, :6]
        patch_embeddings = self.cyclepatch(cycle_features)
        
        # First LSTM block
        lstm_out, (h_n, c_n) = self.lstm(patch_embeddings)
        
        # Additional LSTM blocks
        for i, (lstm_block, fusion) in enumerate(zip(self.lstm_blocks, self.block_fusion)):
            lstm_out_new, (h_n_new, c_n_new) = lstm_block(lstm_out)
            
            # Fuse with previous output
            lstm_out = fusion(torch.cat([lstm_out, lstm_out_new], dim=-1))
            h_n = h_n_new
            c_n = c_n_new
        
        # Continue with rest of forward pass
        conv_out = self.temporal_conv(lstm_out.transpose(1, 2)).transpose(1, 2)
        conv_out = torch.relu(conv_out)
        
        if self.use_attention:
            if self.lstm.bidirectional:
                query = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                query = h_n[-1]
            context, _ = self.attention(query, lstm_out)
        else:
            context = lstm_out[:, -1, :]
        
        conv_pooled = torch.max(conv_out, dim=1)[0]
        current_features = x[:, -1, :]
        
        fused = torch.cat([context, conv_pooled, current_features], dim=-1)
        features = self.fusion(fused)
        
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions