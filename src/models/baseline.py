"""
Baseline models for battery performance prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from .base import BatteryPredictionModel


class LinearBaselineModel(BatteryPredictionModel):
    """Simple linear regression baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Linear layers for each task
        self.rul_model = nn.Linear(self.input_dim, 1)
        self.soh_model = nn.Linear(self.input_dim, 1)
        self.soc_model = nn.Linear(self.input_dim, 1)
        self.capacity_model = nn.Linear(self.input_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through linear models."""
        # Handle sequence input by taking the last time step
        if len(x.shape) == 3:
            x = x[:, -1, :]
        
        predictions = {
            'rul': self.rul_model(x).squeeze(-1),
            'soh': torch.sigmoid(self.soh_model(x)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_model(x)).squeeze(-1),
            'capacity': self.capacity_model(x).squeeze(-1)
        }
        
        return predictions


class MLPBaselineModel(BatteryPredictionModel):
    """Multi-Layer Perceptron baseline model."""
    
    def __init__(self, num_layers: int = 3, **kwargs):
        super().__init__(**kwargs)
        
        # Build MLP layers
        layers = []
        current_dim = self.input_dim
        
        for i in range(num_layers - 1):
            next_dim = self.hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(current_dim, next_dim),
                nn.BatchNorm1d(next_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = next_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Task-specific heads
        self.rul_head = nn.Linear(current_dim, 1)
        self.soh_head = nn.Linear(current_dim, 1)
        self.soc_head = nn.Linear(current_dim, 1)
        self.capacity_head = nn.Linear(current_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through MLP."""
        # Handle sequence input by taking the last time step
        if len(x.shape) == 3:
            x = x[:, -1, :]
        
        # Encode features
        encoded = self.encoder(x)
        
        # Task-specific predictions
        predictions = {
            'rul': self.rul_head(encoded).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(encoded)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(encoded)).squeeze(-1),
            'capacity': self.capacity_head(encoded).squeeze(-1)
        }
        
        return predictions


class CNN1DBaselineModel(BatteryPredictionModel):
    """1D CNN baseline model for sequence data."""
    
    def __init__(self, num_filters: list = [32, 64, 128], **kwargs):
        super().__init__(**kwargs)
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = self.input_dim
        
        for out_channels in num_filters:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2, padding=1)
                )
            )
            in_channels = out_channels
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(num_filters[-1], self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task heads
        self.rul_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soh_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soc_head = nn.Linear(self.hidden_dim // 2, 1)
        self.capacity_head = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through 1D CNN."""
        # Handle different input shapes
        if len(x.shape) == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)
        elif len(x.shape) == 3:
            # Transpose for Conv1d (batch, channels, sequence)
            x = x.transpose(1, 2)
        
        # Apply convolutions
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)
        
        # Fully connected layers
        features = self.fc(x)
        
        # Predictions
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions


class VanillaRNN(BatteryPredictionModel):
    """Vanilla RNN baseline model."""
    
    def __init__(self, rnn_type: str = 'LSTM', **kwargs):
        super().__init__(**kwargs)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.input_dim, 
                self.hidden_dim, 
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                self.input_dim, 
                self.hidden_dim, 
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
        else:
            self.rnn = nn.RNN(
                self.input_dim, 
                self.hidden_dim, 
                num_layers=2,
                batch_first=True,
                dropout=0.2
            )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task heads
        self.rul_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soh_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soc_head = nn.Linear(self.hidden_dim // 2, 1)
        self.capacity_head = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through RNN."""
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(x)
        
        # Take last output
        last_output = rnn_out[:, -1, :]
        
        # Process through output layer
        features = self.output_layer(last_output)
        
        # Predictions
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions


class AttentionBaselineModel(BatteryPredictionModel):
    """Simple attention-based baseline model."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Feature projection
        self.feature_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            self.hidden_dim, 
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task heads
        self.rul_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soh_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soc_head = nn.Linear(self.hidden_dim // 2, 1)
        self.capacity_head = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with attention."""
        # Handle 2D input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Project features
        x = self.feature_proj(x)
        
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Pool over sequence
        x = x.mean(dim=1)
        
        # Output projection
        features = self.output_proj(x)
        
        # Predictions
        predictions = {
            'rul': self.rul_head(features).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(features)).squeeze(-1),
            'soc': torch.sigmoid(self.soc_head(features)).squeeze(-1),
            'capacity': self.capacity_head(features).squeeze(-1)
        }
        
        return predictions


# Sklearn-based models for comparison
class SklearnBaselineWrapper:
    """Wrapper for sklearn models to work with PyTorch pipeline."""
    
    def __init__(self, model_class, **model_kwargs):
        self.models = {
            'rul': model_class(**model_kwargs),
            'soh': model_class(**model_kwargs),
            'soc': model_class(**model_kwargs),
            'capacity': model_class(**model_kwargs)
        }
        self.is_fitted = False
        
    def fit(self, X, y_dict):
        """Fit sklearn models."""
        for task, model in self.models.items():
            if task in y_dict:
                model.fit(X, y_dict[task])
        self.is_fitted = True
        
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = {}
        for task, model in self.models.items():
            predictions[task] = model.predict(X)
            
            # Apply sigmoid for SOH and SOC
            if task in ['soh', 'soc']:
                predictions[task] = 1 / (1 + np.exp(-predictions[task]))
        
        return predictions


def create_sklearn_baselines():
    """Create sklearn baseline models."""
    return {
        'LinearRegression': SklearnBaselineWrapper(LinearRegression),
        'Ridge': SklearnBaselineWrapper(Ridge, alpha=1.0),
        'RandomForest': SklearnBaselineWrapper(
            RandomForestRegressor, 
            n_estimators=100,
            max_depth=10,
            random_state=42
        ),
        'GradientBoosting': SklearnBaselineWrapper(
            GradientBoostingRegressor,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }