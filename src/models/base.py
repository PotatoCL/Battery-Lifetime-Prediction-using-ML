"""
Base model class for battery performance prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any
import pytorch_lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class BatteryPredictionModel(pl.LightningModule, ABC):
    """Base class for battery prediction models."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 4,  # RUL, SOH, SOC, Capacity
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler: str = 'plateau',
        loss_weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler
        
        # Loss weights for multi-task learning
        self.loss_weights = loss_weights or {
            'rul': 1.0,
            'soh': 1.0,
            'soc': 1.0,
            'capacity': 1.0
        }
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning predictions for each target."""
        pass
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted multi-task loss."""
        losses = {}
        total_loss = 0
        
        # Individual losses
        for key in predictions:
            if key in targets:
                if key == 'rul':
                    # MAE loss for RUL
                    loss = nn.L1Loss()(predictions[key], targets[key])
                elif key in ['soh', 'capacity']:
                    # MSE loss for SOH and capacity
                    loss = nn.MSELoss()(predictions[key], targets[key])
                elif key == 'soc':
                    # Huber loss for SOC (robust to outliers)
                    loss = nn.HuberLoss()(predictions[key], targets[key])
                else:
                    loss = nn.MSELoss()(predictions[key], targets[key])
                
                losses[key] = loss
                total_loss += self.loss_weights.get(key, 1.0) * loss
        
        return total_loss, losses
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step."""
        features = batch['features']
        targets = {k: batch[k] for k in ['rul', 'soh', 'soc', 'capacity'] if k in batch}
        
        # Forward pass
        predictions = self(features)
        
        # Compute loss
        total_loss, losses = self.compute_loss(predictions, targets)
        
        # Log metrics
        self.log('train/loss', total_loss, prog_bar=True)
        for key, loss in losses.items():
            self.log(f'train/{key}_loss', loss)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step."""
        features = batch['features']
        targets = {k: batch[k] for k in ['rul', 'soh', 'soc', 'capacity'] if k in batch}
        
        # Forward pass
        predictions = self(features)
        
        # Compute loss
        total_loss, losses = self.compute_loss(predictions, targets)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)
        
        # Log metrics
        self.log('val/loss', total_loss, prog_bar=True)
        for key, loss in losses.items():
            self.log(f'val/{key}_loss', loss)
        for key, metric in metrics.items():
            self.log(f'val/{key}', metric)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step."""
        features = batch['features']
        targets = {k: batch[k] for k in ['rul', 'soh', 'soc', 'capacity'] if k in batch}
        
        # Forward pass
        predictions = self(features)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets)
        
        # Log metrics
        for key, metric in metrics.items():
            self.log(f'test/{key}', metric)
        
        return metrics
    
    def compute_metrics(self, predictions: Dict[str, torch.Tensor], 
                       targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        for key in predictions:
            if key in targets:
                pred = predictions[key].detach()
                target = targets[key].detach()
                
                # MAE
                mae = torch.mean(torch.abs(pred - target))
                metrics[f'{key}_mae'] = mae
                
                # RMSE
                rmse = torch.sqrt(torch.mean((pred - target) ** 2))
                metrics[f'{key}_rmse'] = rmse
                
                # MAPE (avoid division by zero)
                mask = target != 0
                if mask.any():
                    mape = torch.mean(torch.abs((pred[mask] - target[mask]) / target[mask])) * 100
                    metrics[f'{key}_mape'] = mape
                
                # R2 score
                ss_res = torch.sum((target - pred) ** 2)
                ss_tot = torch.sum((target - target.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-8)
                metrics[f'{key}_r2'] = r2
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        if self.scheduler_type == 'plateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=10,
                    verbose=True
                ),
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1
            }
        elif self.scheduler_type == 'cosine':
            scheduler = {
                'scheduler': CosineAnnealingLR(
                    optimizer,
                    T_max=100,
                    eta_min=1e-6
                ),
                'interval': 'epoch',
                'frequency': 1
            }
        else:
            return optimizer
        
        return [optimizer], [scheduler]
    
    def predict(self, features: torch.Tensor) -> Dict[str, np.ndarray]:
        """Make predictions on new data."""
        self.eval()
        with torch.no_grad():
            predictions = self(features)
            
        return {k: v.cpu().numpy() for k, v in predictions.items()}


class BaselineModel(BatteryPredictionModel):
    """Simple baseline model with fully connected layers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Network architecture
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task-specific heads
        self.rul_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soh_head = nn.Linear(self.hidden_dim // 2, 1)
        self.soc_head = nn.Linear(self.hidden_dim // 2, 1)
        self.capacity_head = nn.Linear(self.hidden_dim // 2, 1)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode features
        encoded = self.encoder(x)
        
        # Task-specific predictions
        predictions = {
            'rul': self.rul_head(encoded).squeeze(-1),
            'soh': torch.sigmoid(self.soh_head(encoded)).squeeze(-1),  # SOH in [0, 1]
            'soc': torch.sigmoid(self.soc_head(encoded)).squeeze(-1),  # SOC in [0, 1]
            'capacity': self.capacity_head(encoded).squeeze(-1)
        }
        
        return predictions