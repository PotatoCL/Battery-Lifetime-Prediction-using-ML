"""
Utility functions for model management.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import yaml
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing different model architectures."""
    
    _models = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class."""
        def decorator(model_class):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str, **kwargs):
        """Get a registered model by name."""
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered. Available: {list(cls._models.keys())}")
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls):
        """List all registered models."""
        return list(cls._models.keys())


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Union[str, Path],
    config: Optional[Dict] = None
):
    """Save model checkpoint with metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat(),
        'model_class': model.__class__.__name__,
        'num_parameters': count_parameters(model)
    }
    
    if config:
        checkpoint['config'] = config
    
    torch.save(checkpoint, save_path)
    logger.info(f"Saved checkpoint to {save_path}")
    
    # Save metadata separately for easy inspection
    metadata_path = save_path.with_suffix('.json')
    metadata = {
        'epoch': epoch,
        'metrics': metrics,
        'timestamp': checkpoint['timestamp'],
        'model_class': checkpoint['model_class'],
        'num_parameters': checkpoint['num_parameters']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
    logger.info(f"Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


def create_model_summary(model: nn.Module, input_shape: tuple) -> str:
    """Create a summary of the model architecture."""
    from torchsummary import summary
    import io
    from contextlib import redirect_stdout
    
    # Capture summary output
    f = io.StringIO()
    with redirect_stdout(f):
        try:
            summary(model, input_shape)
        except:
            # Fallback to manual summary
            total_params = count_parameters(model)
            return f"{model.__class__.__name__}\nTotal parameters: {total_params:,}"
    
    return f.getvalue()


def export_model_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    export_path: Union[str, Path],
    input_names: list = ['features'],
    output_names: list = ['rul', 'soh', 'soc', 'capacity'],
    dynamic_axes: Optional[Dict] = None
):
    """Export model to ONNX format for deployment."""
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dynamic_axes is None:
        dynamic_axes = {
            'features': {0: 'batch_size'},
            'rul': {0: 'batch_size'},
            'soh': {0: 'batch_size'},
            'soc': {0: 'batch_size'},
            'capacity': {0: 'batch_size'}
        }
    
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=11,
        do_constant_folding=True
    )
    
    logger.info(f"Exported model to ONNX: {export_path}")


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """Check if should stop training."""
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score improved."""
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = torch.tensor(weights)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble."""
        all_predictions = []
        
        for model in self.models:
            predictions = model(x)
            all_predictions.append(predictions)
        
        # Weighted average of predictions
        ensemble_predictions = {}
        for key in all_predictions[0].keys():
            stacked = torch.stack([pred[key] for pred in all_predictions])
            weighted = stacked * self.weights.view(-1, 1).to(stacked.device)
            ensemble_predictions[key] = weighted.sum(dim=0)
        
        return ensemble_predictions


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer based on name."""
    optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'rmsprop': torch.optim.RMSprop
    }
    
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    optimizer_class = optimizers[optimizer_name]
    
    # Default parameters for each optimizer
    default_params = {
        'adam': {'betas': (0.9, 0.999), 'eps': 1e-8},
        'adamw': {'betas': (0.9, 0.999), 'eps': 1e-8},
        'sgd': {'momentum': 0.9},
        'rmsprop': {'alpha': 0.99}
    }
    
    # Merge with provided kwargs
    params = {
        'lr': learning_rate,
        'weight_decay': weight_decay,
        **default_params.get(optimizer_name, {}),
        **kwargs
    }
    
    return optimizer_class(model.parameters(), **params)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler."""
    schedulers = {
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'step': torch.optim.lr_scheduler.StepLR,
        'exponential': torch.optim.lr_scheduler.ExponentialLR,
        'cyclic': torch.optim.lr_scheduler.CyclicLR
    }
    
    if scheduler_name not in schedulers:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    scheduler_class = schedulers[scheduler_name]
    
    # Default parameters for each scheduler
    default_params = {
        'cosine': {'T_max': 100, 'eta_min': 1e-6},
        'plateau': {'mode': 'min', 'factor': 0.5, 'patience': 10},
        'step': {'step_size': 30, 'gamma': 0.1},
        'exponential': {'gamma': 0.95},
        'cyclic': {'base_lr': 1e-4, 'max_lr': 1e-2, 'step_size_up': 10}
    }
    
    # Merge with provided kwargs
    params = {**default_params.get(scheduler_name, {}), **kwargs}
    
    return scheduler_class(optimizer, **params)


if __name__ == "__main__":
    # Example usage
    from .base import BaselineModel
    
    # Create model
    model = BaselineModel(input_dim=10, hidden_dim=128)
    
    # Count parameters
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, 'adamw', learning_rate=1e-3)
    scheduler = create_scheduler(optimizer, 'cosine', T_max=50)
    
    print("Model utilities loaded successfully!")