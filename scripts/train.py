"""
Training script for battery performance prediction models.
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import BatteryDataLoader
from src.features.extractor import FeatureEngineering
from src.models.cp_gru import CPGRU, EnhancedCPGRU
from src.models.cp_lstm import CPLSTM, StackedCPLSTM
from src.models.cp_transformer import CPTransformer, HierarchicalCPTransformer
from src.models.base import BaselineModel
from src.evaluation.metrics import evaluate_model
from src.visualization.plots import BatteryVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatteryDataset(Dataset):
    """PyTorch Dataset for battery data."""
    
    def __init__(self, features_df, sequence_length=50):
        self.features_df = features_df
        self.sequence_length = sequence_length
        
        # Separate features and targets
        self.feature_cols = [col for col in features_df.columns 
                           if not col.endswith('_current') and col != 'cycle' and col != 'battery_id']
        self.target_cols = ['rul_current', 'soh_current', 'capacity_current']
        
        # Create sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self):
        """Create sequences from dataframe."""
        sequences = []
        
        for battery_id in self.features_df['battery_id'].unique():
            battery_data = self.features_df[self.features_df['battery_id'] == battery_id]
            
            for i in range(len(battery_data) - self.sequence_length + 1):
                seq_data = battery_data.iloc[i:i+self.sequence_length]
                
                # Features sequence
                features = seq_data[self.feature_cols].values
                
                # Targets (last time step)
                targets = seq_data[self.target_cols].iloc[-1].values
                
                # Add SOC (simplified - could be more sophisticated)
                soc = np.random.uniform(0.2, 0.9)  # Placeholder
                
                sequences.append({
                    'features': features,
                    'rul': targets[0],
                    'soh': targets[1],
                    'capacity': targets[2],
                    'soc': soc
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sample = self.sequences[idx]
        return {
            'features': torch.FloatTensor(sample['features']),
            'rul': torch.FloatTensor([sample['rul']]),
            'soh': torch.FloatTensor([sample['soh']]),
            'capacity': torch.FloatTensor([sample['capacity']]),
            'soc': torch.FloatTensor([sample['soc']])
        }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Prepare data for training."""
    logger.info("Loading battery data...")
    
    # Load data
    loader = BatteryDataLoader(config['data']['raw_path'])
    train_df, val_df, test_df = loader.load_nasa_data()
    
    logger.info("Extracting features...")
    
    # Feature engineering
    fe = FeatureEngineering(window_sizes=config['features']['window_sizes'])
    
    train_features = []
    val_features = []
    test_features = []
    
    # Process each battery
    for battery_id in train_df['battery_id'].unique():
        battery_data = train_df[train_df['battery_id'] == battery_id]
        features = fe.engineer_features(battery_data)
        features['battery_id'] = battery_id
        train_features.append(features)
    
    for battery_id in val_df['battery_id'].unique():
        battery_data = val_df[val_df['battery_id'] == battery_id]
        features = fe.engineer_features(battery_data)
        features['battery_id'] = battery_id
        val_features.append(features)
    
    for battery_id in test_df['battery_id'].unique():
        battery_data = test_df[test_df['battery_id'] == battery_id]
        features = fe.engineer_features(battery_data)
        features['battery_id'] = battery_id
        test_features.append(features)
    
    # Combine features
    train_features_df = pd.concat(train_features, ignore_index=True)
    val_features_df = pd.concat(val_features, ignore_index=True)
    test_features_df = pd.concat(test_features, ignore_index=True)
    
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    feature_cols = [col for col in train_features_df.columns 
                   if col not in ['cycle', 'battery_id'] and not col.endswith('_current')]
    
    train_features_df[feature_cols] = scaler.fit_transform(train_features_df[feature_cols])
    val_features_df[feature_cols] = scaler.transform(val_features_df[feature_cols])
    test_features_df[feature_cols] = scaler.transform(test_features_df[feature_cols])
    
    logger.info(f"Feature extraction complete. Train: {len(train_features_df)}, "
               f"Val: {len(val_features_df)}, Test: {len(test_features_df)}")
    
    return train_features_df, val_features_df, test_features_df, scaler


def create_model(model_type: str, config: dict):
    """Create model based on type and configuration."""
    input_dim = config['model']['input_dim']
    hidden_dim = config['model']['hidden_dim']
    
    if model_type == 'baseline':
        return BaselineModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            learning_rate=config['training']['learning_rate']
        )
    
    elif model_type == 'cp-gru':
        return CPGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate']
        )
    
    elif model_type == 'enhanced-cp-gru':
        return EnhancedCPGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate'],
            use_residual=True,
            use_layer_norm=True
        )
    
    elif model_type == 'cp-lstm':
        return CPLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate']
        )
    
    elif model_type == 'stacked-cp-lstm':
        return StackedCPLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate'],
            num_blocks=2
        )
    
    elif model_type == 'cp-transformer':
        return CPTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate']
        )
    
    elif model_type == 'hierarchical-cp-transformer':
        return HierarchicalCPTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            patch_size=config['cyclepatch']['patch_size'],
            patch_stride=config['cyclepatch']['stride'],
            learning_rate=config['training']['learning_rate'],
            scales=[5, 10, 20]
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model(model, train_loader, val_loader, config: dict, model_name: str):
    """Train the model using PyTorch Lightning."""
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{model_name}",
        filename='{epoch}-{val_loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=config['training']['early_stopping_patience'],
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    tb_logger = TensorBoardLogger(
        save_dir='logs',
        name=model_name,
        version=datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=tb_logger,
        gradient_clip_val=config['training'].get('gradient_clip', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        precision=config['training'].get('precision', 32)
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Train battery prediction models')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='cp-transformer',
                       choices=['baseline', 'cp-gru', 'enhanced-cp-gru', 'cp-lstm', 
                               'stacked-cp-lstm', 'cp-transformer', 'hierarchical-cp-transformer'],
                       help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data
    train_features, val_features, test_features, scaler = prepare_data(config)
    
    # Create datasets
    train_dataset = BatteryDataset(train_features, config['data']['sequence_length'])
    val_dataset = BatteryDataset(val_features, config['data']['sequence_length'])
    test_dataset = BatteryDataset(test_features, config['data']['sequence_length'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                          shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers)
    
    # Update config with actual input dim
    config['model']['input_dim'] = len(train_dataset.feature_cols)
    
    # Create model
    model = create_model(args.model, config)
    logger.info(f"Created {args.model} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    logger.info("Starting training...")
    trainer = train_model(model, train_loader, val_loader, config, args.model)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.test(model, test_loader)
    
    # Save results
    results_dir = Path('results') / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    import json
    with open(results_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Create visualizations
    visualizer = BatteryVisualizer()
    
    # Plot training history
    if hasattr(trainer.logger, 'experiment'):
        # Extract training history from tensorboard logs
        # This is simplified - in practice you'd extract from the actual logs
        pass
    
    logger.info(f"Training complete! Results saved to {results_dir}")


if __name__ == "__main__":
    main()