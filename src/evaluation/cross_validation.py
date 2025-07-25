"""
Cross-validation strategies for battery performance prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Generator, Union
from sklearn.model_selection import KFold, TimeSeriesSplit
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pathlib import Path
import logging
from copy import deepcopy
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CVResults:
    """Container for cross-validation results."""
    fold_metrics: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    fold_predictions: Optional[List[Dict]] = None
    best_fold: Optional[int] = None
    
    def summary(self) -> pd.DataFrame:
        """Create summary DataFrame of CV results."""
        summary_data = {
            'Mean': self.mean_metrics,
            'Std': self.std_metrics
        }
        
        # Add individual fold results
        for i, fold_metric in enumerate(self.fold_metrics):
            summary_data[f'Fold_{i+1}'] = fold_metric
        
        return pd.DataFrame(summary_data).T
    
    def print_summary(self):
        """Print CV results summary."""
        print("\nCross-Validation Results:")
        print("=" * 50)
        
        for metric, mean_val in self.mean_metrics.items():
            std_val = self.std_metrics.get(metric, 0)
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        if self.best_fold is not None:
            print(f"\nBest fold: {self.best_fold + 1}")


class BatteryKFold:
    """K-Fold cross-validation for battery data (split by battery ID)."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    def split(self, dataset, battery_ids: np.ndarray) -> Generator:
        """
        Generate train/val indices split by battery ID.
        
        Args:
            dataset: PyTorch dataset
            battery_ids: Array of battery IDs for each sample
            
        Yields:
            train_indices, val_indices for each fold
        """
        unique_batteries = np.unique(battery_ids)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(unique_batteries)
        
        # Create folds of batteries
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        
        for train_batteries, val_batteries in kf.split(unique_batteries):
            # Get battery IDs for this fold
            train_battery_ids = unique_batteries[train_batteries]
            val_battery_ids = unique_batteries[val_batteries]
            
            # Get sample indices
            train_mask = np.isin(battery_ids, train_battery_ids)
            val_mask = np.isin(battery_ids, val_battery_ids)
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            yield train_indices, val_indices


class TimeSeriesCV:
    """Time series cross-validation for battery data."""
    
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, 
                 gap: int = 0):
        """
        Initialize time series CV.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses expanding window)
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
    def split(self, dataset, time_indices: Optional[np.ndarray] = None) -> Generator:
        """
        Generate train/val indices for time series CV.
        
        Args:
            dataset: PyTorch dataset
            time_indices: Time indices for each sample (if None, uses sequential)
            
        Yields:
            train_indices, val_indices for each fold
        """
        n_samples = len(dataset)
        
        if time_indices is None:
            time_indices = np.arange(n_samples)
        
        # Sort by time
        sorted_indices = np.argsort(time_indices)
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size, 
                              gap=self.gap)
        
        for train_idx, val_idx in tscv.split(sorted_indices):
            train_indices = sorted_indices[train_idx]
            val_indices = sorted_indices[val_idx]
            
            yield train_indices, val_indices


class BlockingTimeSeriesCV:
    """Blocking time series CV for battery cycle data."""
    
    def __init__(self, n_splits: int = 5, min_train_size: int = 100):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        
    def split(self, dataset, battery_ids: np.ndarray, 
              cycle_numbers: np.ndarray) -> Generator:
        """
        Split data by blocking time series within each battery.
        
        Args:
            dataset: PyTorch dataset
            battery_ids: Battery ID for each sample
            cycle_numbers: Cycle number for each sample
            
        Yields:
            train_indices, val_indices for each fold
        """
        unique_batteries = np.unique(battery_ids)
        
        for fold in range(self.n_splits):
            train_indices = []
            val_indices = []
            
            for battery_id in unique_batteries:
                # Get indices for this battery
                battery_mask = battery_ids == battery_id
                battery_indices = np.where(battery_mask)[0]
                battery_cycles = cycle_numbers[battery_mask]
                
                # Sort by cycle
                sorted_idx = np.argsort(battery_cycles)
                sorted_indices = battery_indices[sorted_idx]
                
                # Determine split point
                n_battery_samples = len(sorted_indices)
                split_size = max(self.min_train_size, 
                               n_battery_samples // self.n_splits)
                
                # Calculate train/val split for this fold
                val_start = fold * (n_battery_samples - split_size) // (self.n_splits - 1)
                val_end = min(val_start + split_size, n_battery_samples)
                
                # Add to fold indices
                if val_start > 0:
                    train_indices.extend(sorted_indices[:val_start])
                if val_end < n_battery_samples:
                    train_indices.extend(sorted_indices[val_end:])
                val_indices.extend(sorted_indices[val_start:val_end])
            
            yield np.array(train_indices), np.array(val_indices)


class CrossValidator:
    """Main cross-validation class for battery models."""
    
    def __init__(self, cv_strategy: Union[str, object] = 'battery_kfold', 
                 n_splits: int = 5, **cv_kwargs):
        """
        Initialize cross-validator.
        
        Args:
            cv_strategy: 'battery_kfold', 'timeseries', 'blocking_timeseries', or custom
            n_splits: Number of CV splits
            **cv_kwargs: Additional arguments for CV strategy
        """
        if isinstance(cv_strategy, str):
            if cv_strategy == 'battery_kfold':
                self.cv = BatteryKFold(n_splits=n_splits, **cv_kwargs)
            elif cv_strategy == 'timeseries':
                self.cv = TimeSeriesCV(n_splits=n_splits, **cv_kwargs)
            elif cv_strategy == 'blocking_timeseries':
                self.cv = BlockingTimeSeriesCV(n_splits=n_splits, **cv_kwargs)
            else:
                raise ValueError(f"Unknown CV strategy: {cv_strategy}")
        else:
            self.cv = cv_strategy
        
        self.n_splits = n_splits
        
    def cross_validate(
        self,
        model_class,
        dataset,
        model_kwargs: Dict,
        trainer_kwargs: Dict,
        batch_size: int = 32,
        num_workers: int = 4,
        split_args: Optional[Dict] = None,
        save_predictions: bool = False,
        save_models: bool = False,
        output_dir: Optional[Path] = None
    ) -> CVResults:
        """
        Perform cross-validation.
        
        Args:
            model_class: Model class to instantiate
            dataset: PyTorch dataset
            model_kwargs: Arguments for model initialization
            trainer_kwargs: Arguments for PyTorch Lightning trainer
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            split_args: Arguments for split method (e.g., battery_ids)
            save_predictions: Whether to save fold predictions
            save_models: Whether to save trained models
            output_dir: Directory to save outputs
            
        Returns:
            CVResults object with metrics and predictions
        """
        fold_metrics = []
        fold_predictions = [] if save_predictions else None
        
        # Get splits
        if split_args is None:
            split_args = {}
        
        splits = list(self.cv.split(dataset, **split_args))
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\nFold {fold + 1}/{self.n_splits}")
            logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            
            # Create data loaders
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers
            )
            
            # Create model
            model = model_class(**model_kwargs)
            
            # Create trainer
            fold_trainer_kwargs = deepcopy(trainer_kwargs)
            if output_dir and 'logger' in fold_trainer_kwargs:
                # Update logger for this fold
                fold_trainer_kwargs['logger'].version = f"fold_{fold + 1}"
            
            trainer = pl.Trainer(**fold_trainer_kwargs)
            
            # Train model
            trainer.fit(model, train_loader, val_loader)
            
            # Evaluate on validation set
            val_results = trainer.test(model, val_loader)
            fold_metrics.append(val_results[0])
            
            # Save predictions if requested
            if save_predictions:
                predictions = self._get_predictions(model, val_loader)
                fold_predictions.append({
                    'fold': fold + 1,
                    'predictions': predictions,
                    'val_indices': val_idx
                })
            
            # Save model if requested
            if save_models and output_dir:
                model_path = output_dir / f"model_fold_{fold + 1}.pth"
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved model to {model_path}")
        
        # Calculate summary statistics
        mean_metrics = {}
        std_metrics = {}
        
        for metric in fold_metrics[0].keys():
            values = [fold[metric] for fold in fold_metrics]
            mean_metrics[metric] = np.mean(values)
            std_metrics[metric] = np.std(values)
        
        # Find best fold (based on validation loss)
        if 'val_loss' in fold_metrics[0]:
            val_losses = [fold['val_loss'] for fold in fold_metrics]
            best_fold = np.argmin(val_losses)
        else:
            best_fold = None
        
        results = CVResults(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            fold_predictions=fold_predictions,
            best_fold=best_fold
        )
        
        return results
    
    def _get_predictions(self, model, dataloader):
        """Get model predictions on a dataset."""
        model.eval()
        all_predictions = {}
        
        with torch.no_grad():
            for batch in dataloader:
                predictions = model(batch['features'])
                
                for key, pred in predictions.items():
                    if key not in all_predictions:
                        all_predictions[key] = []
                    all_predictions[key].append(pred.cpu().numpy())
        
        # Concatenate predictions
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
        
        return all_predictions


class NestedCV:
    """Nested cross-validation for hyperparameter tuning."""
    
    def __init__(self, outer_cv, inner_cv, param_grid: Dict):
        """
        Initialize nested CV.
        
        Args:
            outer_cv: Outer CV for model evaluation
            inner_cv: Inner CV for hyperparameter tuning
            param_grid: Hyperparameter search space
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.param_grid = param_grid
        
    def fit(
        self,
        model_class,
        dataset,
        base_model_kwargs: Dict,
        trainer_kwargs: Dict,
        optimization_metric: str = 'val_loss',
        minimize: bool = True,
        **cv_kwargs
    ) -> Dict:
        """
        Perform nested cross-validation.
        
        Returns:
            Dictionary with results and best parameters for each outer fold
        """
        from sklearn.model_selection import ParameterGrid
        
        outer_results = []
        best_params_per_fold = []
        
        # Get outer splits
        outer_splits = list(self.outer_cv.split(dataset, **cv_kwargs.get('split_args', {})))
        
        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
            logger.info(f"\nOuter Fold {outer_fold + 1}/{len(outer_splits)}")
            
            # Create outer train dataset
            outer_train_subset = Subset(dataset, outer_train_idx)
            
            # Grid search on inner CV
            param_scores = []
            
            for params in ParameterGrid(self.param_grid):
                # Update model kwargs with current params
                model_kwargs = {**base_model_kwargs, **params}
                
                # Perform inner CV
                inner_validator = CrossValidator(self.inner_cv)
                inner_results = inner_validator.cross_validate(
                    model_class,
                    outer_train_subset,
                    model_kwargs,
                    trainer_kwargs,
                    **cv_kwargs
                )
                
                # Get optimization metric
                score = inner_results.mean_metrics[optimization_metric]
                param_scores.append((params, score))
                
                logger.info(f"  Params: {params}, Score: {score:.4f}")
            
            # Find best parameters
            if minimize:
                best_params, best_score = min(param_scores, key=lambda x: x[1])
            else:
                best_params, best_score = max(param_scores, key=lambda x: x[1])
            
            logger.info(f"  Best params: {best_params}, Score: {best_score:.4f}")
            best_params_per_fold.append(best_params)
            
            # Train final model with best params on full outer train set
            final_model_kwargs = {**base_model_kwargs, **best_params}
            model = model_class(**final_model_kwargs)
            
            # Create data loaders
            outer_train_loader = DataLoader(
                outer_train_subset,
                batch_size=cv_kwargs.get('batch_size', 32),
                shuffle=True,
                num_workers=cv_kwargs.get('num_workers', 4)
            )
            outer_test_subset = Subset(dataset, outer_test_idx)
            outer_test_loader = DataLoader(
                outer_test_subset,
                batch_size=cv_kwargs.get('batch_size', 32),
                shuffle=False,
                num_workers=cv_kwargs.get('num_workers', 4)
            )
            
            # Train and evaluate
            trainer = pl.Trainer(**trainer_kwargs)
            trainer.fit(model, outer_train_loader)
            test_results = trainer.test(model, outer_test_loader)
            
            outer_results.append({
                'fold': outer_fold + 1,
                'best_params': best_params,
                'inner_cv_score': best_score,
                'test_metrics': test_results[0]
            })
        
        return {
            'outer_results': outer_results,
            'best_params_per_fold': best_params_per_fold,
            'mean_test_metrics': self._calculate_mean_metrics(
                [r['test_metrics'] for r in outer_results]
            )
        }
    
    def _calculate_mean_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Calculate mean metrics across folds."""
        mean_metrics = {}
        
        for metric in metrics_list[0].keys():
            values = [m[metric] for m in metrics_list]
            mean_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return mean_metrics


def plot_cv_results(cv_results: CVResults, metrics: List[str] = None, 
                    save_path: Optional[Path] = None):
    """Plot cross-validation results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if metrics is None:
        metrics = list(cv_results.mean_metrics.keys())[:4]
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in cv_results.mean_metrics]
    
    fig, axes = plt.subplots(1, len(available_metrics), 
                            figsize=(5 * len(available_metrics), 5))
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Get fold values
        fold_values = [fold.get(metric, 0) for fold in cv_results.fold_metrics]
        
        # Plot bars
        x = range(1, len(fold_values) + 1)
        bars = ax.bar(x, fold_values, alpha=0.7)
        
        # Highlight best fold
        if cv_results.best_fold is not None and metric == 'val_loss':
            bars[cv_results.best_fold].set_color('green')
        
        # Add mean line
        mean_val = cv_results.mean_metrics[metric]
        ax.axhline(y=mean_val, color='red', linestyle='--', 
                  label=f'Mean: {mean_val:.3f}')
        
        # Add std band
        std_val = cv_results.std_metrics.get(metric, 0)
        ax.fill_between(x, mean_val - std_val, mean_val + std_val, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Across Folds')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# Example usage
if __name__ == "__main__":
    # Example CV setup
    cv = CrossValidator('battery_kfold', n_splits=5)
    print(f"Cross-validator created with {cv.n_splits} splits")
    
    # Example nested CV
    outer_cv = BatteryKFold(n_splits=5)
    inner_cv = BatteryKFold(n_splits=3)
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'hidden_dim': [128, 256]
    }
    
    nested_cv = NestedCV(outer_cv, inner_cv, param_grid)
    print("Nested CV created for hyperparameter tuning")