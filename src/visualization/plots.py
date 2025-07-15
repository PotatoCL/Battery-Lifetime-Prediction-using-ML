"""
Evaluation metrics for battery performance prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PredictionMetrics:
    """Container for prediction metrics."""
    mae: float
    rmse: float
    mape: float
    r2: float
    max_error: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2,
            'max_error': self.max_error
        }
    
    def __str__(self) -> str:
        return (f"MAE: {self.mae:.4f}, RMSE: {self.rmse:.4f}, "
                f"MAPE: {self.mape:.2f}%, R²: {self.r2:.4f}")


class BatteryMetrics:
    """Compute metrics for battery performance prediction."""
    
    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not mask.any():
            return 0.0
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return mape
    
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def compute_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute maximum absolute error."""
        return np.max(np.abs(y_true - y_pred))
    
    @classmethod
    def compute_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> PredictionMetrics:
        """Compute all metrics for a given prediction."""
        return PredictionMetrics(
            mae=cls.compute_mae(y_true, y_pred),
            rmse=cls.compute_rmse(y_true, y_pred),
            mape=cls.compute_mape(y_true, y_pred),
            r2=cls.compute_r2(y_true, y_pred),
            max_error=cls.compute_max_error(y_true, y_pred)
        )


class RULMetrics(BatteryMetrics):
    """Specialized metrics for Remaining Useful Life prediction."""
    
    @staticmethod
    def compute_alpha_lambda(y_true: np.ndarray, y_pred: np.ndarray,
                           alpha: float = 13, beta: float = 10) -> float:
        """
        Compute alpha-lambda accuracy for RUL prediction.
        
        This metric penalizes late predictions more than early predictions,
        which is important for maintenance scheduling.
        """
        diff = y_pred - y_true
        
        # Scoring function
        scores = np.where(
            diff < 0,  # Early prediction
            np.exp(-diff / alpha) - 1,
            np.exp(diff / beta) - 1  # Late prediction (higher penalty)
        )
        
        return np.mean(scores)
    
    @staticmethod
    def compute_prognostic_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                                 threshold: float = 0.1) -> float:
        """
        Compute prognostic horizon - how early the model can accurately predict failure.
        
        Returns the earliest cycle where prediction error is within threshold.
        """
        relative_error = np.abs(y_pred - y_true) / (y_true + 1e-8)
        
        # Find first prediction within threshold
        within_threshold = relative_error <= threshold
        if within_threshold.any():
            return len(y_true) - np.argmax(within_threshold)
        else:
            return 0
    
    @classmethod
    def compute_rul_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all RUL-specific metrics."""
        base_metrics = cls.compute_all_metrics(y_true, y_pred)
        
        return {
            **base_metrics.to_dict(),
            'alpha_lambda': cls.compute_alpha_lambda(y_true, y_pred),
            'prognostic_horizon': cls.compute_prognostic_horizon(y_true, y_pred)
        }


class SOHMetrics(BatteryMetrics):
    """Specialized metrics for State of Health prediction."""
    
    @staticmethod
    def compute_eol_prediction_error(y_true: np.ndarray, y_pred: np.ndarray,
                                   eol_threshold: float = 0.8) -> float:
        """
        Compute error in predicting End-of-Life point (when SOH drops below threshold).
        """
        # Find EOL cycles
        true_eol = np.where(y_true < eol_threshold)[0]
        pred_eol = np.where(y_pred < eol_threshold)[0]
        
        if len(true_eol) > 0 and len(pred_eol) > 0:
            return abs(true_eol[0] - pred_eol[0])
        elif len(true_eol) > 0:
            return len(y_true)  # Prediction never reached EOL
        else:
            return 0  # Neither reached EOL
    
    @staticmethod
    def compute_degradation_rate_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute error in degradation rate estimation."""
        # Fit linear regression to get degradation rates
        cycles = np.arange(len(y_true))
        
        true_rate = np.polyfit(cycles, y_true, 1)[0]
        pred_rate = np.polyfit(cycles, y_pred, 1)[0]
        
        return abs(true_rate - pred_rate)
    
    @classmethod
    def compute_soh_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all SOH-specific metrics."""
        base_metrics = cls.compute_all_metrics(y_true, y_pred)
        
        return {
            **base_metrics.to_dict(),
            'eol_error': cls.compute_eol_prediction_error(y_true, y_pred),
            'degradation_rate_error': cls.compute_degradation_rate_error(y_true, y_pred)
        }


class MultiTaskMetrics:
    """Metrics for multi-task battery prediction."""
    
    def __init__(self):
        self.rul_metrics = RULMetrics()
        self.soh_metrics = SOHMetrics()
        self.base_metrics = BatteryMetrics()
    
    def compute_task_metrics(self, predictions: Dict[str, np.ndarray],
                           targets: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each prediction task."""
        metrics = {}
        
        # RUL metrics
        if 'rul' in predictions and 'rul' in targets:
            metrics['rul'] = self.rul_metrics.compute_rul_metrics(
                targets['rul'], predictions['rul']
            )
        
        # SOH metrics
        if 'soh' in predictions and 'soh' in targets:
            metrics['soh'] = self.soh_metrics.compute_soh_metrics(
                targets['soh'], predictions['soh']
            )
        
        # SOC metrics
        if 'soc' in predictions and 'soc' in targets:
            metrics['soc'] = self.base_metrics.compute_all_metrics(
                targets['soc'], predictions['soc']
            ).to_dict()
        
        # Capacity metrics
        if 'capacity' in predictions and 'capacity' in targets:
            metrics['capacity'] = self.base_metrics.compute_all_metrics(
                targets['capacity'], predictions['capacity']
            ).to_dict()
        
        return metrics
    
    def aggregate_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across all tasks."""
        aggregated = {}
        
        for task, task_metrics in metrics.items():
            for metric_name, value in task_metrics.items():
                key = f"{task}_{metric_name}"
                aggregated[key] = value
        
        # Compute overall metrics
        mae_values = [m['mae'] for m in metrics.values() if 'mae' in m]
        rmse_values = [m['rmse'] for m in metrics.values() if 'rmse' in m]
        r2_values = [m['r2'] for m in metrics.values() if 'r2' in m]
        
        if mae_values:
            aggregated['overall_mae'] = np.mean(mae_values)
        if rmse_values:
            aggregated['overall_rmse'] = np.mean(rmse_values)
        if r2_values:
            aggregated['overall_r2'] = np.mean(r2_values)
        
        return aggregated


def evaluate_model(model, test_loader, device: str = 'cuda') -> Dict[str, Dict[str, float]]:
    """Evaluate model on test set."""
    model.eval()
    
    all_predictions = {task: [] for task in ['rul', 'soh', 'soc', 'capacity']}
    all_targets = {task: [] for task in ['rul', 'soh', 'soc', 'capacity']}
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            
            # Get predictions
            predictions = model(features)
            
            # Collect predictions and targets
            for task in all_predictions:
                if task in predictions:
                    all_predictions[task].append(predictions[task].cpu().numpy())
                if task in batch:
                    all_targets[task].append(batch[task].cpu().numpy())
    
    # Concatenate all batches
    for task in all_predictions:
        if all_predictions[task]:
            all_predictions[task] = np.concatenate(all_predictions[task])
        if all_targets[task]:
            all_targets[task] = np.concatenate(all_targets[task])
    
    # Compute metrics
    metrics_calculator = MultiTaskMetrics()
    metrics = metrics_calculator.compute_task_metrics(all_predictions, all_targets)
    
    return metrics


def plot_prediction_results(predictions: Dict[str, np.ndarray],
                          targets: Dict[str, np.ndarray],
                          save_path: Optional[str] = None):
    """Plot prediction results for all tasks."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    tasks = ['rul', 'soh', 'soc', 'capacity']
    titles = ['RUL Prediction', 'SOH Prediction', 'SOC Prediction', 'Capacity Prediction']
    
    for idx, (task, title) in enumerate(zip(tasks, titles)):
        if task in predictions and task in targets:
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(targets[task], predictions[task], alpha=0.5, s=10)
            
            # Perfect prediction line
            min_val = min(targets[task].min(), predictions[task].min())
            max_val = max(targets[task].max(), predictions[task].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            # Calculate metrics
            mae = mean_absolute_error(targets[task], predictions[task])
            r2 = r2_score(targets[task], predictions[task])
            
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title(f'{title}\nMAE: {mae:.3f}, R²: {r2:.3f}')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate predictions and targets
    n_samples = 1000
    predictions = {
        'rul': np.random.normal(100, 20, n_samples),
        'soh': np.random.uniform(0.7, 1.0, n_samples),
        'soc': np.random.uniform(0.2, 1.0, n_samples),
        'capacity': np.random.normal(2.0, 0.3, n_samples)
    }
    
    targets = {
        'rul': predictions['rul'] + np.random.normal(0, 5, n_samples),
        'soh': predictions['soh'] + np.random.normal(0, 0.02, n_samples),
        'soc': predictions['soc'] + np.random.normal(0, 0.05, n_samples),
        'capacity': predictions['capacity'] + np.random.normal(0, 0.1, n_samples)
    }
    
    # Compute metrics
    metrics_calc = MultiTaskMetrics()
    metrics = metrics_calc.compute_task_metrics(predictions, targets)
    
    print("Task-specific metrics:")
    for task, task_metrics in metrics.items():
        print(f"\n{task.upper()}:")
        for metric, value in task_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Aggregate metrics
    aggregated = metrics_calc.aggregate_metrics(metrics)
    print("\nAggregated metrics:")
    for metric, value in aggregated.items():
        if metric.startswith('overall'):
            print(f"  {metric}: {value:.4f}")