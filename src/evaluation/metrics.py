"""
Evaluation metrics for battery performance prediction.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict, List, Optional, Tuple, Union
import torch
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


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
    def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not mask.any():
            return 0.0
        
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
        return smape
    
    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def compute_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute maximum absolute error."""
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def compute_percentile_error(y_true: np.ndarray, y_pred: np.ndarray, 
                                percentile: float = 95) -> float:
        """Compute percentile of absolute errors."""
        errors = np.abs(y_true - y_pred)
        return np.percentile(errors, percentile)
    
    @staticmethod
    def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        return np.corrcoef(y_true, y_pred)[0, 1]
    
    @staticmethod
    def compute_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute relative error for each prediction."""
        mask = y_true != 0
        rel_error = np.zeros_like(y_true)
        rel_error[mask] = (y_pred[mask] - y_true[mask]) / y_true[mask]
        return rel_error
    
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
    
    @classmethod
    def compute_extended_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute extended set of metrics."""
compute_extended_metrics(y_true, y_pred)
        
        rul_specific = {
            'alpha_lambda': cls.compute_alpha_lambda(y_true, y_pred),
            'prognostic_horizon': cls.compute_prognostic_horizon(y_true, y_pred),
            'convergence_horizon': cls.compute_convergence_horizon(y_true, y_pred),
            'timeliness_score': cls.compute_timeliness_score(y_true, y_pred)
        }
        
        return {**base_metrics, **rul_specific}


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
    
    @staticmethod
    def compute_knee_point_error(y_true: np.ndarray, y_pred: np.ndarray,
                               smoothing_window: int = 5) -> float:
        """
        Compute error in knee point detection.
        
        The knee point is where degradation accelerates.
        """
        from scipy.signal import savgol_filter
        
        def find_knee_point(data):
            if len(data) < smoothing_window + 2:
                return len(data) // 2
            
            # Smooth data
            smoothed = savgol_filter(data, smoothing_window, 3)
            
            # Compute second derivative
            d2y = np.gradient(np.gradient(smoothed))
            
            # Find maximum curvature
            return np.argmax(np.abs(d2y))
        
        true_knee = find_knee_point(y_true)
        pred_knee = find_knee_point(y_pred)
        
        return abs(true_knee - pred_knee)
    
    @staticmethod
    def compute_monotonicity_score(y_pred: np.ndarray) -> float:
        """
        Compute monotonicity score for SOH predictions.
        
        SOH should generally decrease monotonically.
        """
        # Count violations of monotonicity
        violations = np.sum(np.diff(y_pred) > 0)
        total_transitions = len(y_pred) - 1
        
        if total_transitions == 0:
            return 1.0
        
        return 1.0 - (violations / total_transitions)
    
    @staticmethod
    def compute_capacity_retention_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                          retention_levels: List[float] = [0.95, 0.9, 0.85, 0.8]) -> Dict[str, float]:
        """
        Compute accuracy at specific retention levels.
        """
        accuracies = {}
        
        for level in retention_levels:
            # Find when each reaches the retention level
            true_idx = np.where(y_true <= level)[0]
            pred_idx = np.where(y_pred <= level)[0]
            
            if len(true_idx) > 0 and len(pred_idx) > 0:
                error = abs(true_idx[0] - pred_idx[0])
            elif len(true_idx) > 0:
                error = len(y_true)
            else:
                error = 0
            
            accuracies[f'retention_{int(level*100)}_error'] = error
        
        return accuracies
    
    @classmethod
    def compute_soh_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all SOH-specific metrics."""
        base_metrics = cls.compute_extended_metrics(y_true, y_pred)
        
        soh_specific = {
            'eol_error': cls.compute_eol_prediction_error(y_true, y_pred),
            'degradation_rate_error': cls.compute_degradation_rate_error(y_true, y_pred),
            'knee_point_error': cls.compute_knee_point_error(y_true, y_pred),
            'monotonicity_score': cls.compute_monotonicity_score(y_pred)
        }
        
        # Add retention accuracy metrics
        retention_metrics = cls.compute_capacity_retention_accuracy(y_true, y_pred)
        
        return {**base_metrics, **soh_specific, **retention_metrics}


class CapacityMetrics(BatteryMetrics):
    """Specialized metrics for capacity prediction."""
    
    @staticmethod
    def compute_fade_rate_accuracy(y_true: np.ndarray, y_pred: np.ndarray,
                                 window_size: int = 10) -> float:
        """Compute accuracy of capacity fade rate prediction."""
        if len(y_true) < window_size:
            return 0.0
        
        # Compute fade rates using rolling windows
        true_rates = []
        pred_rates = []
        
        for i in range(len(y_true) - window_size + 1):
            true_window = y_true[i:i+window_size]
            pred_window = y_pred[i:i+window_size]
            
            # Linear fit to get fade rate
            x = np.arange(window_size)
            true_rate = np.polyfit(x, true_window, 1)[0]
            pred_rate = np.polyfit(x, pred_window, 1)[0]
            
            true_rates.append(true_rate)
            pred_rates.append(pred_rate)
        
        # Compute accuracy
        true_rates = np.array(true_rates)
        pred_rates = np.array(pred_rates)
        
        mae = np.mean(np.abs(true_rates - pred_rates))
        return 1.0 - min(mae / (np.mean(np.abs(true_rates)) + 1e-8), 1.0)
    
    @staticmethod
    def compute_uncertainty_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                                      y_std: Optional[np.ndarray] = None,
                                      confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Compute uncertainty calibration metrics if uncertainty estimates are provided.
        """
        if y_std is None:
            return {}
        
        # Compute z-scores
        z_scores = np.abs(y_true - y_pred) / (y_std + 1e-8)
        
        # Expected coverage for confidence level
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Actual coverage
        coverage = np.mean(z_scores <= z_critical)
        
        # Calibration error
        calibration_error = abs(coverage - confidence_level)
        
        # Average uncertainty
        avg_uncertainty = np.mean(y_std)
        
        return {
            'coverage': coverage,
            'calibration_error': calibration_error,
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_mae_ratio': avg_uncertainty / np.mean(np.abs(y_true - y_pred))
        }
    
    @classmethod
    def compute_capacity_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray,
                               y_std: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all capacity-specific metrics."""
        base_metrics = cls.compute_extended_metrics(y_true, y_pred)
        
        capacity_specific = {
            'fade_rate_accuracy': cls.compute_fade_rate_accuracy(y_true, y_pred)
        }
        
        # Add uncertainty metrics if available
        if y_std is not None:
            uncertainty_metrics = cls.compute_uncertainty_calibration(y_true, y_pred, y_std)
            capacity_specific.update(uncertainty_metrics)
        
        return {**base_metrics, **capacity_specific}


class MultiTaskMetrics:
    """Metrics for multi-task battery prediction."""
    
    def __init__(self):
        self.rul_metrics = RULMetrics()
        self.soh_metrics = SOHMetrics()
        self.capacity_metrics = CapacityMetrics()
        self.base_metrics = BatteryMetrics()
    
    def compute_task_metrics(self, predictions: Dict[str, np.ndarray],
                           targets: Dict[str, np.ndarray],
                           uncertainties: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Dict[str, float]]:
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
            metrics['soc'] = self.base_metrics.compute_extended_metrics(
                targets['soc'], predictions['soc']
            )
        
        # Capacity metrics
        if 'capacity' in predictions and 'capacity' in targets:
            capacity_std = uncertainties.get('capacity') if uncertainties else None
            metrics['capacity'] = self.capacity_metrics.compute_capacity_metrics(
                targets['capacity'], predictions['capacity'], capacity_std
            )
        
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
    
    def create_metrics_report(self, metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create a formatted metrics report."""
        # Create DataFrame with tasks as columns and metrics as rows
        report_data = {}
        
        for task, task_metrics in metrics.items():
            report_data[task.upper()] = task_metrics
        
        df = pd.DataFrame(report_data).T
        
        # Sort columns by importance
        priority_metrics = ['mae', 'rmse', 'mape', 'r2', 'correlation']
        other_metrics = [col for col in df.columns if col not in priority_metrics]
        ordered_columns = [col for col in priority_metrics if col in df.columns] + other_metrics
        
        return df[ordered_columns]


def evaluate_model(model, test_loader, device: str = 'cuda',
                  return_predictions: bool = False) -> Union[Dict[str, Dict[str, float]], 
                                                            Tuple[Dict[str, Dict[str, float]], Dict]]:
    """Evaluate model on test set."""
    model.eval()
    model = model.to(device)
    
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
    
    if return_predictions:
        return metrics, {'predictions': all_predictions, 'targets': all_targets}
    
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


def plot_error_analysis(predictions: Dict[str, np.ndarray],
                       targets: Dict[str, np.ndarray],
                       task: str = 'soh',
                       save_path: Optional[str] = None):
    """Detailed error analysis plots for a specific task."""
    if task not in predictions or task not in targets:
        raise ValueError(f"Task {task} not found in predictions/targets")
    
    y_true = targets[task]
    y_pred = predictions[task]
    errors = y_pred - y_true
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    # 1. Error distribution
    ax = axes[0]
    ax.hist(errors, bins=50, density=True, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    # Fit normal distribution
    mu, std = stats.norm.fit(errors)
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={std:.3f})')
    
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()
    
    # 2. Q-Q plot
    ax = axes[1]
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    
    # 3. Residuals vs predicted
    ax = axes[2]
    ax.scatter(y_pred, errors, alpha=0.5, s=10)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    # Add confidence bands
    std_error = np.std(errors)
    ax.fill_between(sorted(y_pred), -2*std_error, 2*std_error, 
                   alpha=0.2, color='red', label='±2σ')
    
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title('Residuals vs Predicted')
    ax.legend()
    
    # 4. Absolute error vs true values
    ax = axes[3]
    ax.scatter(y_true, np.abs(errors), alpha=0.5, s=10)
    
    # Add trend line
    z = np.polyfit(y_true, np.abs(errors), 1)
    p = np.poly1d(z)
    ax.plot(sorted(y_true), p(sorted(y_true)), "r--", linewidth=2)
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs True Values')
    
    # 5. Error autocorrelation
    ax = axes[4]
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(errors, ax=ax, lags=min(40, len(errors)//4))
    ax.set_title('Error Autocorrelation')
    
    # 6. Cumulative error
    ax = axes[5]
    cumulative_error = np.cumsum(errors)
    ax.plot(cumulative_error)
    ax.fill_between(range(len(cumulative_error)), 0, cumulative_error, alpha=0.3)
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Cumulative Error')
    ax.set_title('Cumulative Error')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{task.upper()} Error Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_confidence_intervals(predictions: np.ndarray,
                               method: str = 'bootstrap',
                               confidence_level: float = 0.95,
                               n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confidence intervals for predictions.
    
    Args:
        predictions: Model predictions
        method: 'bootstrap' or 'quantile'
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        lower_bound, upper_bound arrays
    """
    alpha = 1 - confidence_level
    
    if method == 'bootstrap':
        # Bootstrap confidence intervals
        bootstrap_preds = []
        n_samples = len(predictions)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_preds.append(predictions[indices])
        
        bootstrap_preds = np.array(bootstrap_preds)
        lower = np.percentile(bootstrap_preds, alpha/2 * 100, axis=0)
        upper = np.percentile(bootstrap_preds, (1 - alpha/2) * 100, axis=0)
        
    elif method == 'quantile':
        # Simple quantile-based intervals
        std = np.std(predictions)
        z_score = stats.norm.ppf(1 - alpha/2)
        lower = predictions - z_score * std
        upper = predictions + z_score * std
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return lower, upper


# Utility functions for metric visualization
def create_metric_heatmap(metrics_dict: Dict[str, Dict[str, float]],
                         save_path: Optional[str] = None):
    """Create a heatmap of metrics across models and tasks."""
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Normalize metrics to [0, 1] for better visualization
    # For error metrics (lower is better), invert the scale
    normalized_df = df.copy()
    
    for col in normalized_df.columns:
        if any(error_term in col for error_term in ['mae', 'rmse', 'error', 'mape']):
            # Invert error metrics
            normalized_df[col] = 1 - (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
        else:
            # Regular normalization for scores
            normalized_df[col] = (normalized_df[col] - normalized_df[col].min()) / (normalized_df[col].max() - normalized_df[col].min())
    
    sns.heatmap(normalized_df, annot=True, fmt='.3f', cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Score (Higher is Better)'})
    
    plt.title('Model Performance Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


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
    report = metrics_calc.create_metrics_report(metrics)
    print(report)
    
    # Aggregate metrics
    aggregated = metrics_calc.aggregate_metrics(metrics)
    print("\nAggregated metrics:")
    for metric, value in aggregated.items():
        if metric.startswith('overall'):
            print(f"  {metric}: {value:.4f}")compute_all_metrics(y_true, y_pred)
        
        extended = base_metrics.to_dict()
        extended.update({
            'smape': cls.compute_smape(y_true, y_pred),
            'correlation': cls.compute_correlation(y_true, y_pred),
            'p95_error': cls.compute_percentile_error(y_true, y_pred, 95),
            'p90_error': cls.compute_percentile_error(y_true, y_pred, 90),
            'median_error': np.median(np.abs(y_true - y_pred)),
            'std_error': np.std(y_true - y_pred)
        })
        
        return extended


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
    
    @staticmethod
    def compute_convergence_horizon(y_true: np.ndarray, y_pred: np.ndarray,
                                  window: int = 10, threshold: float = 0.05) -> float:
        """
        Compute convergence horizon - when predictions stabilize.
        
        Args:
            window: Window size for computing variance
            threshold: Variance threshold for convergence
        """
        if len(y_pred) < window:
            return len(y_pred)
        
        # Compute rolling variance of predictions
        variances = []
        for i in range(len(y_pred) - window + 1):
            window_variance = np.var(y_pred[i:i+window])
            variances.append(window_variance)
        
        # Find convergence point
        converged = np.array(variances) < threshold
        if converged.any():
            return np.argmax(converged)
        else:
            return len(y_pred)
    
    @staticmethod
    def compute_timeliness_score(y_true: np.ndarray, y_pred: np.ndarray,
                               early_factor: float = 0.5) -> float:
        """
        Compute timeliness score balancing accuracy and early warning.
        
        Args:
            early_factor: Weight for early predictions (0-1)
        """
        errors = y_pred - y_true
        
        # Separate early and late predictions
        early_mask = errors < 0
        late_mask = errors >= 0
        
        # Compute weighted score
        score = 0
        if early_mask.any():
            early_score = 1 - np.mean(np.abs(errors[early_mask]) / (y_true[early_mask] + 1e-8))
            score += early_factor * early_score * early_mask.sum() / len(errors)
        
        if late_mask.any():
            late_score = 1 - np.mean(np.abs(errors[late_mask]) / (y_true[late_mask] + 1e-8))
            score += (1 - early_factor) * late_score * late_mask.sum() / len(errors)
        
        return max(0, score)
    
    @classmethod
    def compute_rul_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all RUL-specific metrics."""
        base_metrics = cls."""
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