"""
Model comparison utilities for battery performance prediction.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from .factory import ModelFactory
from .utils import count_parameters
from ..evaluation.metrics import MultiTaskMetrics


class ModelComparison:
    """Compare multiple models on battery prediction tasks."""
    
    def __init__(self, model_factory: ModelFactory):
        self.factory = model_factory
        self.results = {}
        self.models = {}
        
    def add_model(self, name: str, model):
        """Add a model to the comparison."""
        self.models[name] = model
        
    def compare_models(
        self,
        test_loader,
        device: str = 'cuda',
        tasks: List[str] = ['rul', 'soh', 'soc', 'capacity']
    ) -> pd.DataFrame:
        """Compare all models on test data."""
        metrics_calculator = MultiTaskMetrics()
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Move model to device
            model = model.to(device)
            model.eval()
            
            # Collect predictions
            all_predictions = {task: [] for task in tasks}
            all_targets = {task: [] for task in tasks}
            
            # Measure inference time
            inference_times = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
                    # Move batch to device
                    features = batch['features'].to(device)
                    
                    # Time inference
                    start_time = time.time()
                    predictions = model(features)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    # Collect predictions
                    for task in tasks:
                        if task in predictions:
                            all_predictions[task].extend(
                                predictions[task].cpu().numpy()
                            )
                        if task in batch:
                            all_targets[task].extend(
                                batch[task].cpu().numpy()
                            )
            
            # Convert to arrays
            for task in tasks:
                if all_predictions[task]:
                    all_predictions[task] = np.array(all_predictions[task])
                    all_targets[task] = np.array(all_targets[task])
            
            # Calculate metrics
            task_metrics = metrics_calculator.compute_task_metrics(
                all_predictions, all_targets
            )
            
            # Aggregate metrics
            aggregated = metrics_calculator.aggregate_metrics(task_metrics)
            
            # Add model-specific metrics
            aggregated['num_parameters'] = count_parameters(model)
            aggregated['avg_inference_time_ms'] = np.mean(inference_times) * 1000
            aggregated['total_inference_time_s'] = sum(inference_times)
            
            # Store results
            self.results[model_name] = {
                'metrics': aggregated,
                'predictions': all_predictions,
                'targets': all_targets,
                'task_metrics': task_metrics
            }
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        return comparison_df
    
    def plot_comparison(self, metrics: List[str] = None, save_path: Optional[Path] = None):
        """Plot model comparison."""
        if not self.results:
            raise ValueError("No results to plot. Run compare_models first.")
        
        # Default metrics to plot
        if metrics is None:
            metrics = ['overall_mae', 'overall_rmse', 'overall_r2', 
                      'num_parameters', 'avg_inference_time_ms']
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each metric
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            values = comparison_df[metric].values
            models = comparison_df.index.tolist()
            
            # Create bar plot
            bars = ax.bar(range(len(models)), values)
            
            # Color best performing model
            if 'r2' in metric or 'accuracy' in metric:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_color('green')
            
            # Customize plot
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.01 * max(values), f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        # Remove empty subplots
        for idx in range(len(available_metrics), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_predictions(self, model_name: str, task: str = 'soh', 
                        save_path: Optional[Path] = None):
        """Plot predictions vs actual for a specific model and task."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        predictions = self.results[model_name]['predictions'][task]
        targets = self.results[model_name]['targets'][task]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(targets, predictions, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        ax1.set_xlabel(f'True {task.upper()}')
        ax1.set_ylabel(f'Predicted {task.upper()}')
        ax1.set_title(f'{model_name} - {task.upper()} Predictions\nMAE: {mae:.3f}, R²: {r2:.3f}')
        ax1.grid(True, alpha=0.3)
        
        # Residual plot
        residuals = predictions - targets
        ax2.scatter(predictions, residuals, alpha=0.5, s=10)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        
        ax2.set_xlabel(f'Predicted {task.upper()}')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_report(self, save_path: Optional[Path] = None) -> str:
        """Create a summary report of model comparison."""
        if not self.results:
            raise ValueError("No results to summarize. Run compare_models first.")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        # Sort by overall performance
        comparison_df = comparison_df.sort_values('overall_mae')
        
        report = []
        report.append("# Battery Performance Prediction Model Comparison\n")
        report.append(f"Total models compared: {len(self.models)}\n")
        
        # Best model
        best_model = comparison_df.index[0]
        report.append(f"## Best Overall Model: {best_model}\n")
        
        # Summary table
        report.append("## Performance Summary\n")
        report.append("| Model | Overall MAE | Overall R² | Parameters | Inference (ms) |")
        report.append("|-------|-------------|------------|------------|----------------|")
        
        for model_name in comparison_df.index:
            mae = comparison_df.loc[model_name, 'overall_mae']
            r2 = comparison_df.loc[model_name, 'overall_r2']
            params = int(comparison_df.loc[model_name, 'num_parameters'])
            inference = comparison_df.loc[model_name, 'avg_inference_time_ms']
            
            report.append(f"| {model_name} | {mae:.3f} | {r2:.3f} | {params:,} | {inference:.2f} |")
        
        # Task-specific best models
        report.append("\n## Task-Specific Best Models\n")
        
        tasks = ['rul', 'soh', 'soc', 'capacity']
        for task in tasks:
            mae_col = f'{task}_mae'
            if mae_col in comparison_df.columns:
                best_task_model = comparison_df[mae_col].idxmin()
                best_mae = comparison_df.loc[best_task_model, mae_col]
                report.append(f"- **{task.upper()}**: {best_task_model} (MAE: {best_mae:.3f})")
        
        # Model complexity vs performance
        report.append("\n## Model Complexity Analysis\n")
        
        # Find most efficient model (best performance per parameter)
        comparison_df['efficiency'] = comparison_df['overall_mae'] / (comparison_df['num_parameters'] / 1e6)
        most_efficient = comparison_df['efficiency'].idxmin()
        
        report.append(f"Most parameter-efficient model: {most_efficient}")
        report.append(f"Fastest inference: {comparison_df['avg_inference_time_ms'].idxmin()}")
        
        report_text = '\n'.join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_complexity_vs_performance(self, save_path: Optional[Path] = None):
        """Plot model complexity vs performance."""
        if not self.results:
            raise ValueError("No results to plot. Run compare_models first.")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            model_name: result['metrics']
            for model_name, result in self.results.items()
        }).T
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameters vs MAE
        ax1.scatter(comparison_df['num_parameters'] / 1e6, 
                   comparison_df['overall_mae'], s=100)
        
        for idx, model_name in enumerate(comparison_df.index):
            ax1.annotate(model_name, 
                        (comparison_df.loc[model_name, 'num_parameters'] / 1e6,
                         comparison_df.loc[model_name, 'overall_mae']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Parameters (Millions)')
        ax1.set_ylabel('Overall MAE')
        ax1.set_title('Model Complexity vs Performance')
        ax1.grid(True, alpha=0.3)
        
        # Inference time vs MAE
        ax2.scatter(comparison_df['avg_inference_time_ms'], 
                   comparison_df['overall_mae'], s=100)
        
        for idx, model_name in enumerate(comparison_df.index):
            ax2.annotate(model_name, 
                        (comparison_df.loc[model_name, 'avg_inference_time_ms'],
                         comparison_df.loc[model_name, 'overall_mae']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Inference Time (ms)')
        ax2.set_ylabel('Overall MAE')
        ax2.set_title('Inference Speed vs Performance')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Create model factory
    factory = ModelFactory(Path(__file__).parent.parent.parent / 'configs' / 'model_config.yaml')
    
    # Create comparison object
    comparison = ModelComparison(factory)
    
    # Add models
    input_dim = 50
    for model_name in ['baseline', 'cp_gru', 'cp_lstm', 'cp_transformer']:
        model = factory.create_model(model_name, input_dim)
        comparison.add_model(model_name, model)
    
    print("Model comparison utilities loaded successfully!")
    print(f"Models ready for comparison: {list(comparison.models.keys())}")