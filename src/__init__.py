"""
Battery performance prediction package.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from .data.loader import BatteryDataLoader
from .features.extractor import FeatureEngineering
from .models.factory import ModelFactory
from .evaluation.metrics import MultiTaskMetrics
from .visualization.plots import BatteryVisualizer

__all__ = [
    'BatteryDataLoader',
    'FeatureEngineering',
    'ModelFactory',
    'MultiTaskMetrics',
    'BatteryVisualizer'
]