"""
Feature extraction for battery performance prediction.
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BatteryFeatureExtractor:
    """Extract features from battery cycle data."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        
    def extract_statistical_features(self, data: pd.Series) -> Dict[str, float]:
        """Extract statistical features from a time series."""
        features = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'range': data.max() - data.min(),
            'skew': data.skew(),
            'kurtosis': data.kurtosis(),
            'cv': data.std() / data.mean() if data.mean() != 0 else 0
        }
        
        # Percentiles
        for p in [25, 50, 75]:
            features[f'percentile_{p}'] = data.quantile(p / 100)
        
        return features
    
    def extract_trend_features(self, cycles: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """Extract trend-based features."""
        features = {}
        
        # Linear regression
        if len(cycles) > 1:
            slope, intercept, r_value, _, _ = stats.linregress(cycles, values)
            features['linear_slope'] = slope
            features['linear_intercept'] = intercept
            features['linear_r2'] = r_value ** 2
        else:
            features['linear_slope'] = 0
            features['linear_intercept'] = values[0] if len(values) > 0 else 0
            features['linear_r2'] = 0
        
        # Polynomial fit (2nd order)
        if len(cycles) > 2:
            poly_coeffs = np.polyfit(cycles, values, 2)
            features['poly2_a'] = poly_coeffs[0]
            features['poly2_b'] = poly_coeffs[1]
            features['poly2_c'] = poly_coeffs[2]
        else:
            features['poly2_a'] = 0
            features['poly2_b'] = 0
            features['poly2_c'] = values[0] if len(values) > 0 else 0
        
        return features
    
    def extract_degradation_features(self, capacity_data: pd.Series) -> Dict[str, float]:
        """Extract degradation-specific features."""
        features = {}
        
        if len(capacity_data) < 2:
            return {
                'capacity_fade_rate': 0,
                'capacity_fade_acceleration': 0,
                'cycles_to_80_soh': 0,
                'degradation_knee': 0
            }
        
        # Capacity fade rate
        initial_capacity = capacity_data.iloc[0]
        capacity_fade = (initial_capacity - capacity_data) / initial_capacity
        
        # Average fade rate
        features['capacity_fade_rate'] = capacity_fade.iloc[-1] / len(capacity_data)
        
        # Fade acceleration (2nd derivative)
        if len(capacity_data) > 2:
            fade_diff = np.diff(capacity_fade)
            features['capacity_fade_acceleration'] = np.mean(np.diff(fade_diff))
        else:
            features['capacity_fade_acceleration'] = 0
        
        # Cycles to 80% SOH
        soh = capacity_data / initial_capacity
        cycles_80 = soh[soh <= 0.8].index
        features['cycles_to_80_soh'] = cycles_80[0] if len(cycles_80) > 0 else len(soh) * 2
        
        # Find degradation knee (point of maximum curvature)
        if len(capacity_data) > 10:
            # Smooth the data
            smoothed = signal.savgol_filter(capacity_data.values, 
                                           min(11, len(capacity_data) // 2 * 2 + 1), 3)
            
            # Calculate curvature
            dx = 1
            dy = np.gradient(smoothed, dx)
            d2y = np.gradient(dy, dx)
            curvature = np.abs(d2y) / (1 + dy**2)**1.5
            
            # Find knee point
            knee_idx = np.argmax(curvature)
            features['degradation_knee'] = capacity_data.index[knee_idx]
        else:
            features['degradation_knee'] = len(capacity_data) // 2
        
        return features
    
    def extract_voltage_features(self, voltage_data: pd.Series) -> Dict[str, float]:
        """Extract voltage-specific features."""
        features = self.extract_statistical_features(voltage_data)
        
        # Voltage stability
        if len(voltage_data) > 1:
            features['voltage_stability'] = 1 / (1 + voltage_data.std())
            
            # Voltage drop over cycles
            features['voltage_drop_rate'] = (voltage_data.iloc[0] - voltage_data.iloc[-1]) / len(voltage_data)
        else:
            features['voltage_stability'] = 1
            features['voltage_drop_rate'] = 0
        
        return features
    
    def extract_temperature_features(self, temp_data: pd.Series) -> Dict[str, float]:
        """Extract temperature-specific features."""
        features = self.extract_statistical_features(temp_data)
        
        # Temperature variability
        features['temp_variability'] = temp_data.std() / temp_data.mean() if temp_data.mean() != 0 else 0
        
        # Temperature anomalies (values > 2 std from mean)
        if len(temp_data) > 3:
            anomaly_threshold = temp_data.mean() + 2 * temp_data.std()
            features['temp_anomaly_ratio'] = (temp_data > anomaly_threshold).sum() / len(temp_data)
        else:
            features['temp_anomaly_ratio'] = 0
        
        return features
    
    def extract_cycle_features(self, battery_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features for each cycle."""
        feature_dfs = []
        
        # Rolling window features
        for i in range(len(battery_df)):
            features = {'cycle': battery_df.iloc[i]['cycle']}
            
            # Get window data
            start_idx = max(0, i - self.window_size + 1)
            window_data = battery_df.iloc[start_idx:i+1]
            
            # Current cycle features
            features.update({
                'capacity_current': battery_df.iloc[i]['capacity'],
                'voltage_current': battery_df.iloc[i]['voltage_mean'],
                'current_current': battery_df.iloc[i]['current_mean'],
                'temperature_current': battery_df.iloc[i]['temperature_mean'],
                'soh_current': battery_df.iloc[i]['soh'],
                'rul_current': battery_df.iloc[i]['rul']
            })
            
            # Window-based statistical features
            prefix = f'window{self.window_size}_'
            
            # Capacity features
            cap_features = self.extract_statistical_features(window_data['capacity'])
            features.update({f'{prefix}capacity_{k}': v for k, v in cap_features.items()})
            
            # Voltage features
            volt_features = self.extract_voltage_features(window_data['voltage_mean'])
            features.update({f'{prefix}voltage_{k}': v for k, v in volt_features.items()})
            
            # Temperature features
            temp_features = self.extract_temperature_features(window_data['temperature_mean'])
            features.update({f'{prefix}temperature_{k}': v for k, v in temp_features.items()})
            
            # Trend features
            cycles = window_data.index.values
            trend_features = self.extract_trend_features(cycles, window_data['capacity'].values)
            features.update({f'{prefix}trend_{k}': v for k, v in trend_features.items()})
            
            # Degradation features
            deg_features = self.extract_degradation_features(window_data['capacity'])
            features.update({f'{prefix}degradation_{k}': v for k, v in deg_features.items()})
            
            feature_dfs.append(features)
        
        return pd.DataFrame(feature_dfs)
    
    def extract_battery_features(self, battery_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for an entire battery."""
        # Extract cycle-level features
        cycle_features = self.extract_cycle_features(battery_df)
        
        # Add battery-level features
        battery_features = {
            'initial_capacity': battery_df['capacity'].iloc[0],
            'total_cycles': len(battery_df),
            'capacity_retention': battery_df['capacity'].iloc[-1] / battery_df['capacity'].iloc[0]
        }
        
        for col in cycle_features.columns:
            if col != 'cycle':
                cycle_features[f'battery_{col}'] = battery_features.get(col, np.nan)
        
        return cycle_features


class FeatureEngineering:
    """High-level feature engineering interface."""
    
    def __init__(self, window_sizes: List[int] = [5, 10, 20]):
        self.extractors = [BatteryFeatureExtractor(ws) for ws in window_sizes]
        self.window_sizes = window_sizes
        
    def engineer_features(self, battery_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features using multiple window sizes."""
        all_features = []
        
        for extractor in self.extractors:
            features = extractor.extract_battery_features(battery_df)
            all_features.append(features)
        
        # Merge features from different window sizes
        merged_features = all_features[0]
        for features in all_features[1:]:
            # Merge on cycle, avoiding duplicate columns
            cols_to_add = [col for col in features.columns if col not in merged_features.columns or col == 'cycle']
            merged_features = pd.merge(merged_features, features[cols_to_add], on='cycle', how='outer')
        
        # Add interaction features
        merged_features = self._add_interaction_features(merged_features)
        
        # Add lagged features
        merged_features = self._add_lagged_features(merged_features)
        
        return merged_features
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables."""
        # Capacity-voltage interaction
        if 'capacity_current' in df.columns and 'voltage_current' in df.columns:
            df['capacity_voltage_product'] = df['capacity_current'] * df['voltage_current']
            df['capacity_voltage_ratio'] = df['capacity_current'] / (df['voltage_current'] + 1e-6)
        
        # Temperature effects
        if 'temperature_current' in df.columns and 'capacity_current' in df.columns:
            df['temp_capacity_interaction'] = df['temperature_current'] * df['capacity_current']
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Add lagged features."""
        key_features = ['capacity_current', 'soh_current', 'voltage_current']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        
        # Fill NaN values with forward fill
        df = df.fillna(method='ffill').fillna(0)
        
        return df


if __name__ == "__main__":
    # Example usage
    from src.data.loader import BatteryDataLoader
    
    loader = BatteryDataLoader()
    batteries = loader.nasa_loader.load_all_batteries()
    
    # Extract features for one battery
    battery_id = list(batteries.keys())[0]
    battery_data = batteries[battery_id]
    
    # Feature engineering
    fe = FeatureEngineering()
    features = fe.engineer_features(battery_data.cycle_data)
    
    print(f"Extracted {len(features.columns)} features")
    print(f"Feature shape: {features.shape}")
    print("\nSample features:")
    print(features.head())