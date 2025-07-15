"""
Battery data loader for NASA battery dataset.
"""

import os
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BatteryData:
    """Container for battery cycle data."""
    cycle_data: pd.DataFrame
    metadata: Dict
    battery_id: str
    
    @property
    def num_cycles(self) -> int:
        return len(self.cycle_data)
    
    @property
    def capacity_fade(self) -> np.ndarray:
        """Calculate capacity fade percentage."""
        initial_capacity = self.cycle_data['capacity'].iloc[0]
        return (1 - self.cycle_data['capacity'] / initial_capacity) * 100


class NASABatteryDataLoader:
    """Load and preprocess NASA battery dataset."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.batteries = {}
        
    def load_battery_file(self, filepath: Path) -> BatteryData:
        """Load a single battery MAT file."""
        try:
            with h5py.File(filepath, 'r') as f:
                # Extract battery data structure
                battery_ref = f['B0005']  # Example battery
                
                # Initialize lists for cycle data
                cycles = []
                capacities = []
                voltages = []
                currents = []
                temperatures = []
                
                # Extract cycle data
                cycle_refs = battery_ref['cycle'][()]
                
                for cycle_ref in cycle_refs.flatten():
                    cycle_data = f[cycle_ref]
                    
                    # Extract cycle type and measurements
                    if 'type' in cycle_data:
                        cycle_type = ''.join(chr(c) for c in cycle_data['type'][()])
                        
                        if cycle_type == 'discharge':
                            # Extract discharge data
                            data = cycle_data['data'][()]
                            
                            # Voltage, Current, Temperature, Capacity
                            V = f[data[0, 0]][()]  # Voltage
                            I = f[data[1, 0]][()]  # Current
                            T = f[data[2, 0]][()]  # Temperature
                            
                            # Calculate capacity from current integration
                            time = np.arange(len(I)) * 1  # Assuming 1s sampling
                            capacity = np.trapz(I.flatten(), time) / 3600  # Ah
                            
                            cycles.append(len(cycles) + 1)
                            capacities.append(capacity)
                            voltages.append(np.mean(V))
                            currents.append(np.mean(I))
                            temperatures.append(np.mean(T))
                
                # Create DataFrame
                cycle_df = pd.DataFrame({
                    'cycle': cycles,
                    'capacity': capacities,
                    'voltage_mean': voltages,
                    'current_mean': currents,
                    'temperature_mean': temperatures
                })
                
                # Calculate additional features
                cycle_df['soh'] = cycle_df['capacity'] / cycle_df['capacity'].iloc[0]
                cycle_df['capacity_fade'] = 1 - cycle_df['soh']
                
                # Estimate RUL (cycles until 80% SOH)
                eol_capacity = cycle_df['capacity'].iloc[0] * 0.8
                eol_cycle = cycle_df[cycle_df['capacity'] <= eol_capacity]['cycle'].min()
                if pd.isna(eol_cycle):
                    eol_cycle = len(cycle_df) + 100  # Estimate if not reached
                
                cycle_df['rul'] = eol_cycle - cycle_df['cycle']
                
                metadata = {
                    'battery_id': filepath.stem,
                    'initial_capacity': cycle_df['capacity'].iloc[0],
                    'final_capacity': cycle_df['capacity'].iloc[-1],
                    'total_cycles': len(cycle_df),
                    'eol_cycle': eol_cycle
                }
                
                return BatteryData(
                    cycle_data=cycle_df,
                    metadata=metadata,
                    battery_id=filepath.stem
                )
                
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            raise
    
    def load_all_batteries(self) -> Dict[str, BatteryData]:
        """Load all battery files from the data directory."""
        battery_files = list(self.data_dir.glob("*.mat"))
        
        for filepath in battery_files:
            logger.info(f"Loading {filepath.name}...")
            battery_data = self.load_battery_file(filepath)
            self.batteries[battery_data.battery_id] = battery_data
        
        logger.info(f"Loaded {len(self.batteries)} batteries")
        return self.batteries
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """Combine all battery data into a single DataFrame."""
        all_data = []
        
        for battery_id, battery_data in self.batteries.items():
            df = battery_data.cycle_data.copy()
            df['battery_id'] = battery_id
            all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    def train_test_split(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets."""
        # Split by battery to avoid data leakage
        battery_ids = list(self.batteries.keys())
        np.random.seed(random_state)
        np.random.shuffle(battery_ids)
        
        n_test = int(len(battery_ids) * test_size)
        n_val = int(len(battery_ids) * val_size)
        
        test_ids = battery_ids[:n_test]
        val_ids = battery_ids[n_test:n_test + n_val]
        train_ids = battery_ids[n_test + n_val:]
        
        combined_df = self.get_combined_dataset()
        
        train_df = combined_df[combined_df['battery_id'].isin(train_ids)]
        val_df = combined_df[combined_df['battery_id'].isin(val_ids)]
        test_df = combined_df[combined_df['battery_id'].isin(test_ids)]
        
        logger.info(f"Train batteries: {len(train_ids)}, "
                   f"Val batteries: {len(val_ids)}, "
                   f"Test batteries: {len(test_ids)}")
        
        return train_df, val_df, test_df


class BatteryDataLoader:
    """High-level interface for battery data loading."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.nasa_loader = NASABatteryDataLoader(data_dir)
        
    def load_nasa_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load NASA battery data and return train/val/test splits."""
        self.nasa_loader.load_all_batteries()
        return self.nasa_loader.train_test_split()
    
    def get_battery_stats(self) -> pd.DataFrame:
        """Get statistics for all loaded batteries."""
        stats = []
        
        for battery_id, battery_data in self.nasa_loader.batteries.items():
            stats.append({
                'battery_id': battery_id,
                'num_cycles': battery_data.num_cycles,
                'initial_capacity': battery_data.metadata['initial_capacity'],
                'final_capacity': battery_data.metadata['final_capacity'],
                'capacity_retention': battery_data.metadata['final_capacity'] / 
                                    battery_data.metadata['initial_capacity'],
                'eol_cycle': battery_data.metadata['eol_cycle']
            })
        
        return pd.DataFrame(stats)


if __name__ == "__main__":
    # Example usage
    loader = BatteryDataLoader()
    train_df, val_df, test_df = loader.load_nasa_data()
    
    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    print("\nBattery Statistics:")
    print(loader.get_battery_stats())