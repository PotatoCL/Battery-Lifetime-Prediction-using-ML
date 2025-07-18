"""
Script to download and prepare the NASA battery dataset.

Note: You'll need to download the data manually from Kaggle:
https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset 

This script helps organize the downloaded CSV files.
"""

import os
import shutil
from pathlib import Path
import zipfile
import argparse
import pandas as pd
import numpy as np


def setup_data_directories():
    """Create necessary data directories."""
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'results',
        'results/figures',
        'models/checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def process_kaggle_csv(csv_path, output_dir='data/raw'):
    """Process the Kaggle CSV files and organize them."""
    output_path = Path(output_dir)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Identify unique batteries if there's a battery ID column
    if 'Battery_ID' in df.columns:
        battery_ids = df['Battery_ID'].unique()
        print(f"Found {len(battery_ids)} unique batteries")
        
        # Save each battery separately
        for battery_id in battery_ids:
            battery_data = df[df['Battery_ID'] == battery_id]
            battery_data.to_csv(output_path / f'{battery_id}.csv', index=False)
            print(f"Saved {battery_id}.csv with {len(battery_data)} records")
    else:
        # If no battery ID, save as single file
        df.to_csv(output_path / 'battery_data.csv', index=False)
        print(f"Saved battery_data.csv")
    
    return df


def extract_from_zip(zip_path, output_dir='data/raw'):
    """Extract CSV files from downloaded zip."""
    output_path = Path(output_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract only CSV files
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        
        for file in csv_files:
            # Extract to raw directory
            zip_ref.extract(file, output_path)
            print(f"Extracted: {file}")
            
            # If it's in a subdirectory, move it to raw
            extracted_path = output_path / file
            if extracted_path.parent != output_path:
                shutil.move(str(extracted_path), str(output_path / extracted_path.name))
    
    print(f"\nExtracted {len(csv_files)} CSV files to {output_dir}")


def preprocess_battery_data(raw_dir='data/raw', processed_dir='data/processed'):
    """Preprocess raw battery CSV data."""
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    csv_files = list(raw_path.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files to process")
    
    all_batteries = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # Standardize column names (adjust based on actual column names)
        column_mapping = {
            'Cycle_Index': 'cycle',
            'Discharge_Capacity(Ah)': 'capacity',
            'Voltage_measured': 'voltage_mean',
            'Current_measured': 'current_mean',
            'Temperature_measured': 'temperature_mean',
            'Battery_ID': 'battery_id'
        }
        
        # Rename columns if they exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Calculate derived features if not present
        if 'capacity' in df.columns and 'soh' not in df.columns:
            initial_capacity = df['capacity'].iloc[0]
            df['soh'] = df['capacity'] / initial_capacity
            df['capacity_fade'] = 1 - df['soh']
        
        # Estimate RUL if not present
        if 'rul' not in df.columns and 'soh' in df.columns:
            # Find cycle at 80% SOH
            eol_cycle = df[df['soh'] <= 0.8]['cycle'].min()
            if pd.isna(eol_cycle):
                eol_cycle = len(df) + 100  # Estimate
            df['rul'] = eol_cycle - df['cycle']
        
        # Add battery ID if not present
        if 'battery_id' not in df.columns:
            battery_id = csv_file.stem
            df['battery_id'] = battery_id
        
        # Save processed data
        output_file = processed_path / f"{df['battery_id'].iloc[0]}_processed.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved processed data: {output_file.name}")
        
        all_batteries.append(df)
    
    # Create combined dataset
    if all_batteries:
        combined = pd.concat(all_batteries, ignore_index=True)
        combined.to_csv(processed_path / 'all_batteries_combined.csv', index=False)
        print(f"\nCreated combined dataset with {len(combined)} total records")
        print(f"Batteries: {combined['battery_id'].nunique()}")
        print(f"Features: {combined.columns.tolist()}")


def create_sample_data(num_samples=3):
    """Create sample data for testing (if full dataset not available)."""
    print("Creating sample data for demonstration...")
    
    processed_path = Path('data/processed')
    
    for i in range(num_samples):
        battery_id = f'B000{i+5}'
        cycles = np.arange(1, 201)
        
        # Simulate battery degradation
        initial_capacity = 2.0 + np.random.normal(0, 0.1)
        capacity = initial_capacity * np.exp(-0.001 * cycles) + np.random.normal(0, 0.01, len(cycles))
        
        battery_data = pd.DataFrame({
            'cycle': cycles,
            'capacity': capacity,
            'voltage_mean': 3.7 - 0.0005 * cycles + np.random.normal(0, 0.02, len(cycles)),
            'current_mean': -2.0 + np.random.normal(0, 0.1, len(cycles)),
            'temperature_mean': 25 + np.random.normal(0, 2, len(cycles)),
            'battery_id': battery_id
        })
        
        # Add derived features
        battery_data['soh'] = battery_data['capacity'] / initial_capacity
        battery_data['capacity_fade'] = 1 - battery_data['soh']
        
        # Estimate RUL
        eol_cycle = battery_data[battery_data['soh'] <= 0.8]['cycle'].min()
        if pd.isna(eol_cycle):
            eol_cycle = 180
        battery_data['rul'] = eol_cycle - battery_data['cycle']
        battery_data.loc[battery_data['rul'] < 0, 'rul'] = 0
        
        # Save to processed directory
        battery_data.to_csv(processed_path / f'{battery_id}_processed.csv', index=False)
        print(f"Created sample data: {battery_id}")
    
    # Create combined dataset
    all_batteries = []
    for i in range(num_samples):
        battery_id = f'B000{i+5}'
        df = pd.read_csv(processed_path / f'{battery_id}_processed.csv')
        all_batteries.append(df)
    
    combined = pd.concat(all_batteries, ignore_index=True)
    combined.to_csv(processed_path / 'all_batteries_combined.csv', index=False)
    print(f"\nCreated combined dataset with {len(combined)} records")


def main():
    parser = argparse.ArgumentParser(description='Setup data for battery prediction project')
    parser.add_argument('--csv-path', type=str, help='Path to downloaded NASA battery CSV file')
    parser.add_argument('--zip-path', type=str, help='Path to downloaded zip file')
    parser.add_argument('--process', action='store_true', help='Process raw CSV files')
    parser.add_argument('--create-sample', action='store_true', 
                       help='Create sample data for testing')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_data_directories()
    
    if args.csv_path and os.path.exists(args.csv_path):
        # Process single CSV file
        df = process_kaggle_csv(args.csv_path)
        if args.process:
            preprocess_battery_data()
    elif args.zip_path and os.path.exists(args.zip_path):
        # Extract from zip
        extract_from_zip(args.zip_path)
        if args.process:
            preprocess_battery_data()
    elif args.process:
        # Process existing raw files
        preprocess_battery_data()
    elif args.create_sample:
        # Create sample data
        create_sample_data()
    else:
        print("\nUsage options:")
        print("1. Process a single CSV: python download_data.py --csv-path path/to/file.csv")
        print("2. Extract from zip: python download_data.py --zip-path path/to/file.zip")
        print("3. Process existing files: python download_data.py --process")
        print("4. Create sample data: python download_data.py --create-sample")
        print("\nDownload from: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset")


if __name__ == "__main__":
    main()