"""
Data Preparation Module
Handles data loading, formatting, and preprocessing for Prophet model
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def generate_sample_data(n_days=60, save_path='data/raw/heart_rate_data.csv'):
    """
    Generate sample heart rate data for demonstration
    
    Args:
        n_days (int): Number of days of data to generate
        save_path (str): Path to save the CSV file
    
    Returns:
        pd.DataFrame: Generated heart rate data
    """
    print(f"Generating {n_days} days of sample heart rate data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Starting date
    start_date = datetime(2025, 8, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]
    
    # Generate realistic heart rate data with patterns
    base_heart_rate = 72  # Average resting heart rate
    daily_variation = np.random.normal(0, 3, n_days)  # Daily random variation
    weekly_pattern = np.sin(np.arange(n_days) * 2 * np.pi / 7) * 2  # Weekly cycle
    trend = np.linspace(0, -2, n_days)  # Slight decreasing trend
    
    heart_rates = base_heart_rate + daily_variation + weekly_pattern + trend
    
    # Add occasional spikes (exercise days)
    exercise_days = np.random.choice(n_days, n_days // 5, replace=False)
    heart_rates[exercise_days] += np.random.uniform(3, 8, len(exercise_days))
    
    # Ensure realistic range (60-85 bpm)
    heart_rates = np.clip(heart_rates, 60, 85)
    heart_rates = np.round(heart_rates, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'average_heart_rate': heart_rates
    })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"✓ Data saved to {save_path}")
    print(f"  - Total days: {len(df)}")
    print(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  - Mean heart rate: {df['average_heart_rate'].mean():.1f} bpm\n")
    
    return df


def load_raw_data(file_path='data/raw/heart_rate_data.csv'):
    """
    Load raw heart rate data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded heart rate data
    """
    print(f"Loading data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Data loaded successfully: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        print("  Generating sample data instead...")
        return generate_sample_data(save_path=file_path)


def format_for_prophet(df):
    """
    Format data for Prophet model (requires 'ds' and 'y' columns)
    
    Args:
        df (pd.DataFrame): Input dataframe with 'date' and 'average_heart_rate' columns
    
    Returns:
        pd.DataFrame: Formatted dataframe with 'ds' and 'y' columns
    """
    print("Formatting data for Prophet model...")
    
    # Prophet requires specific column names: 'ds' (datestamp) and 'y' (value)
    prophet_df = pd.DataFrame({
        'ds': df['date'],
        'y': df['average_heart_rate']
    })
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Check for missing values
    if prophet_df.isnull().any().any():
        print("  ⚠ Warning: Missing values detected, filling with forward fill...")
        prophet_df = prophet_df.fillna(method='ffill')
    
    print(f"✓ Data formatted successfully")
    print(f"  - Shape: {prophet_df.shape}")
    print(f"  - Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(f"  - Value range: {prophet_df['y'].min():.1f} - {prophet_df['y'].max():.1f} bpm\n")
    
    return prophet_df


def save_processed_data(df, file_path='data/processed/formatted_data.csv'):
    """
    Save processed data to CSV
    
    Args:
        df (pd.DataFrame): Processed dataframe
        file_path (str): Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"✓ Processed data saved to {file_path}\n")


def prepare_data():
    """
    Main function to prepare data for Prophet model
    
    Returns:
        pd.DataFrame: Formatted dataframe ready for Prophet
    """
    print("=" * 50)
    print("STEP 1: DATA PREPARATION")
    print("=" * 50 + "\n")
    
    # Load raw data (or generate if not exists)
    raw_df = load_raw_data()
    
    # Format for Prophet
    prophet_df = format_for_prophet(raw_df)
    
    # Save processed data
    save_processed_data(prophet_df)
    
    return prophet_df


if __name__ == "__main__":
    # Test the module
    df = prepare_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
