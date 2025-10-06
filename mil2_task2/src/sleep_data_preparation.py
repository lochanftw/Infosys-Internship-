"""
Sleep Data Preparation Module
Handles sleep data loading, formatting, and preprocessing for Prophet
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def load_sleep_data(file_path='data/raw/sleep_data.csv'):
    """
    Load sleep data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded sleep data
    """
    print(f"Loading sleep data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Sleep data loaded successfully: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        raise FileNotFoundError(f"Sleep data file not found: {file_path}")


def format_for_prophet(df):
    """
    Format sleep data for Prophet model (requires 'ds' and 'y' columns)
    
    Args:
        df (pd.DataFrame): Input dataframe with 'date' and 'sleep_hours' columns
    
    Returns:
        pd.DataFrame: Formatted dataframe with 'ds' and 'y' columns
    """
    print("Formatting sleep data for Prophet model...")
    
    # Prophet requires specific column names: 'ds' (datestamp) and 'y' (value)
    prophet_df = pd.DataFrame({
        'ds': df['date'],
        'y': df['sleep_hours']
    })
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    # Check for missing values
    if prophet_df.isnull().any().any():
        print("  ⚠ Warning: Missing values detected, filling with forward fill...")
        prophet_df = prophet_df.fillna(method='ffill')
    
    print(f"✓ Sleep data formatted successfully")
    print(f"  - Shape: {prophet_df.shape}")
    print(f"  - Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(f"  - Sleep range: {prophet_df['y'].min():.1f} - {prophet_df['y'].max():.1f} hours\n")
    
    return prophet_df


def save_processed_data(df, file_path='data/processed/formatted_sleep_data.csv'):
    """
    Save processed sleep data to CSV
    
    Args:
        df (pd.DataFrame): Processed dataframe
        file_path (str): Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df.to_csv(file_path, index=False)
    print(f"✓ Processed sleep data saved to {file_path}\n")


def prepare_sleep_data():
    """
    Main function to prepare sleep data for Prophet model
    
    Returns:
        pd.DataFrame: Formatted dataframe ready for Prophet
    """
    print("=" * 50)
    print("STEP 1: SLEEP DATA PREPARATION")
    print("=" * 50 + "\n")
    
    # Load sleep data
    raw_df = load_sleep_data()
    
    # Format for Prophet
    prophet_df = format_for_prophet(raw_df)
    
    # Save processed data
    save_processed_data(prophet_df)
    
    return prophet_df


if __name__ == "__main__":
    # Test the module
    df = prepare_sleep_data()
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nLast 5 rows:")
    print(df.tail())
