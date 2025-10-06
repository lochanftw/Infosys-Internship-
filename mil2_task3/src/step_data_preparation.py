"""
Step Data Preparation Module
Handles step count data loading and holidays DataFrame creation
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def create_holidays_dataframe():
    """
    Create holidays DataFrame for Prophet model
    
    Returns:
        pd.DataFrame: Holidays dataframe with Prophet format
    """
    print("Creating holidays DataFrame...")
    
    # Base start date (matches our data)
    start_date = datetime(2025, 6, 1)
    
    # Define special events with their date ranges and effects
    holidays = []
    
    # Vacation: Days 30-37 (June 30 - July 7, 2025)
    vacation_start = start_date + timedelta(days=29)  # Day 30 (0-indexed)
    for i in range(8):  # 8 days vacation
        holidays.append({
            'holiday': 'vacation',
            'ds': vacation_start + timedelta(days=i),
            'lower_window': 0,
            'upper_window': 0
        })
    
    # Sick Days: Days 60-62 (July 30 - August 1, 2025)
    sick_start = start_date + timedelta(days=59)  # Day 60 (0-indexed)
    for i in range(3):  # 3 sick days
        holidays.append({
            'holiday': 'sick',
            'ds': sick_start + timedelta(days=i),
            'lower_window': 0,
            'upper_window': 0
        })
    
    # Marathon: Day 90 (August 29, 2025)
    marathon_date = start_date + timedelta(days=89)  # Day 90 (0-indexed)
    holidays.append({
        'holiday': 'marathon',
        'ds': marathon_date,
        'lower_window': 0,
        'upper_window': 0
    })
    
    # Pre and post marathon recovery
    holidays.append({
        'holiday': 'marathon_recovery',
        'ds': marathon_date - timedelta(days=1),  # Day before
        'lower_window': 0,
        'upper_window': 0
    })
    holidays.append({
        'holiday': 'marathon_recovery',
        'ds': marathon_date + timedelta(days=1),  # Day after
        'lower_window': 0,
        'upper_window': 0
    })
    
    # Create DataFrame
    holidays_df = pd.DataFrame(holidays)
    
    print("✓ Holidays DataFrame created successfully")
    print(f"  - Total holiday events: {len(holidays_df)}")
    print(f"  - Vacation days: 8")
    print(f"  - Sick days: 3") 
    print(f"  - Marathon + recovery: 3")
    print(f"\nHolidays DataFrame structure:")
    print(holidays_df.head(10))
    print("\n")
    
    return holidays_df


def load_step_data(file_path='data/raw/step_count_data.csv'):
    """
    Load step count data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded step count data
    """
    print(f"Loading step count data from {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Step count data loaded successfully: {len(df)} rows")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        raise FileNotFoundError(f"Step count data file not found: {file_path}")


def format_for_prophet(df):
    """
    Format step count data for Prophet model
    
    Args:
        df (pd.DataFrame): Input dataframe with 'date' and 'step_count' columns
    
    Returns:
        pd.DataFrame: Formatted dataframe with 'ds' and 'y' columns
    """
    print("Formatting step count data for Prophet model...")
    
    # Prophet requires specific column names: 'ds' (datestamp) and 'y' (value)
    prophet_df = pd.DataFrame({
        'ds': df['date'],
        'y': df['step_count']
    })
    
    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
    
    print(f"✓ Step count data formatted successfully")
    print(f"  - Shape: {prophet_df.shape}")
    print(f"  - Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(f"  - Step range: {prophet_df['y'].min():,} - {prophet_df['y'].max():,} steps\n")
    
    return prophet_df


def save_processed_data(df, file_path='data/processed/formatted_step_data.csv'):
    """Save processed data to CSV"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"✓ Processed step data saved to {file_path}\n")


def prepare_step_data():
    """
    Main function to prepare step data and holidays for Prophet model
    
    Returns:
        tuple: (formatted_dataframe, holidays_dataframe)
    """
    print("=" * 50)
    print("STEP 1: STEP DATA & HOLIDAYS PREPARATION")
    print("=" * 50 + "\n")
    
    # Load step count data
    raw_df = load_step_data()
    
    # Format for Prophet
    prophet_df = format_for_prophet(raw_df)
    
    # Create holidays DataFrame
    holidays_df = create_holidays_dataframe()
    
    # Save processed data
    save_processed_data(prophet_df)
    
    return prophet_df, holidays_df


if __name__ == "__main__":
    # Test the module
    df, holidays_df = prepare_step_data()
    print("Step data preparation test completed!")
