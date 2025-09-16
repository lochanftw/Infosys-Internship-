# load_csv.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

print("Let's start building our data ingestion pipeline!")

# Ensure sample_data folder exists
if not os.path.exists('sample_data'):
    os.makedirs('sample_data')

def create_sample_heart_rate_csv():
    """Create sample heart rate data in CSV format"""
    
    start_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
    timestamps = []
    heart_rates = []
    
    for i in range(120):  # 2 hours of data at 1-min interval
        timestamp = start_time + timedelta(minutes=i)
        base_hr = 75 + np.sin(i/30) * 10  # simulate heart rate variation
        noise = np.random.normal(0, 5)   # random noise
        hr = max(50, min(120, base_hr + noise))  # constrain to realistic HR
        
        timestamps.append(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
        heart_rates.append(round(hr))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate_bpm': heart_rates
    })
    
    # Save to sample_data folder
    df.to_csv('sample_data/sample_heart_rate.csv', index=False)
    print("Created sample_heart_rate.csv in sample_data/")
    return df

def load_csv_data(file_path='sample_data/sample_heart_rate.csv'):
    """Load CSV data and standardize heart_rate column"""
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'heart_rate_bpm' in df.columns:
        df['heart_rate'] = df['heart_rate_bpm']
    df = df.sort_values('timestamp')
    return df

# Generate sample CSV and load it
hr_data = create_sample_heart_rate_csv()
df_csv = load_csv_data()
print(df_csv.head())
