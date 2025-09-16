import pandas as pd

def load_file(file):
    """
    Load CSV or JSON file and return a pandas DataFrame.
    """
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.json'):
        df = pd.read_json(file)
    else:
        raise ValueError("Only CSV or JSON files are supported")
    return df

def process_dataframe(df, resample_rule='1min', fill_method='ffill'):
    """
    Process DataFrame: set timestamp index, sort, convert to UTC, 
    resample numeric columns, fill missing values.
    """
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    
    # Convert to datetime (coerce errors)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
    
    # Drop invalid timestamps
    df = df.dropna(subset=['timestamp'])
    
    # Convert to UTC
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')  # naive timestamps
    else:
        df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')    # already tz-aware
    
    # Set timestamp as index and sort
    df = df.set_index('timestamp').sort_index()
    
    # Resample numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df_resampled = df[numeric_cols].resample(resample_rule).mean()
    
    # Fill missing values
    if fill_method in ['ffill', 'bfill']:
        df_resampled = df_resampled.fillna(method=fill_method)
    
    # Reset index for final output
    df_resampled = df_resampled.reset_index()
    
    return df_resampled
