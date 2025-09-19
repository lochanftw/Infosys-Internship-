"""
Multi-Timezone Processor for Task 3
Handles fitness data from users in different timezones
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Timezone mapping for common locations
TIMEZONE_MAPPING = {
    'New York': 'America/New_York',
    'NYC': 'America/New_York',
    'Tokyo': 'Asia/Tokyo',
    'Japan': 'Asia/Tokyo',
    'London': 'Europe/London',
    'UK': 'Europe/London',
    'Los Angeles': 'America/Los_Angeles',
    'LA': 'America/Los_Angeles',
    'California': 'America/Los_Angeles',
    'Berlin': 'Europe/Berlin',
    'Germany': 'Europe/Berlin',
    'Sydney': 'Australia/Sydney',
    'Australia': 'Australia/Sydney',
    'Mumbai': 'Asia/Kolkata',
    'India': 'Asia/Kolkata'
}

def detect_and_normalize_timestamps(df: pd.DataFrame, user_location: Optional[str] = None) -> pd.DataFrame:
    """
    Automatically detect timezone and normalize to UTC
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw fitness data with timestamp column
    user_location : str, optional
        User's primary location (e.g., 'New York', 'London')
    
    Returns:
    --------
    pandas.DataFrame
        Data with normalized UTC timestamps
    """
    
    logger.info(f"🔄 Processing timestamps for {len(df)} records")
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(df_processed)
    if not timestamp_col:
        logger.error("❌ No timestamp column found")
        return df_processed
    
    logger.info(f"📅 Found timestamp column: {timestamp_col}")
    
    # Convert to datetime if not already
    df_processed[timestamp_col] = pd.to_datetime(df_processed[timestamp_col], errors='coerce')
    
    # Store original timestamps for comparison
    df_processed['original_timestamp'] = df_processed[timestamp_col].copy()
    
    # Detect timezone if not provided
    if user_location:
        timezone_str = TIMEZONE_MAPPING.get(user_location, user_location)
        logger.info(f"🌍 Using provided location: {user_location} -> {timezone_str}")
    else:
        timezone_str = _auto_detect_timezone(df_processed, timestamp_col)
        logger.info(f"🔍 Auto-detected timezone: {timezone_str}")
    
    # Normalize to UTC
    df_processed = _normalize_to_utc(df_processed, timestamp_col, timezone_str)
    
    # Add metadata
    df_processed['source_timezone'] = timezone_str
    df_processed['processing_timestamp'] = datetime.utcnow()
    
    logger.info(f"✅ Successfully normalized {len(df_processed)} timestamps to UTC")
    
    return df_processed

def _find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Find the timestamp column in the DataFrame"""
    
    timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'created_at', 'recorded_at']
    
    # Check exact matches first
    for col in df.columns:
        if col.lower() in timestamp_candidates:
            return col
    
    # Check partial matches
    for col in df.columns:
        if any(candidate in col.lower() for candidate in timestamp_candidates):
            return col
    
    # Check data types
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    return None

def _auto_detect_timezone(df: pd.DataFrame, timestamp_col: str) -> str:
    """
    Auto-detect timezone from data patterns
    """
    
    # Check if timestamps already have timezone info
    sample_ts = df[timestamp_col].dropna().iloc[0]
    if hasattr(sample_ts, 'tz') and sample_ts.tz is not None:
        return str(sample_ts.tz)
    
    # Analyze hour distribution to guess timezone
    if not df[timestamp_col].dt.tz:
        hours = df[timestamp_col].dt.hour
        
        # Business hours pattern detection (9 AM - 5 PM peak)
        business_hours_count = len(hours[(hours >= 9) & (hours <= 17)])
        total_hours = len(hours.dropna())
        
        if business_hours_count / total_hours > 0.6:
            # Likely local timezone data - default to UTC for processing
            logger.info("📊 Detected business hours pattern - assuming local time")
            return 'UTC'
    
    # Default fallback
    return 'UTC'

def _normalize_to_utc(df: pd.DataFrame, timestamp_col: str, source_timezone: str) -> pd.DataFrame:
    """
    Normalize timestamps to UTC
    """
    
    try:
        if source_timezone == 'UTC':
            # Already UTC or naive timestamps treated as UTC
            if df[timestamp_col].dt.tz is None:
                df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
            return df
        
        # Get timezone object
        tz = pytz.timezone(source_timezone)
        
        if df[timestamp_col].dt.tz is None:
            # Naive timestamps - localize first
            df[timestamp_col] = df[timestamp_col].dt.tz_localize(tz, ambiguous='infer')
        else:
            # Already timezone-aware - convert
            df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)
        
        # Convert to UTC
        df[timestamp_col] = df[timestamp_col].dt.tz_convert('UTC')
        
        logger.info(f"🌍 Converted from {source_timezone} to UTC")
        
    except Exception as e:
        logger.error(f"❌ Timezone conversion failed: {e}")
        # Fallback - treat as UTC
        if df[timestamp_col].dt.tz is None:
            df[timestamp_col] = df[timestamp_col].dt.tz_localize('UTC')
    
    return df

def process_traveling_user_data(df: pd.DataFrame, travel_schedule: List[Dict]) -> pd.DataFrame:
    """
    Process data from a traveling user with timezone changes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Fitness data with timestamps
    travel_schedule : List[Dict]
        List of {'start_date': datetime, 'end_date': datetime, 'timezone': str}
    """
    
    logger.info(f"✈️ Processing travel data with {len(travel_schedule)} timezone changes")
    
    df_travel = df.copy()
    timestamp_col = _find_timestamp_column(df_travel)
    
    if not timestamp_col:
        return df_travel
    
    # Convert to datetime
    df_travel[timestamp_col] = pd.to_datetime(df_travel[timestamp_col])
    df_travel['travel_timezone'] = 'Unknown'
    
    # Apply timezone based on travel schedule
    for period in travel_schedule:
        mask = (df_travel[timestamp_col] >= period['start_date']) & \
               (df_travel[timestamp_col] <= period['end_date'])
        
        df_travel.loc[mask, 'travel_timezone'] = period['timezone']
        
        # Normalize each segment
        segment_data = df_travel[mask].copy()
        if len(segment_data) > 0:
            segment_normalized = detect_and_normalize_timestamps(
                segment_data, user_location=period['timezone']
            )
            df_travel.loc[mask, timestamp_col] = segment_normalized[timestamp_col]
    
    logger.info("✅ Travel timezone processing completed")
    return df_travel

def generate_timezone_report(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive timezone processing report
    """
    
    timestamp_col = _find_timestamp_column(df)
    
    report = {
        'total_records': len(df),
        'timestamp_column': timestamp_col,
        'timezone_info': {},
        'time_range': {},
        'data_quality': {}
    }
    
    if timestamp_col and timestamp_col in df.columns:
        timestamps = df[timestamp_col].dropna()
        
        report['time_range'] = {
            'start': timestamps.min(),
            'end': timestamps.max(),
            'duration_days': (timestamps.max() - timestamps.min()).days
        }
        
        if 'source_timezone' in df.columns:
            report['timezone_info'] = df['source_timezone'].value_counts().to_dict()
        
        report['data_quality'] = {
            'missing_timestamps': df[timestamp_col].isna().sum(),
            'duplicate_timestamps': df[timestamp_col].duplicated().sum(),
            'timezone_aware': bool(timestamps.dt.tz is not None if len(timestamps) > 0 else False)
        }
    
    return report

# Test function
def test_timezone_processor():
    """Test the timezone processor with sample data"""
    
    print("🧪 Testing Timezone Processor")
    print("=" * 40)
    
    # Create test data
    dates = pd.date_range('2024-03-10 06:00:00', '2024-03-12 18:00:00', freq='H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, len(dates)),
        'steps': np.random.poisson(100, len(dates))
    })
    
    print(f"📊 Created test data: {len(test_data)} records")
    print(f"Time range: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
    
    # Test timezone processing
    locations = ['New York', 'Tokyo', 'London']
    
    for location in locations:
        print(f"\n🌍 Testing {location}:")
        processed = detect_and_normalize_timestamps(test_data.copy(), user_location=location)
        
        original_times = processed['original_timestamp'].dt.strftime('%H:%M').value_counts().head(3)
        utc_times = processed['timestamp'].dt.strftime('%H:%M').value_counts().head(3)
        
        print(f"   Original times (top 3): {dict(original_times)}")
        print(f"   UTC times (top 3): {dict(utc_times)}")
        print(f"   Timezone: {processed['source_timezone'].iloc[0]}")

if __name__ == "__main__":
    test_timezone_processor()
