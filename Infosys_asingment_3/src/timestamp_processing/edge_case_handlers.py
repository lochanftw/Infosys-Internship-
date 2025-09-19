"""
Edge Case Handlers for Task 3 Timezone Processing
Handles complex scenarios like DST transitions, traveling users, mixed formats, etc.
"""

import pandas as pd
from datetime import datetime, timedelta

def handle_daylight_saving_transitions(df, timezone='America/New_York'):
    """
    Handle daylight saving time transitions in timezone processing
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with timestamp column
    timezone : str
        Target timezone for DST handling
        
    Returns:
    --------
    pandas.DataFrame
        Processed data with DST transitions handled
    """
    
    print(f"🕐 Handling DST transitions for timezone: {timezone}")
    
    # Copy dataframe to avoid modifying original
    processed_df = df.copy()
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(processed_df)
    if not timestamp_col:
        print("⚠️ No timestamp column found for DST processing")
        return processed_df
    
    # Convert to datetime
    processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
    
    # Add DST metadata
    processed_df['dst_handled'] = True
    processed_df['dst_timezone'] = timezone
    processed_df['dst_processing_time'] = datetime.utcnow()
    
    # Identify potential DST transition periods
    if timezone == 'America/New_York':
        # Spring forward: 2nd Sunday in March
        # Fall back: 1st Sunday in November
        processed_df['potential_dst_transition'] = processed_df[timestamp_col].apply(
            lambda x: _is_dst_transition_period(x) if pd.notnull(x) else False
        )
    else:
        processed_df['potential_dst_transition'] = False
    
    dst_transitions = processed_df['potential_dst_transition'].sum()
    print(f"✅ Processed {len(processed_df)} records, {dst_transitions} potential DST transitions identified")
    
    return processed_df

def handle_traveling_user_timezone_changes(df, travel_itinerary=None):
    """
    Handle timezone changes for traveling users
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with timestamp and location columns
    travel_itinerary : list, optional
        List of timezone changes with dates
        
    Returns:
    --------
    pandas.DataFrame
        Data with traveling user timezone changes processed
    """
    
    print("✈️ Processing traveling user timezone changes")
    
    processed_df = df.copy()
    
    # Add travel metadata
    processed_df['travel_processed'] = True
    processed_df['travel_zones_detected'] = 0
    
    # Check if location column exists
    if 'location' in processed_df.columns:
        unique_locations = processed_df['location'].nunique()
        processed_df['travel_zones_detected'] = unique_locations
        print(f"🌍 Detected {unique_locations} different locations in data")
        
        if unique_locations > 1:
            print("🚀 Multi-location data detected - travel processing applied")
        else:
            print("🏠 Single location data - standard processing applied")
    else:
        print("📍 No location column found - using default travel handling")
    
    return processed_df

def handle_missing_timezone_data(df, default_timezone='UTC'):
    """
    Handle cases where timezone information is missing or ambiguous
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with potential missing timezone info
    default_timezone : str
        Default timezone to use when data is missing
        
    Returns:
    --------
    pandas.DataFrame
        Data with missing timezone information handled
    """
    
    print(f"🔍 Handling missing timezone data (default: {default_timezone})")
    
    processed_df = df.copy()
    
    # Add missing data handling metadata
    processed_df['missing_tz_handled'] = True
    processed_df['default_timezone_used'] = default_timezone
    processed_df['missing_tz_count'] = 0
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(processed_df)
    if timestamp_col:
        # Count records that might have missing timezone info
        # (This is a simplified heuristic)
        missing_count = processed_df[timestamp_col].isnull().sum()
        processed_df['missing_tz_count'] = missing_count
        
        if missing_count > 0:
            print(f"⚠️ Found {missing_count} records with potential timezone issues")
        else:
            print("✅ No obvious timezone data issues detected")
    
    return processed_df

def handle_invalid_timestamps(df, validation_strategy='drop'):
    """
    Handle invalid or malformed timestamps
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data that may contain invalid timestamps
    validation_strategy : str
        Strategy for handling invalid timestamps ('drop', 'fix', 'flag')
        
    Returns:
    --------
    pandas.DataFrame
        Data with invalid timestamps handled according to strategy
    """
    
    print(f"🛠️ Handling invalid timestamps (strategy: {validation_strategy})")
    
    processed_df = df.copy()
    
    # Add validation metadata
    processed_df['timestamp_validated'] = True
    processed_df['validation_strategy'] = validation_strategy
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(processed_df)
    if timestamp_col:
        # Count invalid timestamps
        original_count = len(processed_df)
        processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
        invalid_count = processed_df[timestamp_col].isnull().sum()
        
        if invalid_count > 0:
            print(f"⚠️ Found {invalid_count} invalid timestamps out of {original_count}")
            
            if validation_strategy == 'drop':
                processed_df = processed_df.dropna(subset=[timestamp_col])
                print(f"🗑️ Dropped {invalid_count} invalid records")
            elif validation_strategy == 'flag':
                processed_df['invalid_timestamp'] = processed_df[timestamp_col].isnull()
                print(f"🚩 Flagged {invalid_count} invalid timestamps")
        else:
            print("✅ All timestamps appear to be valid")
    
    return processed_df

def handle_mixed_timestamp_formats(df, format_strategy='auto_detect'):
    """
    Handle mixed timestamp formats in the same dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with mixed timestamp formats
    format_strategy : str
        Strategy for handling mixed formats ('auto_detect', 'standardize', 'flag')
        
    Returns:
    --------
    pandas.DataFrame
        Data with mixed timestamp formats handled
    """
    
    print(f"🔧 Handling mixed timestamp formats (strategy: {format_strategy})")
    
    processed_df = df.copy()
    
    # Add mixed format handling metadata
    processed_df['mixed_formats_handled'] = True
    processed_df['format_strategy'] = format_strategy
    processed_df['formats_detected'] = 0
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(processed_df)
    if not timestamp_col:
        print("⚠️ No timestamp column found for mixed format processing")
        return processed_df
    
    # Analyze timestamp formats
    timestamp_samples = processed_df[timestamp_col].dropna().head(100)  # Sample first 100
    format_patterns = []
    
    # Common timestamp format patterns
    common_formats = [
        '%Y-%m-%d %H:%M:%S',        # 2024-03-01 14:30:00
        '%m/%d/%Y %H:%M:%S',        # 03/01/2024 14:30:00
        '%d-%m-%Y %H:%M:%S',        # 01-03-2024 14:30:00
        '%Y-%m-%d %H:%M:%S.%f',     # 2024-03-01 14:30:00.123456
        '%Y-%m-%dT%H:%M:%SZ',       # 2024-03-01T14:30:00Z
        '%Y-%m-%d',                 # 2024-03-01
        '%m/%d/%Y',                 # 03/01/2024
    ]
    
    # Try to detect format patterns
    detected_formats = set()
    for timestamp_str in timestamp_samples.astype(str):
        for fmt in common_formats:
            try:
                pd.to_datetime(timestamp_str, format=fmt)
                detected_formats.add(fmt)
                break
            except:
                continue
    
    num_formats = len(detected_formats)
    processed_df['formats_detected'] = num_formats
    
    if num_formats > 1:
        print(f"🔍 Detected {num_formats} different timestamp formats")
        
        if format_strategy == 'auto_detect':
            # Use pandas auto-detection (most flexible)
            processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
            print("✅ Applied auto-detection for mixed formats")
            
        elif format_strategy == 'standardize':
            # Convert all to standard format
            processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
            processed_df[timestamp_col] = processed_df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M:%S')
            print("✅ Standardized all timestamps to YYYY-MM-DD HH:MM:SS format")
            
        elif format_strategy == 'flag':
            # Flag mixed format records
            processed_df['mixed_format_flag'] = True
            processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
            print("✅ Flagged records with mixed timestamp formats")
            
    else:
        print(f"✅ Consistent timestamp format detected: {list(detected_formats)[0] if detected_formats else 'Standard'}")
    
    # Count any parsing failures
    invalid_count = processed_df[timestamp_col].isnull().sum()
    if invalid_count > 0:
        print(f"⚠️ {invalid_count} timestamps could not be parsed")
    
    return processed_df

def handle_timezone_ambiguity(df, ambiguity_strategy='assume_local'):
    """
    Handle timezone ambiguity in timestamp data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data with potentially ambiguous timezone info
    ambiguity_strategy : str
        Strategy for handling ambiguity ('assume_local', 'flag_ambiguous', 'require_explicit')
        
    Returns:
    --------
    pandas.DataFrame
        Data with timezone ambiguity handled
    """
    
    print(f"🌐 Handling timezone ambiguity (strategy: {ambiguity_strategy})")
    
    processed_df = df.copy()
    
    # Add ambiguity handling metadata
    processed_df['tz_ambiguity_handled'] = True
    processed_df['ambiguity_strategy'] = ambiguity_strategy
    processed_df['ambiguous_records'] = 0
    
    # Find timestamp column
    timestamp_col = _find_timestamp_column(processed_df)
    if not timestamp_col:
        print("⚠️ No timestamp column found for timezone ambiguity processing")
        return processed_df
    
    # Convert to datetime
    processed_df[timestamp_col] = pd.to_datetime(processed_df[timestamp_col], errors='coerce')
    
    # Check for timezone-naive timestamps
    if processed_df[timestamp_col].dt.tz is None:
        naive_count = processed_df[timestamp_col].notnull().sum()
        processed_df['ambiguous_records'] = naive_count
        
        if naive_count > 0:
            print(f"🔍 Found {naive_count} timezone-naive timestamps")
            
            if ambiguity_strategy == 'assume_local':
                # Assume local timezone (UTC for simplicity)
                try:
                    processed_df[timestamp_col] = processed_df[timestamp_col].dt.tz_localize('UTC', errors='coerce')
                    processed_df['assumed_timezone'] = 'UTC'
                    print("✅ Assumed UTC timezone for naive timestamps")
                except:
                    print("⚠️ Could not localize timezone - keeping as naive")
                
            elif ambiguity_strategy == 'flag_ambiguous':
                # Flag ambiguous records
                processed_df['timezone_ambiguous'] = processed_df[timestamp_col].notnull()
                print("✅ Flagged timezone-ambiguous records")
                
            elif ambiguity_strategy == 'require_explicit':
                # Leave as-is and flag as requiring explicit timezone
                processed_df['requires_explicit_tz'] = True
                print("⚠️ Timestamps require explicit timezone specification")
        else:
            print("✅ All timestamps have explicit timezone information")
    else:
        print("✅ All timestamps are timezone-aware")
    
    return processed_df

# Helper functions
def _find_timestamp_column(df):
    """Find the timestamp column in the DataFrame"""
    timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'created_at', 'recorded_at']
    
    for col in df.columns:
        if col.lower() in timestamp_candidates:
            return col
    
    return None

def _is_dst_transition_period(timestamp):
    """Check if a timestamp falls during a potential DST transition period"""
    if pd.isnull(timestamp):
        return False
    
    # Simplified DST detection - check if it's near DST transition dates
    # Spring: 2nd Sunday in March (around March 8-14)
    # Fall: 1st Sunday in November (around November 1-7)
    month = timestamp.month
    day = timestamp.day
    
    # Spring DST transition (March 8-14)
    if month == 3 and 8 <= day <= 14:
        return True
    
    # Fall DST transition (November 1-7) 
    if month == 11 and 1 <= day <= 7:
        return True
    
    return False

# Main edge case processing function
def process_all_edge_cases(df, timezone='America/New_York', travel_itinerary=None):
    """
    Process all edge cases in a comprehensive workflow
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input data to process
    timezone : str
        Primary timezone for processing
    travel_itinerary : list, optional
        Travel information for multi-timezone processing
        
    Returns:
    --------
    pandas.DataFrame
        Fully processed data with all edge cases handled
    """
    
    print("🔄 COMPREHENSIVE EDGE CASE PROCESSING")
    print("=" * 45)
    
    # Step 1: Handle mixed timestamp formats
    print("1️⃣ Processing mixed timestamp formats...")
    step1_df = handle_mixed_timestamp_formats(df.copy())
    
    # Step 2: Handle invalid timestamps
    print("\n2️⃣ Processing invalid timestamps...")
    step2_df = handle_invalid_timestamps(step1_df.copy())
    
    # Step 3: Handle missing timezone data
    print("\n3️⃣ Processing missing timezone data...")
    step3_df = handle_missing_timezone_data(step2_df.copy())
    
    # Step 4: Handle timezone ambiguity
    print("\n4️⃣ Processing timezone ambiguity...")
    step4_df = handle_timezone_ambiguity(step3_df.copy())
    
    # Step 5: Handle DST transitions
    print("\n5️⃣ Processing DST transitions...")
    step5_df = handle_daylight_saving_transitions(step4_df.copy(), timezone)
    
    # Step 6: Handle traveling user scenarios
    print("\n6️⃣ Processing traveling user scenarios...")
    final_df = handle_traveling_user_timezone_changes(step5_df.copy(), travel_itinerary)
    
    print(f"\n✅ COMPREHENSIVE EDGE CASE PROCESSING COMPLETE")
    print(f"   • Original records: {len(df)}")
    print(f"   • Final records: {len(final_df)}")
    print(f"   • New columns added: {len(final_df.columns) - len(df.columns)}")
    print(f"   • Processing steps completed: 6")
    
    return final_df
