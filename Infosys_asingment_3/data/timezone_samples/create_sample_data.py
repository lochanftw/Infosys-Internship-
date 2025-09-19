"""
Enhanced Sample Data Creator for Task 3 with Debug Output
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
import json
import os

def create_new_york_user_data():
    """Create data for New York user traveling to Tokyo"""
    
    print("🗽 Creating New York user data...")
    
    try:
        # Base date range - 15 days of hourly data
        start_date = datetime(2024, 3, 1, 6, 0)
        end_date = datetime(2024, 3, 15, 22, 0)
        dates = []
        
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date)
            current_date += timedelta(hours=1)
        
        print(f"   📅 Generated {len(dates)} timestamps")
        
        # Create base fitness data
        np.random.seed(42)  # For reproducible data
        
        heart_rates = []
        steps_data = []
        calories_data = []
        locations = []
        
        for i, date in enumerate(dates):
            # Normal heart rate with some variation
            base_hr = 75
            if date.hour >= 22 or date.hour <= 6:  # Sleep hours
                base_hr = 65
            elif 9 <= date.hour <= 17:  # Active hours
                base_hr = 80
            
            heart_rates.append(base_hr + np.random.normal(0, 8))
            
            # Steps - higher during day, lower at night
            if date.hour >= 22 or date.hour <= 6:
                steps_data.append(np.random.poisson(10))
            else:
                steps_data.append(np.random.poisson(120))
            
            # Calories
            calories_data.append(np.random.normal(2.5, 1.0))
            
            # Location - Travel to Tokyo March 8-12
            if datetime(2024, 3, 8) <= date <= datetime(2024, 3, 12):
                locations.append('Tokyo')
            else:
                locations.append('New York')
        
        # Create DataFrame
        ny_data = pd.DataFrame({
            'timestamp': dates,
            'heart_rate': [max(40, min(200, int(hr))) for hr in heart_rates],  # Realistic range
            'steps': [max(0, int(s)) for s in steps_data],  # Non-negative
            'calories': [max(0, round(c, 2)) for c in calories_data],  # Non-negative, 2 decimals
            'user_id': 'user_ny_001',
            'location': locations
        })
        
        print(f"   📊 Created DataFrame with {len(ny_data)} rows and {len(ny_data.columns)} columns")
        print(f"   📋 Columns: {list(ny_data.columns)}")
        print(f"   📈 Data sample:")
        print(ny_data.head(3).to_string(index=False))
        
        # Ensure directory exists
        os.makedirs('data/timezone_samples', exist_ok=True)
        
        # Save CSV
        output_path = 'data/timezone_samples/new_york_user.csv'
        ny_data.to_csv(output_path, index=False)
        
        # Verify file was created and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✅ File saved: {output_path} ({file_size} bytes)")
            
            # Test reading it back
            test_df = pd.read_csv(output_path)
            print(f"   ✅ Verification: File contains {len(test_df)} records")
        else:
            print(f"   ❌ File not created: {output_path}")
        
        return ny_data
        
    except Exception as e:
        print(f"   ❌ Error creating New York data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame on error

def create_london_dst_data():
    """Create London user data with DST transition"""
    
    print("🇬🇧 Creating London DST data...")
    
    try:
        # DST transition period in UK (March 28-April 2, 2024)
        dates = pd.date_range('2024-03-28 00:00:00', '2024-04-02 23:00:00', freq='H')
        
        london_data = pd.DataFrame({
            'timestamp': dates,
            'heart_rate': np.random.normal(72, 10, len(dates)).astype(int),
            'steps': np.random.poisson(100, len(dates)),
            'calories': np.round(np.random.normal(2.8, 0.9, len(dates)), 2),
            'sleep_quality': np.random.choice(['poor', 'fair', 'good'], len(dates), p=[0.2, 0.5, 0.3]),
            'user_id': 'user_uk_001',
            'location': 'London'
        })
        
        # Mark DST transition period
        dst_transition = datetime(2024, 3, 31, 1, 0)
        dst_mask = (london_data['timestamp'] >= dst_transition - timedelta(hours=12)) & \
                   (london_data['timestamp'] <= dst_transition + timedelta(hours=12))
        
        london_data['is_dst_transition'] = dst_mask
        
        print(f"   📊 Created {len(london_data)} records")
        print(f"   🕐 DST transition records: {dst_mask.sum()}")
        
        # Save
        output_path = 'data/timezone_samples/london_dst.csv'
        london_data.to_csv(output_path, index=False)
        
        # Verify
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✅ File saved: {output_path} ({file_size} bytes)")
        
        return london_data
        
    except Exception as e:
        print(f"   ❌ Error creating London data: {e}")
        return pd.DataFrame()

def create_mixed_timezones_data():
    """Create data with mixed timezone formats"""
    
    print("🌍 Creating mixed timezone data...")
    
    try:
        mixed_data = []
        
        # Tokyo data with timezone-aware timestamps
        tokyo_tz = pytz.timezone('Asia/Tokyo')
        base_time = datetime(2024, 3, 10, 8, 0)
        
        for i in range(30):
            ts = base_time + timedelta(minutes=i * 10)
            tokyo_ts = tokyo_tz.localize(ts)
            
            mixed_data.append({
                'timestamp': tokyo_ts.isoformat(),
                'heart_rate': round(np.random.normal(78, 8), 1),
                'steps': int(np.random.poisson(90)),
                'source': 'fitness_watch_tokyo',
                'format': 'iso_with_tz',
                'user_id': 'user_global_001'
            })
        
        # Los Angeles naive timestamps
        base_time_la = datetime(2024, 3, 10, 15, 0)  # 3 PM LA time
        for i in range(30):
            ts = base_time_la + timedelta(minutes=i * 10)
            
            mixed_data.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate': round(np.random.normal(74, 12), 1),
                'steps': int(np.random.poisson(85)),
                'source': 'phone_app_la',
                'format': 'naive_local',
                'user_id': 'user_global_001'
            })
        
        # European format (Berlin)
        base_time_berlin = datetime(2024, 3, 10, 12, 0)
        for i in range(30):
            ts = base_time_berlin + timedelta(minutes=i * 10)
            
            mixed_data.append({
                'timestamp': ts.strftime('%d/%m/%Y %H:%M:%S'),  # DD/MM/YYYY
                'heart_rate': round(np.random.normal(76, 9), 1),
                'steps': int(np.random.poisson(95)),
                'source': 'health_app_berlin',
                'format': 'european_date',
                'user_id': 'user_global_001'
            })
        
        # Shuffle data
        import random
        random.shuffle(mixed_data)
        
        print(f"   📊 Created {len(mixed_data)} mixed records")
        
        # Save as JSON
        output_path = 'data/timezone_samples/mixed_timezones.json'
        with open(output_path, 'w') as f:
            json.dump(mixed_data, f, indent=2)
        
        # Verify
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"   ✅ File saved: {output_path} ({file_size} bytes)")
        
        return pd.DataFrame(mixed_data)
        
    except Exception as e:
        print(f"   ❌ Error creating mixed data: {e}")
        return pd.DataFrame()

def create_travel_schedule_data():
    """Create detailed travel schedule"""
    
    print("✈️ Creating travel schedule...")
    
    travel_schedule = [
        {
            'start_date': '2024-03-01T00:00:00',
            'end_date': '2024-03-07T23:59:59',
            'timezone': 'America/New_York',
            'location': 'New York',
            'activity_pattern': 'business'
        },
        {
            'start_date': '2024-03-08T00:00:00',
            'end_date': '2024-03-10T23:59:59',
            'timezone': 'Asia/Tokyo',
            'location': 'Tokyo',
            'activity_pattern': 'travel'
        },
        {
            'start_date': '2024-03-11T00:00:00',
            'end_date': '2024-03-12T23:59:59',
            'timezone': 'Australia/Sydney',
            'location': 'Sydney',
            'activity_pattern': 'vacation'
        },
        {
            'start_date': '2024-03-13T00:00:00',
            'end_date': '2024-03-15T23:59:59',
            'timezone': 'America/New_York',
            'location': 'New York',
            'activity_pattern': 'business'
        }
    ]
    
    # Save
    output_path = 'data/timezone_samples/travel_schedule.json'
    with open(output_path, 'w') as f:
        json.dump(travel_schedule, f, indent=2)
    
    print(f"   ✅ Created {len(travel_schedule)} travel periods")
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"   ✅ File saved: {output_path} ({file_size} bytes)")
    
    return travel_schedule

if __name__ == "__main__":
    print("🏗️ CREATING SAMPLE DATA FOR TASK 3")
    print("=" * 50)
    
    # Create directory
    os.makedirs('data/timezone_samples', exist_ok=True)
    print("📁 Created directory: data/timezone_samples")
    
    # Create all datasets with error handling
    results = {}
    
    try:
        results['ny_data'] = create_new_york_user_data()
        results['london_data'] = create_london_dst_data()
        results['mixed_data'] = create_mixed_timezones_data()
        results['travel_schedule'] = create_travel_schedule_data()
        
        print("\n" + "=" * 50)
        print("📊 SAMPLE DATA SUMMARY")
        print("=" * 50)
        
        for name, data in results.items():
            if isinstance(data, pd.DataFrame):
                print(f"  • {name}: {len(data)} records")
            elif isinstance(data, list):
                print(f"  • {name}: {len(data)} periods")
        
        print(f"\n✅ All sample data created successfully!")
        print("📁 Files saved in: data/timezone_samples/")
        
        # List all created files
        print("\n📋 Created files:")
        for filename in os.listdir('data/timezone_samples'):
            filepath = os.path.join('data/timezone_samples', filename)
            filesize = os.path.getsize(filepath)
            print(f"  • {filename}: {filesize} bytes")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
