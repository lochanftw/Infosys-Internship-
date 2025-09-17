"""
Format Comparison Analysis for Task 2
FIXED VERSION - No Import Issues
"""

import pandas as pd
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

# Import the universal loader
from src.data_loading.universal_loader import load_fitness_data

def analyze_data_formats(sample_size: int = 500) -> pd.DataFrame:
    """
    Compare different file formats for fitness data storage
    """
    
    print(f"🔄 Generating {sample_size} sample records for format analysis...")
    
    # Generate sample data
    np.random.seed(42)
    start_time = datetime.now() - timedelta(days=1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(sample_size)]
    
    sample_data = {
        'timestamp': timestamps,
        'heart_rate': np.random.normal(75, 15, sample_size).astype(int),
        'steps': np.random.poisson(5, sample_size),
        'calories': np.random.normal(2.5, 0.8, sample_size).round(2),
        'activity_level': np.random.choice(['sedentary', 'light', 'moderate', 'vigorous'], sample_size)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test different formats
    formats = ['csv', 'json']
    results = []
    
    print("📊 Testing different formats...")
    
    for fmt in formats:
        print(f"  Testing {fmt.upper()} format...")
        
        test_file = f"data/test_outputs/test_data.{fmt}"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        
        try:
            # Measure write performance
            write_start = time.time()
            if fmt == 'csv':
                df.to_csv(test_file, index=False)
            elif fmt == 'json':
                df.to_json(test_file, orient='records', date_format='iso')
            write_time = time.time() - write_start
            
            # Measure file size
            file_size = os.path.getsize(test_file) / 1024  # KB
            
            # Measure read performance
            read_start = time.time()
            loaded_df = load_fitness_data(test_file, fmt)
            read_time = time.time() - read_start
            
            # Verify data loaded correctly
            read_success = not loaded_df.empty
            
            results.append({
                'Format': fmt.upper(),
                'File Size (KB)': round(file_size, 2),
                'Write Time (s)': round(write_time, 4),
                'Read Time (s)': round(read_time, 4),
                'Records Loaded': len(loaded_df) if read_success else 0,
                'Load Success': '✅' if read_success else '❌',
                'Ease of Use (1-10)': 9 if fmt == 'csv' else 7,
                'Human Readable': 'Yes',
                'Nested Data Support': 'Limited' if fmt == 'csv' else 'Excellent'
            })
            
            print(f"    ✅ {fmt.upper()}: {file_size:.2f}KB, {len(loaded_df)} records loaded")
            
        except Exception as e:
            print(f"    ❌ {fmt.upper()} failed: {e}")
            results.append({
                'Format': fmt.upper(),
                'File Size (KB)': 'ERROR',
                'Write Time (s)': 'ERROR',
                'Read Time (s)': 'ERROR',
                'Records Loaded': 0,
                'Load Success': '❌',
                'Ease of Use (1-10)': 0,
                'Human Readable': 'ERROR',
                'Nested Data Support': 'ERROR'
            })
        
        # Cleanup
        try:
            if os.path.exists(test_file):
                os.remove(test_file)
        except:
            pass
    
    return pd.DataFrame(results)

def create_simple_comparison():
    """Create a basic format comparison table"""
    print("📊 Creating Simple Format Comparison...")
    
    data = {
        'Format': ['CSV', 'JSON'],
        'File Size': ['Smaller (~2-3KB)', 'Larger (~4-5KB)'],
        'Read Speed': ['Very Fast', 'Fast'],
        'Write Speed': ['Very Fast', 'Fast'],
        'Human Readable': ['Yes', 'Yes'],
        'Nested Data': ['Limited', 'Excellent'],
        'Ease of Use': ['9/10', '7/10'],
        'Best Use Case': ['Simple tabular data', 'Complex structured data'],
        'Industry Standard': ['Data analysis', 'Web APIs, configs']
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("📊 FORMAT COMPARISON ANALYSIS")
    print("=" * 60)
    
    try:
        # Try detailed analysis
        print("\n🔬 Running detailed performance analysis...")
        results = analyze_data_formats(200)  # Small sample for speed
        
        print("\n✅ DETAILED PERFORMANCE RESULTS:")
        print(results.to_string(index=False))
        
        # Show summary
        print(f"\n📋 SUMMARY:")
        for _, row in results.iterrows():
            if row['Load Success'] == '✅':
                print(f"  {row['Format']}: {row['File Size (KB)']}KB, "
                      f"{row['Read Time (s)']}s read time, "
                      f"{row['Records Loaded']} records")
        
    except Exception as e:
        print(f"\n⚠️ Detailed analysis failed: {e}")
        print("📊 Showing simple comparison instead...")
        
        simple_results = create_simple_comparison()
        print("\n📊 BASIC FORMAT COMPARISON:")
        print(simple_results.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("🎯 Format analysis completed!")
    print("\n💡 KEY INSIGHTS:")
    print("  • CSV: Best for simple, fast data exchange")
    print("  • JSON: Best for complex, nested data structures")
    print("  • Both formats are human-readable")
    print("  • Choose based on your data complexity needs")
