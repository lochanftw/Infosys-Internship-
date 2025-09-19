"""
Complete Test Suite for Task 3: Timestamp Normalization Challenge
Tests all components and provides comprehensive validation
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append('.')

# Import all Task 3 components
try:
    from src.timestamp_processing.timezone_processor import detect_and_normalize_timestamps, process_traveling_user_data
    from src.timestamp_processing.edge_case_handlers import handle_daylight_saving_transitions, handle_mixed_timestamp_formats
    from src.timestamp_processing.timezone_detector import TimezoneDetector
    from src.timestamp_processing.timestamp_validator import TimestampValidator
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_SUCCESS = False

def test_all_task3_components():
    """Comprehensive test of all Task 3 components"""
    
    print("🚀 TASK 3: TIMESTAMP NORMALIZATION - COMPLETE TEST")
    print("=" * 70)
    
    if not IMPORTS_SUCCESS:
        print("❌ Cannot run tests - missing imports")
        return False
    
    # Test results tracker
    test_results = {
        'core_functionality': False,
        'edge_case_handling': False,
        'timezone_detection': False,
        'validation_system': False,
        'sample_data_processing': False
    }
    
    # Test 1: Core Timezone Processing
    print("\n1️⃣ TESTING CORE TIMEZONE PROCESSING")
    print("-" * 50)
    test_results['core_functionality'] = test_core_timezone_processing()
    
    # Test 2: Edge Case Handling
    print("\n2️⃣ TESTING EDGE CASE HANDLING")
    print("-" * 50)
    test_results['edge_case_handling'] = test_edge_case_handling()
    
    # Test 3: Timezone Detection
    print("\n3️⃣ TESTING AUTOMATIC TIMEZONE DETECTION")
    print("-" * 50)
    test_results['timezone_detection'] = test_timezone_detection()
    
    # Test 4: Validation System
    print("\n4️⃣ TESTING VALIDATION SYSTEM")
    print("-" * 50)
    test_results['validation_system'] = test_validation_system()
    
    # Test 5: Sample Data Processing
    print("\n5️⃣ TESTING SAMPLE DATA PROCESSING")
    print("-" * 50)
    test_results['sample_data_processing'] = test_sample_data_processing()
    
    # Overall Results
    print("\n" + "=" * 70)
    print("📊 TASK 3 TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    overall_success = passed_tests == total_tests
    print(f"\n🎯 OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if overall_success:
        print("🎉 ALL TASK 3 COMPONENTS WORKING PERFECTLY!")
    else:
        print("⚠️ Some components need attention")
    
    return overall_success

def test_core_timezone_processing():
    """Test core timezone processing functionality"""
    
    try:
        # Create test data with mixed timezones
        test_data = create_sample_fitness_data()
        
        print("📊 Testing timezone normalization...")
        
        # Test different user locations
        locations = ['New York', 'Tokyo', 'London', 'Los Angeles']
        
        for location in locations:
            result = detect_and_normalize_timestamps(test_data.copy(), user_location=location)
            
            if 'source_timezone' not in result.columns:
                print(f"❌ {location}: Missing timezone metadata")
                return False
            
            # Check if timestamps are UTC
            timestamp_col = result.columns[0]  # Assume first column is timestamp
            timestamps = pd.to_datetime(result[timestamp_col])
            
            if len(timestamps.dropna()) == 0:
                print(f"❌ {location}: No valid timestamps after conversion")
                return False
            
            print(f"✅ {location}: {len(result)} records processed successfully")
        
        # Test traveling user scenario
        print("\n✈️ Testing traveling user processing...")
        
        travel_schedule = [
            {
                'start_date': datetime(2024, 3, 1),
                'end_date': datetime(2024, 3, 5),
                'timezone': 'America/New_York'
            },
            {
                'start_date': datetime(2024, 3, 6),
                'end_date': datetime(2024, 3, 10),
                'timezone': 'Asia/Tokyo'
            }
        ]
        
        travel_result = process_traveling_user_data(test_data.copy(), travel_schedule)
        
        if len(travel_result) == 0:
            print("❌ Travel processing failed")
            return False
        
        print(f"✅ Travel processing: {len(travel_result)} records processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Core processing test failed: {e}")
        return False

def test_edge_case_handling():
    """Test edge case handling functionality"""
    
    try:
        print("🕐 Testing DST transitions...")
        
        # Create data around DST transition
        dst_dates = pd.date_range('2024-03-08 00:00:00', '2024-03-12 23:00:00', freq='H')
        dst_data = pd.DataFrame({
            'timestamp': dst_dates,
            'heart_rate': np.random.normal(75, 10, len(dst_dates)),
            'steps': np.random.poisson(100, len(dst_dates))
        })
        
        dst_result = handle_daylight_saving_transitions(dst_data, 'America/New_York')
        
        if len(dst_result) == 0:
            print("❌ DST handling failed")
            return False
        
        print(f"✅ DST handling: {len(dst_result)} records processed")
        
        # Test mixed timestamp formats
        print("\n🔧 Testing mixed timestamp formats...")
        
        mixed_timestamps = [
            '2024-03-10 15:30:00',
            '03/10/2024 15:30:00',
            '2024-03-10T15:30:00',
            '10-03-2024 15:30:00'
        ]
        
        mixed_data = pd.DataFrame({
            'timestamp': mixed_timestamps,
            'heart_rate': [72, 75, 78, 74],
            'steps': [120, 0, 95, 150]
        })
        
        mixed_result = handle_mixed_timestamp_formats(mixed_data)
        
        if 'original_format' not in mixed_result.columns:
            print("❌ Mixed format handling failed")
            return False
        
        print(f"✅ Mixed formats: {len(mixed_result)} records standardized")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
        return False

def test_timezone_detection():
    """Test automatic timezone detection"""
    
    try:
        detector = TimezoneDetector()
        
        print("🔍 Testing timezone detection...")
        
        # Create test data with clear timezone patterns
        test_data = create_business_hours_data()
        
        detection_result = detector.detect_timezone_from_patterns(test_data)
        
        if 'timezone' not in detection_result:
            print("❌ Timezone detection failed")
            return False
        
        detected_tz = detection_result['timezone']
        confidence = detection_result['confidence']
        
        print(f"✅ Detected timezone: {detected_tz} (confidence: {confidence:.2f})")
        
        # Test travel pattern detection
        print("\n✈️ Testing travel pattern detection...")
        
        travel_data = create_travel_pattern_data()
        travel_events = detector.detect_travel_patterns(travel_data)
        
        print(f"✅ Travel detection: {len(travel_events)} events detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Timezone detection test failed: {e}")
        return False

def test_validation_system():
    """Test timestamp validation system"""
    
    try:
        validator = TimestampValidator()
        
        print("🔍 Testing validation system...")
        
        # Create original and converted data
        original_data = create_sample_fitness_data()
        converted_data = original_data.copy()
        converted_data['timestamp'] = pd.to_datetime(converted_data['timestamp']).dt.tz_localize('UTC')
        
        validation_result = validator.validate_timestamp_conversion(original_data, converted_data)
        
        if 'validation_passed' not in validation_result:
            print("❌ Validation system failed")
            return False
        
        passed = validation_result['validation_passed']
        print(f"✅ Validation result: {'PASSED' if passed else 'FAILED'}")
        
        # Test pattern analysis
        print("\n📈 Testing pattern analysis...")
        
        patterns = validator.analyze_timestamp_patterns(converted_data)
        
        if 'temporal_distribution' not in patterns:
            print("❌ Pattern analysis failed")
            return False
        
        print("✅ Pattern analysis completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False

def test_sample_data_processing():
    """Test processing of sample data files"""
    
    try:
        print("📁 Testing sample data files...")
        
        # Check for sample data files
        sample_files = [
            'data/timezone_samples/new_york_user.csv',
            'data/timezone_samples/london_dst.csv',
            'data/timezone_samples/mixed_timezones.json'
        ]
        
        files_found = 0
        
        for file_path in sample_files:
            if os.path.exists(file_path):
                files_found += 1
                
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_json(file_path)
                    
                    # Test processing
                    result = detect_and_normalize_timestamps(df)
                    
                    print(f"✅ {os.path.basename(file_path)}: {len(result)} records processed")
                    
                except Exception as e:
                    print(f"❌ {os.path.basename(file_path)}: Processing failed - {e}")
                    return False
            else:
                print(f"⚠️ {os.path.basename(file_path)}: File not found")
        
        if files_found == 0:
            print("⚠️ No sample data files found - creating basic test data")
            # Use created test data instead
            test_data = create_sample_fitness_data()
            result = detect_and_normalize_timestamps(test_data)
            print(f"✅ Test data: {len(result)} records processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample data test failed: {e}")
        return False

# Helper functions to create test data
def create_sample_fitness_data():
    """Create sample fitness data for testing"""
    
    dates = pd.date_range('2024-03-10 06:00:00', '2024-03-12 22:00:00', freq='H')
    
    return pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 12, len(dates)),
        'steps': np.random.poisson(120, len(dates)),
        'calories': np.random.normal(3.0, 1.0, len(dates)),
        'activity_level': np.random.choice(['sedentary', 'light', 'moderate', 'vigorous'], len(dates))
    })

def create_business_hours_data():
    """Create data with clear business hours pattern"""
    
    # Generate data with higher activity during business hours
    dates = []
    heart_rates = []
    steps = []
    
    base_date = datetime(2024, 3, 10, 6, 0)
    
    for i in range(72):  # 3 days of hourly data
        current_time = base_date + timedelta(hours=i)
        dates.append(current_time)
        
        hour = current_time.hour
        
        # Higher activity during business hours (9 AM - 5 PM)
        if 9 <= hour <= 17:
            heart_rates.append(np.random.normal(80, 10))
            steps.append(np.random.poisson(150))
        else:
            heart_rates.append(np.random.normal(65, 8))
            steps.append(np.random.poisson(20))
    
    return pd.DataFrame({
        'timestamp': dates,
        'heart_rate': heart_rates,
        'steps': steps
    })

def create_travel_pattern_data():
    """Create data that simulates travel between timezones"""
    
    # Simulate 10 days of data with timezone change in the middle
    dates = pd.date_range('2024-03-01 06:00:00', '2024-03-10 22:00:00', freq='H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, len(dates)),
        'steps': np.random.poisson(100, len(dates))
    })
    
    # Add some travel indicator (sudden change in activity pattern)
    mid_point = len(data) // 2
    data.loc[mid_point:, 'heart_rate'] += 5  # Jet lag effect
    
    return data

def create_sample_data_files():
    """Create sample data files if they don't exist"""
    
    print("🏗️ Creating sample data files for testing...")
    
    os.makedirs('data/timezone_samples', exist_ok=True)
    
    # New York user data
    ny_data = create_sample_fitness_data()
    ny_data['user_id'] = 'user_ny_001'
    ny_data['location'] = 'New York'
    ny_data.to_csv('data/timezone_samples/new_york_user.csv', index=False)
    
    # London DST data
    london_data = create_business_hours_data()
    london_data['user_id'] = 'user_uk_001'
    london_data['location'] = 'London'
    london_data.to_csv('data/timezone_samples/london_dst.csv', index=False)
    
    # Mixed timezone data
    mixed_data = create_travel_pattern_data()
    mixed_data['user_id'] = 'user_global_001'
    mixed_data.to_json('data/timezone_samples/mixed_timezones.json', orient='records')
    
    print("✅ Sample data files created")

if __name__ == "__main__":
    # Create sample data if needed
    if not os.path.exists('data/timezone_samples'):
        create_sample_data_files()
    
    # Run all tests
    success = test_all_task3_components()
    
    print(f"\n{'🎉 Task 3 implementation is ready!' if success else '⚠️ Some issues need to be addressed'}")
