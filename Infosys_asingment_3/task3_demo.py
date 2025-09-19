"""
Task 3: Timezone Processing - Main Demonstration
Entry point for showcasing Task 3 functionality
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd

def run_task3_demo():
    """Main Task 3 demonstration function"""
    
    print("🕐 TASK 3: TIMEZONE PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    try:
        # Import Task 3 components
        from timestamp_processing.timezone_processor import detect_and_normalize_timestamps
        from timestamp_processing.timezone_detector import TimezoneDetector
        from timestamp_processing.timestamp_validator import TimestampValidator
        
        print("✅ Task 3 components imported successfully")
        
        # 1. Load sample data
        print("\n📊 Loading sample data...")
        data_path = os.path.join('data', 'timezone_samples', 'new_york_user.csv')
        
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            print("💡 Please ensure data file exists")
            return False
            
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records from {data_path}")
        
        # 2. Timezone detection
        print("\n🔍 Detecting timezone patterns...")
        detector = TimezoneDetector()
        detection_result = detector.detect_timezone_from_patterns(df)
        print(f"🎯 Detected: {detection_result['timezone']}")
        print(f"🎯 Confidence: {detection_result['confidence']:.2f}")
        
        # 3. Process timestamps
        print("\n⚙️ Processing timestamp normalization...")
        processed_df = detect_and_normalize_timestamps(df.copy(), user_location='New York')
        print(f"✅ Successfully processed {len(processed_df)} records")
        
        # 4. Validation
        print("\n🔍 Validating results...")
        validator = TimestampValidator()
        validation_result = validator.validate_timestamp_conversion(df, processed_df)
        status = "PASSED" if validation_result['validation_passed'] else "FAILED"
        print(f"📋 Validation: {status}")
        
        # 5. Results summary
        print(f"\n🎉 TASK 3 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print(f"   • Original records: {len(df)}")
        print(f"   • Processed records: {len(processed_df)}")
        print(f"   • Detection confidence: {detection_result['confidence']:.2f}")
        print(f"   • Validation: {status}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure Task 3 components are created in src/timestamp_processing/")
        return False
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Make sure data file exists: data/timezone_samples/new_york_user.csv")
        return False
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("📁 Current working directory:", os.getcwd())
    success = run_task3_demo()
    exit_code = 0 if success else 1
    print(f"\n🔚 Demo completed with exit code: {exit_code}")
    sys.exit(exit_code)
