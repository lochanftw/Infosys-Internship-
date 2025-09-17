"""
Complete Test Script for Task 2: Data Format Mastery
Run this to verify all components work correctly
"""

import sys
import os

# Add project root to Python path so imports work
sys.path.append('.')

def test_all_components():
    """Test all Task 2 components systematically"""
    print("🚀 TASK 2: DATA FORMAT MASTERY - COMPLETE TEST")
    print("=" * 60)
    
    # Test 1: Import Universal Loader
    print("\n1️⃣ TESTING UNIVERSAL LOADER IMPORT")
    print("-" * 40)
    try:
        from src.data_loading.universal_loader import load_fitness_data
        print("✅ Universal loader imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have:")
        print("   - src/__init__.py (empty file)")
        print("   - src/data_loading/__init__.py (empty file)")  
        print("   - src/data_loading/universal_loader.py")
        return False
    
    # Test 2: File Structure Check
    print("\n2️⃣ TESTING FILE STRUCTURE")
    print("-" * 40)
    required_files = [
        "src/__init__.py",
        "src/data_loading/__init__.py", 
        "src/data_loading/universal_loader.py",
        "data/raw/sample_heart_rate.csv",
        "data/raw/sample_fitness_data.json",
        "streamlit_app.py"
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n⚠️ Some required files are missing!")
    
    # Test 3: CSV Loading
    print("\n3️⃣ TESTING CSV DATA LOADING")
    print("-" * 40)
    csv_file = "data/raw/sample_heart_rate.csv"
    if os.path.exists(csv_file):
        try:
            df_csv = load_fitness_data(csv_file)
            if not df_csv.empty:
                print(f"✅ CSV Success: {len(df_csv)} records, {len(df_csv.columns)} columns")
                print(f"   Columns: {list(df_csv.columns)}")
                print("   First 2 rows:")
                print(df_csv.head(2).to_string(index=False))
            else:
                print("❌ CSV loading returned empty DataFrame")
        except Exception as e:
            print(f"❌ CSV loading error: {e}")
    else:
        print(f"❌ CSV file not found: {csv_file}")
    
    # Test 4: JSON Loading  
    print("\n4️⃣ TESTING JSON DATA LOADING")
    print("-" * 40)
    json_file = "data/raw/sample_fitness_data.json"
    if os.path.exists(json_file):
        try:
            df_json = load_fitness_data(json_file)
            if not df_json.empty:
                print(f"✅ JSON Success: {len(df_json)} records, {len(df_json.columns)} columns")
                print(f"   Columns: {list(df_json.columns)}")
                print("   First 2 rows:")
                print(df_json.head(2).to_string(index=False))
            else:
                print("❌ JSON loading returned empty DataFrame")
        except Exception as e:
            print(f"❌ JSON loading error: {e}")
    else:
        print(f"❌ JSON file not found: {json_file}")
    
    # Test 5: Auto Format Detection
    print("\n5️⃣ TESTING AUTO FORMAT DETECTION")
    print("-" * 40)
    test_cases = [
        ("CSV Auto-detect", csv_file, 'auto'),
        ("JSON Auto-detect", json_file, 'auto')
    ]
    
    for name, file_path, file_type in test_cases:
        if os.path.exists(file_path):
            try:
                df = load_fitness_data(file_path, file_type)
                if not df.empty:
                    print(f"✅ {name}: Detected and loaded successfully")
                else:
                    print(f"❌ {name}: Auto-detection failed")
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
    
    # Test 6: Error Handling
    print("\n6️⃣ TESTING ERROR HANDLING")
    print("-" * 40)
    error_test_cases = [
        ("Non-existent file", "nonexistent_file.csv"),
        ("Empty string path", ""),
        ("Invalid extension", "test.xyz")
    ]
    
    for name, test_file in error_test_cases:
        try:
            df = load_fitness_data(test_file)
            if df.empty:
                print(f"✅ {name}: Handled gracefully (empty DataFrame)")
            else:
                print(f"⚠️ {name}: Unexpected success")
        except Exception as e:
            print(f"❌ {name}: Unhandled exception - {e}")
    
    # Test 7: Format Analyzer (Optional)
    print("\n7️⃣ TESTING FORMAT ANALYZER (OPTIONAL)")
    print("-" * 40)
    try:
        from src.data_loading.format_analyzer import analyze_data_formats
        results = analyze_data_formats(50)  # Small sample for speed
        if not results.empty:
            print("✅ Format analyzer working")
            print("   Results preview:")
            print(results.to_string(index=False))
        else:
            print("❌ Format analyzer returned empty results")
    except ImportError:
        print("⚠️ Format analyzer not implemented yet (optional)")
    except Exception as e:
        print(f"❌ Format analyzer error: {e}")
    
    # Test 8: Error Handler (Optional)
    print("\n8️⃣ TESTING ERROR HANDLER (OPTIONAL)")
    print("-" * 40)
    try:
        from src.data_loading.error_handlers import demonstrate_error_handling
        print("✅ Error handler module found")
        print("   Running error handling demo...")
        demonstrate_error_handling()
    except ImportError:
        print("⚠️ Error handler not implemented yet (optional)")
    except Exception as e:
        print(f"❌ Error handler error: {e}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🎯 TASK 2 TEST SUMMARY")
    print("=" * 60)
    
    print("✅ COMPLETED COMPONENTS:")
    if os.path.exists("src/data_loading/universal_loader.py"):
        print("  • Universal Data Loader")
    if os.path.exists("data/raw/sample_heart_rate.csv"):
        print("  • Sample CSV Data")
    if os.path.exists("data/raw/sample_fitness_data.json"):
        print("  • Sample JSON Data")
    if os.path.exists("streamlit_app.py"):
        print("  • Streamlit Dashboard")
    
    print("\n📋 NEXT STEPS:")
    if not os.path.exists("src/data_loading/format_analyzer.py"):
        print("  • Create format_analyzer.py")
    if not os.path.exists("src/data_loading/error_handlers.py"):
        print("  • Create error_handlers.py")
    if not os.path.exists("data/raw/corrupted_samples/"):
        print("  • Create corrupted sample files")
    
    return True

def quick_test():
    """Quick test of just the core functionality"""
    print("⚡ QUICK TEST - Core Components Only")
    print("=" * 40)
    
    try:
        from src.data_loading.universal_loader import load_fitness_data
        
        # Test CSV
        if os.path.exists("data/raw/sample_heart_rate.csv"):
            df = load_fitness_data("data/raw/sample_heart_rate.csv")
            print(f"CSV: {len(df)} records loaded")
        
        # Test JSON
        if os.path.exists("data/raw/sample_fitness_data.json"):
            df = load_fitness_data("data/raw/sample_fitness_data.json")
            print(f"JSON: {len(df)} records loaded")
        
        print("✅ Core functionality working!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full comprehensive test")
    print("2. Quick test only")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "2":
            quick_test()
        else:
            test_all_components()
    except KeyboardInterrupt:
        print("\nTest cancelled by user")
    except:
        # If input fails, just run full test
        test_all_components()
