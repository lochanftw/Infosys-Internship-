"""
Unit Tests for Task 3 Timezone Processing Components
"""

import unittest
import pandas as pd
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTask3Components(unittest.TestCase):
    """Test suite for Task 3 timezone processing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'timestamp': [
                '2024-03-01 09:00:00',
                '2024-03-01 15:00:00', 
                '2024-03-01 18:00:00'
            ],
            'heart_rate': [75, 80, 70],
            'steps': [120, 150, 80],
            'location': ['New York', 'New York', 'New York']
        })
    
    def test_basic_imports(self):
        """Test that Task 3 components can be imported"""
        try:
            from timestamp_processing.timezone_processor import detect_and_normalize_timestamps
            from timestamp_processing.timezone_detector import TimezoneDetector
            from timestamp_processing.timestamp_validator import TimestampValidator
            print("✅ All Task 3 components imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import Task 3 components: {e}")
    
    def test_timezone_detector_initialization(self):
        """Test timezone detector can be initialized"""
        try:
            from timestamp_processing.timezone_detector import TimezoneDetector
            detector = TimezoneDetector()
            self.assertIsInstance(detector, TimezoneDetector)
            print("✅ TimezoneDetector initialized successfully")
        except Exception as e:
            self.fail(f"TimezoneDetector initialization failed: {e}")
    
    def test_timezone_processing(self):
        """Test basic timezone processing functionality"""
        try:
            from timestamp_processing.timezone_processor import detect_and_normalize_timestamps
            processed = detect_and_normalize_timestamps(self.sample_data.copy())
            
            # Should have same number of rows
            self.assertEqual(len(processed), len(self.sample_data))
            
            # Should have more columns (metadata added)
            self.assertGreaterEqual(len(processed.columns), len(self.sample_data.columns))
            
            print("✅ Timezone processing test passed")
            
        except Exception as e:
            self.fail(f"Timezone processing failed: {e}")

if __name__ == '__main__':
    print("🧪 Running Task 3 Unit Tests")
    print("=" * 40)
    unittest.main(verbosity=2)
