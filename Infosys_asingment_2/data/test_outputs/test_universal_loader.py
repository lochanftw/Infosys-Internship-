"""
Unit tests for universal data loader
"""

import pytest
import pandas as pd
import tempfile
import os
import sys

# Add project root to path
sys.path.append('.')
sys.path.append('../..')

from src.data_loading.universal_loader import load_fitness_data

class TestUniversalLoader:
    """Test class for universal data loader"""
    
    def test_load_csv_file(self):
        """Test loading CSV file"""
        # Create temporary CSV
        csv_content = "timestamp,heart_rate,steps\n2025-09-17 10:00:00,75,120\n2025-09-17 10:01:00,72,100\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            # Test loading
            df = load_fitness_data(temp_file)
            assert len(df) == 2
            assert 'heart_rate' in df.columns
            assert 'steps' in df.columns
        finally:
            # Cleanup
            os.unlink(temp_file)
    
    def test_load_json_file(self):
        """Test loading JSON file"""
        json_content = '''
        {
            "fitness_data": [
                {
                    "timestamp": "2025-09-17T10:00:00",
                    "heart_rate": 75,
                    "steps": 120
                }
            ]
        }
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(json_content)
            temp_file = f.name
        
        try:
            # Test loading
            df = load_fitness_data(temp_file)
            assert len(df) == 1
            assert 'heart_rate' in df.columns
        finally:
            # Cleanup
            os.unlink(temp_file)
    
    def test_auto_format_detection(self):
        """Test auto format detection"""
        csv_content = "timestamp,heart_rate\n2025-09-17 10:00:00,75\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_file = f.name
        
        try:
            # Test with auto detection
            df = load_fitness_data(temp_file, 'auto')
            assert not df.empty
        finally:
            # Cleanup
            os.unlink(temp_file)
    
    def test_error_handling_nonexistent_file(self):
        """Test with non-existent file"""
        df = load_fitness_data("nonexistent_file.csv")
        assert df.empty
    
    def test_error_handling_empty_path(self):
        """Test with empty file path"""
        df = load_fitness_data("")
        assert df.empty

if __name__ == "__main__":
    # Run tests directly
    import sys
    pytest.main([__file__] + sys.argv[1:])
