"""
Universal Data Loader for FitPulse Health Anomaly Detection
Task 2: Data Format Mastery Implementation
"""

import pandas as pd
import json
from pathlib import Path
import logging
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_fitness_data(file_path: str, file_type: str = 'auto') -> pd.DataFrame:
    """
    Universal fitness data loader supporting multiple formats
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
    file_type : str
        'csv', 'json', 'excel', or 'auto'
    
    Returns:
    --------
    pd.DataFrame
        Standardized fitness data
    """
    
    try:
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        
        logger.info(f"Loading file: {file_path}")
        
        # Auto-detect file type if needed
        if file_type == 'auto':
            ext = path.suffix.lower()
            if ext == '.csv':
                file_type = 'csv'
            elif ext == '.json':
                file_type = 'json'  
            elif ext in ['.xlsx', '.xls']:
                file_type = 'excel'
            else:
                logger.error(f"Unsupported file extension: {ext}")
                return pd.DataFrame()
        
        # Load based on detected/specified type
        if file_type == 'csv':
            df = _load_csv(file_path)
        elif file_type == 'json':
            df = _load_json(file_path)
        elif file_type == 'excel':
            df = _load_excel(file_path)
        else:
            logger.error(f"Unsupported file type: {file_type}")
            return pd.DataFrame()
        
        # Standardize the data
        if not df.empty:
            df = _standardize_fitness_data(df)
            logger.info(f"✅ Successfully loaded {len(df)} records with {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame()

def _load_csv(file_path: str) -> pd.DataFrame:
    """Load CSV file with error handling"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning("CSV file is empty")
        return df
    except pd.errors.EmptyDataError:
        logger.error("CSV file is empty or has no valid data")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error reading CSV: {str(e)}")
        return pd.DataFrame()

def _load_json(file_path: str) -> pd.DataFrame:
    """Load JSON file with nested structure handling"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(raw_data, list):
            # Direct list of records
            df = pd.json_normalize(raw_data)
        elif isinstance(raw_data, dict):
            # Look for fitness data in nested structure
            fitness_keys = ['fitness_data', 'data', 'records', 'heart_rate', 'activities']
            
            for key in fitness_keys:
                if key in raw_data and isinstance(raw_data[key], list):
                    df = pd.json_normalize(raw_data[key])
                    logger.info(f"Found fitness data in '{key}' field")
                    break
            else:
                # Try to flatten entire dict
                df = pd.json_normalize(raw_data)
                logger.warning("No specific fitness data structure found, flattening entire JSON")
        else:
            logger.error("JSON structure not recognized for fitness data")
            return pd.DataFrame()
            
        return df
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        return pd.DataFrame()
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error reading JSON: {str(e)}")
        return pd.DataFrame()

def _load_excel(file_path: str) -> pd.DataFrame:
    """Load Excel file with basic error handling"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        logger.error(f"Excel loading error: {str(e)}")
        return pd.DataFrame()

def _standardize_fitness_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and basic data types"""
    if df.empty:
        return df
    
    # Standardize column names (lowercase, replace spaces/special chars with underscores)
    df.columns = df.columns.str.lower().str.replace(r'[^\w]', '_', regex=True)
    
    # Handle timestamp columns
    time_columns = [col for col in df.columns if any(x in col for x in ['time', 'date', 'timestamp'])]
    for col in time_columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            logger.info(f"Converted {col} to datetime")
        except Exception as e:
            logger.warning(f"Could not convert {col} to datetime: {e}")
    
    return df

# Test function
def test_loader():
    """Test the universal loader with sample files"""
    print("🧪 Testing Universal Data Loader")
    print("=" * 40)
    
    test_files = [
        "data/raw/sample_heart_rate.csv",
        "data/raw/sample_fitness_data.json"
    ]
    
    for file_path in test_files:
        print(f"\n📁 Testing: {file_path}")
        df = load_fitness_data(file_path)
        
        if not df.empty:
            print(f"✅ Success: {len(df)} rows, {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)}")
            print("First 3 rows:")
            print(df.head(3))
        else:
            print("❌ Failed to load data")

if __name__ == "__main__":
    test_loader()
