"""
FitPulse Analytics Pro - Milestone 2
Advanced Feature Extraction & Behavioral Pattern Analysis Platform

Author: FitPulse Development Team
Version: 2.0.0
License: MIT
"""

__version__ = "2.0.0"
__author__ = "FitPulse Team"
__license__ = "MIT"
__email__ = "support@fitpulse.ai"

from .feature_engineering import AdvancedFeatureEngine
from .time_series_analysis import TrendForecastingEngine
from .pattern_clustering import BehaviorPatternAnalyzer
from .data_generator import generate_comprehensive_dataset
from . import utils

__all__ = [
    'AdvancedFeatureEngine',
    'TrendForecastingEngine',
    'BehaviorPatternAnalyzer',
    'generate_comprehensive_dataset',
    'utils'
]

# Package metadata
PACKAGE_INFO = {
    'name': 'FitPulse Milestone 2',
    'description': 'Advanced health analytics with TSFresh, Prophet, and ML clustering',
    'features': [
        'TSFresh statistical feature extraction',
        'Facebook Prophet time-series forecasting',
        'KMeans & DBSCAN behavioral clustering',
        'Real-time anomaly detection',
        'Interactive Streamlit dashboard'
    ],
    'supported_metrics': ['heart_rate', 'steps', 'activity', 'sleep'],
    'python_version': '>=3.8'
}

def get_package_info():
    """Return package information"""
    return PACKAGE_INFO

def print_welcome():
    """Print welcome message"""
    print("=" * 70)
    print(f"  🧬 {PACKAGE_INFO['name']} v{__version__}")
    print("=" * 70)
    print(f"  {PACKAGE_INFO['description']}")
    print("\n  Features:")
    for feature in PACKAGE_INFO['features']:
        print(f"    ✓ {feature}")
    print("=" * 70)
