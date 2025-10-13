"""
FitPulse Analytics Pro - Milestone 2
Advanced Feature Extraction and Behavioral Pattern Analysis
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__milestone__ = "2"

from .feature_engineering import AdvancedFeatureEngine
from .time_series_analysis import TrendForecastingEngine
from .pattern_clustering import BehaviorPatternAnalyzer
from . import utils

__all__ = [
    'AdvancedFeatureEngine',
    'TrendForecastingEngine',
    'BehaviorPatternAnalyzer',
    'utils'
]
