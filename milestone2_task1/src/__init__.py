"""
Heart Rate Forecasting Package
Author: ECE Student
Date: October 2025
"""

__version__ = "1.0.0"
__author__ = "ECE Student"

# Import main functions for easy access
from .data_preparation import prepare_data, load_raw_data
from .model_training import train_prophet_model
from .forecasting import make_forecast, calculate_mae
from .visualization import create_plots, plot_forecast, plot_components

__all__ = [
    'prepare_data',
    'load_raw_data',
    'train_prophet_model',
    'make_forecast',
    'calculate_mae',
    'create_plots',
    'plot_forecast',
    'plot_components'
]
