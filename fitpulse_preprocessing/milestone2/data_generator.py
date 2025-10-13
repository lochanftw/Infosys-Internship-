"""
Data Generator Module
Generate realistic health data with circadian rhythms
"""

import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st


def generate_comprehensive_dataset(days=7, frequency='1min'):
    """Generate comprehensive multi-metric health dataset"""
    
    start_date = datetime(2024, 10, 1, 6, 0, 0)
    total_minutes = days * 24 * 60
    timestamps = pd.date_range(start=start_date, periods=total_minutes, freq=frequency)
    
    st.info(f"🔄 Generating {days}-day dataset with {len(timestamps):,} records per metric...")
    
    progress_bar = st.progress(0, text="Initializing...")
    
    datasets = {}
    
    # Heart Rate
    progress_bar.progress(25, text="Generating heart rate data...")
    datasets['heart_rate'] = _generate_heart_rate_data(timestamps)
    
    # Steps
    progress_bar.progress(50, text="Generating step count data...")
    datasets['steps'] = _generate_steps_data(timestamps)
    
    # Activity
    progress_bar.progress(75, text="Generating activity data...")
    datasets['activity'] = _generate_activity_data(timestamps)
    
    progress_bar.progress(100, text="✅ Complete!")
    
    st.success(f"✅ Generated {len(datasets)} datasets successfully!")
    
    return datasets


def _generate_heart_rate_data(timestamps):
    """Generate realistic heart rate data"""
    
    hr_values = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        
        if 0 <= hour < 6:
            base_hr = 55 + np.random.normal(0, 3)
        elif 6 <= hour < 9:
            base_hr = 72 + np.random.normal(8, 5)
        elif 9 <= hour < 12:
            base_hr = 75 + np.random.normal(10, 6)
        elif 12 <= hour < 14:
            base_hr = 78 + np.random.normal(7, 4)
        elif 14 <= hour < 17:
            base_hr = 80 + np.random.normal(12, 7)
        elif 17 <= hour < 20:
            base_hr = 85 + np.random.normal(15, 8)
        elif 20 <= hour < 22:
            base_hr = 70 + np.random.normal(8, 5)
        else:
            base_hr = 62 + np.random.normal(4, 3)
        
        if day_of_week < 5 and 18 <= hour < 19:
            base_hr += np.random.normal(40, 10)
        
        if day_of_week >= 5 and 9 <= hour < 11:
            if np.random.random() < 0.7:
                base_hr += np.random.normal(30, 10)
        
        if np.random.random() < 0.05:
            base_hr += np.random.normal(20, 5)
        
        hr_values.append(max(45, min(190, base_hr)))
    
    return pd.DataFrame({'timestamp': timestamps, 'heart_rate': hr_values})


def _generate_steps_data(timestamps):
    """Generate realistic step count data"""
    
    steps_values = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        
        if 0 <= hour < 6:
            base_steps = np.random.poisson(0.2)
        elif 6 <= hour < 9:
            base_steps = np.random.poisson(10)
        elif 9 <= hour < 12:
            base_steps = np.random.poisson(8)
        elif 12 <= hour < 13:
            base_steps = np.random.poisson(20)
        elif 13 <= hour < 17:
            base_steps = np.random.poisson(10)
        elif 17 <= hour < 20:
            base_steps = np.random.poisson(25)
        elif 20 <= hour < 22:
            base_steps = np.random.poisson(5)
        else:
            base_steps = np.random.poisson(2)
        
        if day_of_week >= 5 and 9 <= hour < 18:
            base_steps += np.random.poisson(15)
        
        if np.random.random() < 0.08:
            base_steps += np.random.poisson(25)
        
        steps_values.append(max(0, base_steps))
    
    return pd.DataFrame({'timestamp': timestamps, 'step_count': steps_values})


def _generate_activity_data(timestamps):
    """Generate activity intensity levels (0-100)"""
    
    activity_values = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        
        if 0 <= hour < 6:
            intensity = np.random.normal(3, 1)
        elif 6 <= hour < 9:
            intensity = np.random.normal(30, 8)
        elif 9 <= hour < 12:
            intensity = np.random.normal(45, 12)
        elif 12 <= hour < 17:
            intensity = np.random.normal(50, 15)
        elif 17 <= hour < 20:
            intensity = np.random.normal(65, 18)
        elif 20 <= hour < 22:
            intensity = np.random.normal(25, 8)
        else:
            intensity = np.random.normal(10, 4)
        
        if day_of_week < 5 and 18 <= hour < 19:
            intensity += np.random.normal(30, 8)
        
        activity_values.append(max(0, min(100, intensity)))
    
    return pd.DataFrame({'timestamp': timestamps, 'activity_level': activity_values})
