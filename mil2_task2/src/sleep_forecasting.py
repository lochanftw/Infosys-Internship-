"""
Sleep Forecasting Module
Handles sleep forecast generation and weekly pattern analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os


def make_sleep_forecast(model, periods=7):
    """
    Generate sleep forecast for the specified number of periods
    
    Args:
        model (Prophet): Trained Prophet model
        periods (int): Number of future periods to forecast (default: 7 days)
    
    Returns:
        pd.DataFrame: Forecast dataframe with predictions and confidence intervals
    """
    print("=" * 50)
    print("STEP 3: SLEEP FORECASTING")
    print("=" * 50 + "\n")
    
    print(f"Generating {periods}-day sleep forecast...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    print(f"  - Future dataframe created: {len(future)} total days")
    print(f"  - Forecast period: {future['ds'].max() - pd.Timedelta(days=periods-1)} to {future['ds'].max()}\n")
    
    # Generate predictions
    forecast = model.predict(future)
    
    print("✓ Sleep forecast generated successfully!")
    print(f"  - Total predictions: {len(forecast)}")
    print(f"  - Includes trend + weekly seasonality components\n")
    
    # Save forecast results
    save_sleep_forecast_results(forecast, periods)
    
    return forecast


def analyze_weekly_patterns(actual_data):
    """
    Analyze weekly sleep patterns from actual data
    
    Args:
        actual_data (pd.DataFrame): Actual sleep data with 'ds' and 'y' columns
    
    Returns:
        dict: Dictionary with weekly pattern statistics
    """
    print("Analyzing weekly sleep patterns...")
    
    # Add day of week information
    df_analysis = actual_data.copy()
    df_analysis['day_of_week'] = df_analysis['ds'].dt.day_name()
    df_analysis['day_num'] = df_analysis['ds'].dt.dayofweek
    
    # Calculate averages by day of week
    daily_averages = df_analysis.groupby('day_of_week')['y'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    # Find best and worst sleep days
    best_day = daily_averages.idxmax()
    worst_day = daily_averages.idxmin() 
    best_hours = daily_averages.max()
    worst_hours = daily_averages.min()
    
    # Calculate overall statistics
    average_sleep = df_analysis['y'].mean()
    min_sleep = df_analysis['y'].min()
    max_sleep = df_analysis['y'].max()
    
    # Calculate trend (simple linear regression slope)
    days = np.arange(len(df_analysis))
    trend_slope = np.polyfit(days, df_analysis['y'], 1)[0]
    
    results = {
        'daily_averages': daily_averages,
        'best_day': best_day,
        'worst_day': worst_day,
        'best_hours': best_hours,
        'worst_hours': worst_hours,
        'average_sleep': average_sleep,
        'min_sleep': min_sleep,
        'max_sleep': max_sleep,
        'trend_slope': trend_slope
    }
    
    print("✓ Weekly pattern analysis completed")
    print(f"  - Best sleep day: {best_day} ({best_hours:.1f} hours)")
    print(f"  - Worst sleep day: {worst_day} ({worst_hours:.1f} hours)")
    print(f"  - Sleep trend: {trend_slope:.3f} hours/day\n")
    
    return results


def save_sleep_forecast_results(forecast, periods=7):
    """
    Save sleep forecast results to CSV
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        periods (int): Number of forecast periods
    """
    # Create output directory
    os.makedirs('outputs/results', exist_ok=True)
    
    # Save full forecast
    forecast.to_csv('outputs/results/sleep_forecast_results.csv', index=False)
    
    # Save only future predictions (7-day forecast)
    future_only = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_only.columns = ['Date', 'Predicted_Sleep_Hours', 'Lower_Bound_95%', 'Upper_Bound_95%']
    future_only.to_csv('outputs/results/sleep_7day_forecast.csv', index=False)
    
    print(f"✓ Sleep forecast results saved to outputs/results/\n")


if __name__ == "__main__":
    # Test the module
    from sleep_data_preparation import prepare_sleep_data
    from sleep_model_training import train_sleep_prophet_model
    
    df = prepare_sleep_data()
    model = train_sleep_prophet_model(df)
    forecast = make_sleep_forecast(model, periods=7)
    
    # Analyze patterns
    weekly_stats = analyze_weekly_patterns(df)
    print(f"\nWeekly Pattern Analysis: {weekly_stats}")
