"""
Forecasting Module
Handles forecast generation and evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import os


def make_forecast(model, periods=14):
    """
    Generate forecast for the specified number of periods
    
    Args:
        model (Prophet): Trained Prophet model
        periods (int): Number of future periods to forecast
    
    Returns:
        pd.DataFrame: Forecast dataframe with predictions and confidence intervals
    """
    print("=" * 50)
    print("STEP 3: FORECASTING")
    print("=" * 50 + "\n")
    
    print(f"Generating {periods}-day forecast...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods, freq='D')
    
    print(f"  - Future dataframe created: {len(future)} total days")
    print(f"  - Forecast period: {future['ds'].max() - pd.Timedelta(days=periods-1)} to {future['ds'].max()}\n")
    
    # Generate predictions
    forecast = model.predict(future)
    
    print("✓ Forecast generated successfully!")
    print(f"  - Total predictions: {len(forecast)}")
    print(f"  - Forecast columns: {', '.join(forecast.columns[:5])}...\n")
    
    # Save forecast results
    save_forecast_results(forecast, periods)
    
    return forecast


def get_day_prediction(forecast, day_number):
    """
    Get the forecasted value for a specific day
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        day_number (int): Day number to retrieve (e.g., 67 for Day 67)
    
    Returns:
        dict: Dictionary with prediction details
    """
    # Get the specific day (day_number - 1 because index starts at 0)
    idx = day_number - 1
    
    if idx >= len(forecast):
        print(f"⚠ Warning: Day {day_number} exceeds forecast range")
        return None
    
    row = forecast.iloc[idx]
    
    prediction = {
        'day': day_number,
        'date': row['ds'],
        'predicted_value': round(row['yhat'], 2),
        'lower_bound': round(row['yhat_lower'], 2),
        'upper_bound': round(row['yhat_upper'], 2),
        'confidence_interval_width': round(row['yhat_upper'] - row['yhat_lower'], 2)
    }
    
    return prediction


def calculate_mae(model, actual_data):
    """
    Calculate Mean Absolute Error on training data
    
    Args:
        model (Prophet): Trained Prophet model
        actual_data (pd.DataFrame): Actual data with 'ds' and 'y' columns
    
    Returns:
        float: Mean Absolute Error
    """
    print("Calculating Mean Absolute Error (MAE)...")
    
    # Make predictions on training data
    train_forecast = model.predict(actual_data)
    
    # Calculate MAE
    mae = mean_absolute_error(actual_data['y'], train_forecast['yhat'])
    
    print(f"✓ MAE calculated: {mae:.3f} bpm")
    print(f"  - Interpretation: On average, predictions are off by ±{mae:.2f} bpm\n")
    
    return mae


def analyze_confidence_interval(forecast, periods=14):
    """
    Analyze confidence intervals of the forecast
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        periods (int): Number of forecast periods
    
    Returns:
        dict: Confidence interval statistics
    """
    print("Analyzing confidence intervals...")
    
    # Get only the forecasted period (last 'periods' rows)
    future_forecast = forecast.tail(periods)
    
    # Calculate confidence interval statistics
    avg_ci_width = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean()
    min_ci_width = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).min()
    max_ci_width = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).max()
    
    stats = {
        'average_width': round(avg_ci_width, 2),
        'min_width': round(min_ci_width, 2),
        'max_width': round(max_ci_width, 2),
        'percentage': 95  # Prophet uses 95% confidence interval by default
    }
    
    print(f"✓ Confidence Interval Analysis:")
    print(f"  - Coverage: {stats['percentage']}% confidence")
    print(f"  - Average width: ±{stats['average_width']/2:.2f} bpm")
    print(f"  - Range: {stats['min_width']:.2f} to {stats['max_width']:.2f} bpm")
    print(f"  - Interpretation: We are 95% confident that the true heart rate")
    print(f"    will fall within these intervals\n")
    
    return stats


def save_forecast_results(forecast, periods=14):
    """
    Save forecast results to CSV
    
    Args:
        forecast (pd.DataFrame): Forecast dataframe
        periods (int): Number of forecast periods
    """
    # Create output directory
    os.makedirs('outputs/results', exist_ok=True)
    
    # Save full forecast
    forecast.to_csv('outputs/results/forecast_results.csv', index=False)
    
    # Save only future predictions
    future_only = forecast.tail(periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    future_only.columns = ['Date', 'Predicted_Heart_Rate', 'Lower_Bound_95%', 'Upper_Bound_95%']
    future_only.to_csv('outputs/results/future_predictions.csv', index=False)
    
    print(f"✓ Forecast results saved to outputs/results/\n")


if __name__ == "__main__":
    # Test the module
    from data_preparation import prepare_data
    from model_training import train_prophet_model
    
    df = prepare_data()
    model = train_prophet_model(df)
    forecast = make_forecast(model, periods=14)
    
    # Get Day 67 prediction
    day_67 = get_day_prediction(forecast, 67)
    print(f"\nDay 67 Prediction: {day_67}")
    
    # Calculate MAE
    mae = calculate_mae(model, df)
    
    # Analyze confidence intervals
    ci_stats = analyze_confidence_interval(forecast)
