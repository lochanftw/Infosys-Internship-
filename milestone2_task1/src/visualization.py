"""
Visualization Module
Handles all plotting and visualization tasks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from prophet.plot import plot_plotly, plot_components_plotly


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10


def plot_forecast(model, forecast, actual_data, save_path='outputs/plots/forecast_plot.png'):
    """
    Create main forecast visualization with actual data and confidence intervals
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        actual_data (pd.DataFrame): Actual historical data
        save_path (str): Path to save the plot
    """
    print("Creating forecast visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot actual data
    ax.plot(actual_data['ds'], actual_data['y'], 
            'ko', markersize=4, label='Actual Heart Rate', alpha=0.7)
    
    # Plot forecast
    ax.plot(forecast['ds'], forecast['yhat'], 
            'b-', linewidth=2, label='Forecast')
    
    # Plot confidence interval
    ax.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'],
                     alpha=0.2, color='blue', label='95% Confidence Interval')
    
    # Highlight forecast period
    forecast_start = actual_data['ds'].max()
    ax.axvline(x=forecast_start, color='red', linestyle='--', 
               linewidth=2, label='Forecast Start', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Heart Rate (bpm)', fontsize=12, fontweight='bold')
    ax.set_title('Heart Rate Forecast: 60 Days Historical + 14 Days Forecast', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Forecast plot saved to {save_path}")
    
    plt.close()


def plot_components(model, forecast, save_path='outputs/plots/components_plot.png'):
    """
    Create components plot showing trend and seasonality
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        save_path (str): Path to save the plot
    """
    print("Creating components visualization...")
    
    # Use Prophet's built-in plot_components
    fig = model.plot_components(forecast, figsize=(14, 8))
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Components plot saved to {save_path}")
    
    plt.close()


def plot_model_performance(model, actual_data, forecast, save_path='outputs/plots/model_performance.png'):
    """
    Create model performance visualization
    
    Args:
        model (Prophet): Trained Prophet model
        actual_data (pd.DataFrame): Actual historical data
        forecast (pd.DataFrame): Forecast dataframe
        save_path (str): Path to save the plot
    """
    print("Creating model performance visualization...")
    
    # Get predictions for actual data
    train_forecast = model.predict(actual_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(actual_data['y'], train_forecast['yhat'], alpha=0.6)
    axes[0, 0].plot([actual_data['y'].min(), actual_data['y'].max()], 
                     [actual_data['y'].min(), actual_data['y'].max()], 
                     'r--', linewidth=2)
    axes[0, 0].set_xlabel('Actual Heart Rate (bpm)', fontweight='bold')
    axes[0, 0].set_ylabel('Predicted Heart Rate (bpm)', fontweight='bold')
    axes[0, 0].set_title('Actual vs Predicted', fontweight='bold', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals over time
    residuals = actual_data['y'] - train_forecast['yhat']
    axes[0, 1].plot(actual_data['ds'], residuals, 'o-', alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Date', fontweight='bold')
    axes[0, 1].set_ylabel('Residual (bpm)', fontweight='bold')
    axes[0, 1].set_title('Residuals Over Time', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Residuals distribution
    axes[1, 0].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residual (bpm)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Residuals Distribution', fontweight='bold', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Forecast uncertainty over time
    future_only = forecast.tail(14)
    uncertainty = future_only['yhat_upper'] - future_only['yhat_lower']
    axes[1, 1].plot(future_only['ds'], uncertainty, 'o-', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Date', fontweight='bold')
    axes[1, 1].set_ylabel('Confidence Interval Width (bpm)', fontweight='bold')
    axes[1, 1].set_title('Forecast Uncertainty Over Time', fontweight='bold', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Performance plot saved to {save_path}")
    
    plt.close()


def create_plots(model, forecast, actual_data):
    """
    Create all visualization plots
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        actual_data (pd.DataFrame): Actual historical data
    """
    print("=" * 50)
    print("STEP 4: VISUALIZATION")
    print("=" * 50 + "\n")
    
    # Create all plots
    plot_forecast(model, forecast, actual_data)
    plot_components(model, forecast)
    plot_model_performance(model, actual_data, forecast)
    
    print("\n✓ All visualizations created successfully!")
    print("  Check outputs/plots/ folder for results\n")


if __name__ == "__main__":
    # Test the module
    from data_preparation import prepare_data
    from model_training import train_prophet_model
    from forecasting import make_forecast
    
    df = prepare_data()
    model = train_prophet_model(df)
    forecast = make_forecast(model, periods=14)
    create_plots(model, forecast, df)
