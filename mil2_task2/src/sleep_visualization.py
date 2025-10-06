"""
Sleep Visualization Module
Handles all sleep-related plotting and visualization tasks
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_sleep_forecast(model, forecast, actual_data, save_path='outputs/plots/sleep_forecast_plot.png'):
    """
    Create main sleep forecast visualization
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        actual_data (pd.DataFrame): Actual historical data
        save_path (str): Path to save the plot
    """
    print("Creating sleep forecast visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot actual data
    ax.plot(actual_data['ds'], actual_data['y'], 
            'ko-', markersize=4, linewidth=1.5, label='Actual Sleep Hours', alpha=0.8)
    
    # Plot forecast (full timeline)
    ax.plot(forecast['ds'], forecast['yhat'], 
            'b-', linewidth=2, label='Forecast', alpha=0.9)
    
    # Plot confidence interval
    ax.fill_between(forecast['ds'], 
                     forecast['yhat_lower'], 
                     forecast['yhat_upper'],
                     alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Highlight forecast period (last 7 days)
    forecast_start = actual_data['ds'].max()
    ax.axvline(x=forecast_start, color='red', linestyle='--', 
               linewidth=2, label='7-Day Forecast Start', alpha=0.8)
    
    # Highlight weekly pattern with different colors for weekends
    for date, sleep in zip(forecast['ds'].tail(7), forecast['yhat'].tail(7)):
        if date.weekday() in [5, 6]:  # Saturday, Sunday
            ax.plot(date, sleep, 'ro', markersize=8, alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sleep Hours', fontsize=12, fontweight='bold')
    ax.set_title('Sleep Pattern Forecast: 90 Days Historical + 7 Days Forecast', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    y_min = min(actual_data['y'].min(), forecast['yhat'].min()) - 0.5
    y_max = max(actual_data['y'].max(), forecast['yhat'].max()) + 0.5
    ax.set_ylim(y_min, y_max)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sleep forecast plot saved to {save_path}")
    
    plt.close()


def plot_sleep_components(model, forecast, save_path='outputs/plots/sleep_components_plot.png'):
    """
    Create sleep components plot showing trend and weekly seasonality
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        save_path (str): Path to save the plot
    """
    print("Creating sleep components visualization...")
    
    # Use Prophet's built-in plot_components but customize it
    fig = model.plot_components(forecast, figsize=(14, 10))
    
    # Customize the components plot
    axes = fig.get_axes()
    
    # Trend plot (top)
    if len(axes) > 0:
        axes[0].set_title('Sleep Trend Over Time', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Sleep Hours', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
    
    # Weekly plot (bottom)
    if len(axes) > 1:
        axes[1].set_title('Weekly Sleep Seasonality Pattern', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Sleep Effect (hours)', fontweight='bold')
        axes[1].set_xlabel('Day of Week', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Add day labels
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1].set_xticks(range(len(days)))
        axes[1].set_xticklabels(days)
    
    plt.suptitle('Sleep Pattern Components Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sleep components plot saved to {save_path}")
    
    plt.close()


def plot_weekly_sleep_pattern(actual_data, save_path='outputs/plots/weekly_pattern_plot.png'):
    """
    Create detailed weekly sleep pattern visualization
    
    Args:
        actual_data (pd.DataFrame): Actual sleep data
        save_path (str): Path to save the plot
    """
    print("Creating weekly sleep pattern visualization...")
    
    # Prepare data
    df_plot = actual_data.copy()
    df_plot['day_of_week'] = df_plot['ds'].dt.day_name()
    df_plot['day_num'] = df_plot['ds'].dt.dayofweek
    
    # Calculate statistics
    daily_stats = df_plot.groupby('day_of_week')['y'].agg(['mean', 'std', 'count']).reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Bar chart of average sleep by day
    days = daily_stats.index
    means = daily_stats['mean']
    stds = daily_stats['std']
    
    bars = ax1.bar(days, means, yerr=stds, capsize=5, 
                   color=['lightcoral' if day in ['Saturday', 'Sunday'] else 'lightblue' for day in days],
                   edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean_val:.1f}h', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Average Sleep Hours by Day of Week', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Sleep Hours', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(means) + 1)
    
    # Plot 2: Box plot showing distribution
    sleep_by_day = [df_plot[df_plot['day_of_week'] == day]['y'].values for day in days]
    
    bp = ax2.boxplot(sleep_by_day, labels=days, patch_artist=True)
    
    # Color weekends differently
    colors = ['lightcoral' if day in ['Saturday', 'Sunday'] else 'lightblue' for day in days]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax2.set_title('Sleep Duration Distribution by Day of Week', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Sleep Hours', fontweight='bold')
    ax2.set_xlabel('Day of Week', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Weekly Sleep Pattern Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Weekly pattern plot saved to {save_path}")
    
    plt.close()


def create_sleep_plots(model, forecast, actual_data):
    """
    Create all sleep visualization plots
    
    Args:
        model (Prophet): Trained Prophet model
        forecast (pd.DataFrame): Forecast dataframe
        actual_data (pd.DataFrame): Actual historical data
    """
    print("=" * 50)
    print("STEP 4: SLEEP VISUALIZATION")
    print("=" * 50 + "\n")
    
    # Create all plots
    plot_sleep_forecast(model, forecast, actual_data)
    plot_sleep_components(model, forecast)
    plot_weekly_sleep_pattern(actual_data)
    
    print("\n✓ All sleep visualizations created successfully!")
    print("  Check outputs/plots/ folder for results\n")


if __name__ == "__main__":
    # Test the module
    from sleep_data_preparation import prepare_sleep_data
    from sleep_model_training import train_sleep_prophet_model
    from sleep_forecasting import make_sleep_forecast
    
    df = prepare_sleep_data()
    model = train_sleep_prophet_model(df)
    forecast = make_sleep_forecast(model, periods=7)
    create_sleep_plots(model, forecast, df)
