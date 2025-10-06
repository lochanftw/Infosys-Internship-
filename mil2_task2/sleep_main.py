import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.sleep_data_preparation import prepare_sleep_data
from src.sleep_model_training import train_sleep_prophet_model
from src.sleep_forecasting import make_sleep_forecast, analyze_weekly_patterns
from src.sleep_visualization import create_sleep_plots


def print_header():
    """Print project header"""
    print("\n" + "=" * 70)
    print(" " * 15 + "SLEEP PATTERN ANALYSIS WITH PROPHET")
    print(" " * 20 + "Infosys Internship - Task 2")
    print("=" * 70 + "\n")


def print_results(forecast, actual_data, model):
    """Print final results and analysis"""
    print("\n" + "=" * 70)
    print(" " * 25 + "FINAL RESULTS")
    print("=" * 70 + "\n")
    
    # Analyze weekly patterns
    weekly_stats = analyze_weekly_patterns(actual_data)
    
    print("📊 WEEKLY SLEEP PATTERN ANALYSIS:")
    print(f"   Most sleep: {weekly_stats['best_day']} ({weekly_stats['best_hours']:.1f} hours)")
    print(f"   Least sleep: {weekly_stats['worst_day']} ({weekly_stats['worst_hours']:.1f} hours)")
    print(f"   Average sleep: {weekly_stats['average_sleep']:.1f} hours")
    print(f"   Sleep range: {weekly_stats['min_sleep']:.1f} - {weekly_stats['max_sleep']:.1f} hours")
    
    print("\n   Weekly Pattern (Average hours per day):")
    for day, hours in weekly_stats['daily_averages'].items():
        print(f"     {day:>9}: {hours:.1f} hours")
    
    print("\n" + "-" * 70 + "\n")
    
    # Trend analysis
    if weekly_stats['trend_slope'] < -0.01:
        trend_direction = "DECREASING"
        trend_desc = "Sleep duration is decreasing over time"
    elif weekly_stats['trend_slope'] > 0.01:
        trend_direction = "INCREASING" 
        trend_desc = "Sleep duration is increasing over time"
    else:
        trend_direction = "STABLE"
        trend_desc = "Sleep duration is relatively stable over time"
    
    print(f"📈 SLEEP TREND ANALYSIS:")
    print(f"   Trend Direction: {trend_direction}")
    print(f"   Trend Rate: {weekly_stats['trend_slope']:.3f} hours per day")
    print(f"   Interpretation: {trend_desc}")
    
    if trend_direction == "DECREASING":
        print(f"   Possible causes: Increased workload, stress, or lifestyle changes")
    elif trend_direction == "INCREASING":
        print(f"   Possible causes: Better sleep hygiene or reduced stress")
    
    print("\n" + "-" * 70 + "\n")
    
    # 7-day forecast
    future_forecast = forecast.tail(7)
    print(f"📅 7-DAY SLEEP FORECAST:")
    for _, row in future_forecast.iterrows():
        day_name = row['ds'].strftime('%A')
        print(f"   {row['ds'].strftime('%Y-%m-%d')} ({day_name}): {row['yhat']:.1f} hours "
              f"[{row['yhat_lower']:.1f}-{row['yhat_upper']:.1f}]")
    
    print("\n" + "=" * 70 + "\n")


def print_summary():
    """Print project completion summary"""
    print("\n" + "=" * 70)
    print(" " * 20 + "TASK 2 COMPLETED SUCCESSFULLY! ✓")
    print("=" * 70 + "\n")
    
    print("📁 OUTPUT FILES GENERATED:")
    print("   ├── outputs/plots/")
    print("   │   ├── sleep_forecast_plot.png      [7-day forecast visualization]")
    print("   │   ├── sleep_components_plot.png    [Trend & weekly seasonality]")
    print("   │   └── weekly_pattern_plot.png      [Weekly sleep patterns]")
    print("   ├── outputs/models/")
    print("   │   └── sleep_prophet_model.pkl      [Saved trained model]")
    print("   └── outputs/results/")
    print("       └── sleep_forecast_results.csv   [Complete forecast data]")
    
    print("\n📝 ASSIGNMENT DELIVERABLES:")
    print("   ✓ Prophet model with weekly seasonality configuration")
    print("   ✓ Component plot showing trend and weekly patterns")
    print("   ✓ Identification of best/worst sleep days")
    print("   ✓ Sleep trend analysis (increasing/decreasing)")
    print("   ✓ 7-day forecast plot with high visualization quality")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main execution function - Sleep Pattern Analysis Pipeline
    """
    try:
        # Print header
        print_header()
        
        # Step 1: Data Preparation
        df = prepare_sleep_data()
        
        # Step 2: Model Training (with weekly seasonality)
        model = train_sleep_prophet_model(df)
        
        # Step 3: Forecasting (7 days)
        forecast = make_sleep_forecast(model, periods=7)
        
        # Step 4: Visualization
        create_sleep_plots(model, forecast, df)
        
        # Step 5: Results Analysis
        print_results(forecast, df, model)
        
        # Print completion summary
        print_summary()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
