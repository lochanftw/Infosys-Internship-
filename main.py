import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preparation import prepare_data
from src.model_training import train_prophet_model
from src.forecasting import make_forecast, calculate_mae, get_day_prediction, analyze_confidence_interval
from src.visualization import create_plots


def print_header():
    """Print project header"""
    print("\n" + "=" * 70)
    print(" " * 15 + "HEART RATE FORECASTING WITH PROPHET")
    print(" " * 20 + "Infosys Internship - Task 1")
    print("=" * 70 + "\n")


def print_results(forecast, actual_data, model):
    """Print final results and analysis"""
    print("\n" + "=" * 70)
    print(" " * 25 + "FINAL RESULTS")
    print("=" * 70 + "\n")
    
    # Get Day 67 prediction
    day_67 = get_day_prediction(forecast, 67)
    
    if day_67:
        print("📊 DAY 67 FORECAST:")
        print(f"   Date: {day_67['date'].strftime('%Y-%m-%d')}")
        print(f"   Predicted Heart Rate: {day_67['predicted_value']} bpm")
        print(f"   95% Confidence Interval: [{day_67['lower_bound']}, {day_67['upper_bound']}] bpm")
        print(f"   Interval Width: ±{day_67['confidence_interval_width']/2:.1f} bpm")
    
    print("\n" + "-" * 70 + "\n")
    
    # Calculate and display MAE
    mae = calculate_mae(model, actual_data)
    print(f"📈 MODEL PERFORMANCE:")
    print(f"   Mean Absolute Error (MAE): {mae:.3f} bpm")
    print(f"   Interpretation: Predictions are typically within ±{mae:.2f} bpm of actual values")
    
    print("\n" + "-" * 70 + "\n")
    
    # Confidence interval analysis
    ci_stats = analyze_confidence_interval(forecast, periods=14)
    print(f"🎯 CONFIDENCE INTERVAL INTERPRETATION:")
    print(f"   The 95% confidence interval means:")
    print(f"   - We are 95% confident the true heart rate will fall within the shaded region")
    print(f"   - There is only a 5% chance the actual value falls outside this range")
    print(f"   - Average prediction uncertainty: ±{ci_stats['average_width']/2:.1f} bpm")
    print(f"   - This uncertainty accounts for:")
    print(f"     • Natural variation in heart rate")
    print(f"     • Model estimation error")
    print(f"     • Unpredictable factors (stress, exercise, illness)")
    
    print("\n" + "=" * 70 + "\n")


def print_summary():
    """Print project completion summary"""
    print("\n" + "=" * 70)
    print(" " * 20 + "PROJECT COMPLETED SUCCESSFULLY! ✓")
    print("=" * 70 + "\n")
    
    print("📁 OUTPUT FILES GENERATED:")
    print("   ├── outputs/plots/")
    print("   │   ├── forecast_plot.png          [Main forecast visualization]")
    print("   │   ├── components_plot.png        [Trend & seasonality analysis]")
    print("   │   └── model_performance.png      [Performance metrics]")
    print("   ├── outputs/models/")
    print("   │   └── prophet_model.pkl          [Saved trained model]")
    print("   └── outputs/results/")
    print("       ├── forecast_results.csv       [Complete forecast data]")
    print("       └── future_predictions.csv     [14-day predictions only]")
    
    print("\n📝 ASSIGNMENT DELIVERABLES:")
    print("   ✓ Code with detailed comments")
    print("   ✓ Forecast plot with actual data + forecast + confidence intervals")
    print("   ✓ Day 67 forecasted heart rate value")
    print("   ✓ Confidence interval interpretation")
    print("   ✓ MAE (Mean Absolute Error) calculation")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main execution function - Runs complete forecasting pipeline
    """
    try:
        # Print header
        print_header()
        
        # Step 1: Data Preparation
        # Load and format data for Prophet (requires 'ds' and 'y' columns)
        df = prepare_data()
        
        # Step 2: Model Training
        # Initialize and train Prophet model
        model = train_prophet_model(df)
        
        # Step 3: Forecasting
        # Generate 14-day forecast
        forecast = make_forecast(model, periods=14)
        
        # Step 4: Visualization
        # Create all plots (forecast, components, performance)
        create_plots(model, forecast, df)
        
        # Step 5: Results Analysis
        # Display final results and interpretations
        print_results(forecast, df, model)
        
        # Print completion summary
        print_summary()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"   Please check the error message and try again.\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
