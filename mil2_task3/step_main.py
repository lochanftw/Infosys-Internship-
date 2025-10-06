
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.step_data_preparation import prepare_step_data, create_holidays_dataframe
from src.step_model_training import train_step_models_comparison
from src.step_forecasting import make_step_forecast_comparison, analyze_holiday_impacts
from src.step_visualization import create_step_comparison_plots


def print_header():
    """Print project header"""
    print("\n" + "=" * 70)
    print(" " * 15 + "STEP COUNT WITH HOLIDAY EFFECTS")
    print(" " * 20 + "Infosys Internship - Task 3")
    print("=" * 70 + "\n")


def print_results(results_dict):
    """Print final results and analysis"""
    print("\n" + "=" * 70)
    print(" " * 25 + "FINAL RESULTS")
    print("=" * 70 + "\n")
    
    # Holiday impact analysis
    impacts = results_dict['holiday_impacts']
    
    print("📊 HOLIDAY IMPACT ANALYSIS:")
    print(f"   Model Performance Comparison:")
    print(f"   - Without holidays MAE: {results_dict['mae_no_holidays']:.1f} steps")
    print(f"   - With holidays MAE: {results_dict['mae_with_holidays']:.1f} steps")
    print(f"   - Improvement: {results_dict['mae_improvement']:.1f} steps ({results_dict['mae_improvement_percent']:.1f}%)")
    
    print(f"\n   Individual Event Impacts:")
    for event, impact in impacts.items():
        print(f"   - {event}: {impact['effect']:+.0f} steps ({impact['effect_percent']:+.1f}%)")
    
    # Find biggest effect
    biggest_event = max(impacts.items(), key=lambda x: abs(x[1]['effect']))
    print(f"\n   Biggest Impact: {biggest_event[0]} ({biggest_event[1]['effect']:+.0f} steps)")
    
    print("\n" + "-" * 70 + "\n")
    
    # Interpretation
    print(f"💡 INTERPRETATION:")
    print(f"   Holiday effects significantly impact step count predictions:")
    print(f"   • Vacation reduces steps by ~70% (relaxation, less walking)")
    print(f"   • Sick days reduce steps by ~85% (bed rest, recovery)")
    print(f"   • Marathon day increases steps by ~350% (race event)")
    print(f"   • Including holidays improves model accuracy by {results_dict['mae_improvement_percent']:.1f}%")
    
    print("\n" + "-" * 70 + "\n")
    
    # 30-day forecast preview
    forecast_preview = results_dict['forecast_comparison'].tail(5)
    print(f"📅 30-DAY FORECAST PREVIEW (Last 5 days):")
    for _, row in forecast_preview.iterrows():
        date_str = row['ds'].strftime('%Y-%m-%d')
        no_holidays = row['yhat_no_holidays']
        with_holidays = row['yhat_with_holidays']
        print(f"   {date_str}: {no_holidays:,.0f} steps (no holidays) vs {with_holidays:,.0f} steps (with holidays)")
    
    print("\n" + "=" * 70 + "\n")


def print_summary():
    """Print project completion summary"""
    print("\n" + "=" * 70)
    print(" " * 20 + "TASK 3 COMPLETED SUCCESSFULLY! ✓")
    print("=" * 70 + "\n")
    
    print("📁 OUTPUT FILES GENERATED:")
    print("   ├── outputs/plots/")
    print("   │   ├── step_comparison_plot.png     [With vs without holidays]")
    print("   │   ├── holiday_effects_plot.png     [Individual event impacts]")
    print("   │   └── step_forecast_plot.png       [30-day forecast comparison]")
    print("   ├── outputs/models/")
    print("   │   ├── step_model_no_holidays.pkl   [Model without holidays]")
    print("   │   └── step_model_with_holidays.pkl [Model with holidays]")
    print("   └── outputs/results/")
    print("       ├── step_forecast_results.csv    [Complete forecast data]")
    print("       └── holiday_impact_analysis.csv  [Holiday effects analysis]")
    
    print("\n📝 ASSIGNMENT DELIVERABLES:")
    print("   ✓ Code showing how to add holidays to Prophet")
    print("   ✓ Holidays DataFrame structure demonstration")
    print("   ✓ Two comparison plots (with vs without holidays)")
    print("   ✓ Quantified holiday impact analysis")
    print("   ✓ Identification of biggest effect event")
    print("   ✓ Complete documentation and interpretation")
    
    print("\n" + "=" * 70 + "\n")


def main():
    """
    Main execution function - Step Count with Holiday Effects Pipeline
    """
    try:
        # Print header
        print_header()
        
        # Step 1: Data Preparation + Holidays DataFrame
        df, holidays_df = prepare_step_data()
        
        # Step 2: Model Training (both versions)
        models = train_step_models_comparison(df, holidays_df)
        
        # Step 3: Forecasting & Comparison (30 days)
        results = make_step_forecast_comparison(models, df, holidays_df)
        
        # Step 4: Visualization
        create_step_comparison_plots(models, results, df)
        
        # Step 5: Results Analysis
        print_results(results)
        
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
