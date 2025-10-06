"""
Step Visualization Module - CORRECTED VERSION
Creates the TWO SEPARATE plots as required by assignment
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 8)
plt.rcParams['font.size'] = 10


def plot_without_holidays(models, actual_data, results, save_path='outputs/plots/plot_without_holidays.png'):
    """
    Plot 1: Model WITHOUT holidays (as required by assignment)
    
    Args:
        models (dict): Both trained models
        actual_data (pd.DataFrame): Historical data
        results (dict): Forecast results
        save_path (str): Path to save the plot
    """
    print("Creating Plot 1: WITHOUT holidays...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    forecast_no_holidays = results['forecast_no_holidays']
    
    # Plot actual data
    ax.plot(actual_data['ds'], actual_data['y'], 'ko-', 
            markersize=4, linewidth=1.5, label='Actual Step Count', alpha=0.8)
    
    # Plot forecast WITHOUT holidays
    ax.plot(forecast_no_holidays['ds'], forecast_no_holidays['yhat'], 
            'r-', linewidth=3, label='Forecast (NO Holidays)', alpha=0.9)
    
    # Plot confidence interval
    ax.fill_between(forecast_no_holidays['ds'], 
                    forecast_no_holidays['yhat_lower'], 
                    forecast_no_holidays['yhat_upper'],
                    alpha=0.3, color='red', label='95% Confidence Interval')
    
    # Highlight forecast period
    forecast_start = actual_data['ds'].max()
    ax.axvline(x=forecast_start, color='green', linestyle='--', 
               linewidth=2, label='30-Day Forecast Start', alpha=0.8)
    
    # Highlight known holiday periods in actual data (for context)
    # Vacation period
    vacation_start = actual_data['ds'].min() + pd.Timedelta(days=29)
    vacation_end = vacation_start + pd.Timedelta(days=7)
    ax.axvspan(vacation_start, vacation_end, alpha=0.2, color='orange', 
               label='Vacation Period (Not in Model)')
    
    # Sick period
    sick_start = actual_data['ds'].min() + pd.Timedelta(days=59)
    sick_end = sick_start + pd.Timedelta(days=2)
    ax.axvspan(sick_start, sick_end, alpha=0.2, color='red', 
               label='Sick Period (Not in Model)')
    
    # Marathon day
    marathon_day = actual_data['ds'].min() + pd.Timedelta(days=89)
    ax.axvline(x=marathon_day, color='green', linestyle=':', linewidth=3, 
               alpha=0.7, label='Marathon Day (Not in Model)')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Step Count', fontsize=12, fontweight='bold')
    ax.set_title('Step Count Forecast - MODEL WITHOUT HOLIDAYS\n' + 
                'Note: Model ignores vacation, sick days, and marathon effects', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot WITHOUT holidays saved to {save_path}")
    
    plt.close()


def plot_with_holidays(models, actual_data, results, holidays_df, save_path='outputs/plots/plot_with_holidays.png'):
    """
    Plot 2: Model WITH holidays (as required by assignment)
    
    Args:
        models (dict): Both trained models
        actual_data (pd.DataFrame): Historical data
        results (dict): Forecast results
        holidays_df (pd.DataFrame): Holidays dataframe
        save_path (str): Path to save the plot
    """
    print("Creating Plot 2: WITH holidays...")
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    forecast_with_holidays = results['forecast_with_holidays']
    
    # Plot actual data
    ax.plot(actual_data['ds'], actual_data['y'], 'ko-', 
            markersize=4, linewidth=1.5, label='Actual Step Count', alpha=0.8)
    
    # Plot forecast WITH holidays
    ax.plot(forecast_with_holidays['ds'], forecast_with_holidays['yhat'], 
            'b-', linewidth=3, label='Forecast (WITH Holidays)', alpha=0.9)
    
    # Plot confidence interval
    ax.fill_between(forecast_with_holidays['ds'], 
                    forecast_with_holidays['yhat_lower'], 
                    forecast_with_holidays['yhat_upper'],
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    
    # Highlight forecast period
    forecast_start = actual_data['ds'].max()
    ax.axvline(x=forecast_start, color='green', linestyle='--', 
               linewidth=2, label='30-Day Forecast Start', alpha=0.8)
    
    # Mark holiday periods with special styling
    holiday_colors = {'vacation': 'orange', 'sick': 'red', 'marathon': 'lime', 'marathon_recovery': 'yellow'}
    
    # Add holiday markers on actual data
    for holiday_type, color in holiday_colors.items():
        holiday_dates = holidays_df[holidays_df['holiday'] == holiday_type]['ds']
        for date in holiday_dates:
            if date in actual_data['ds'].values:
                step_value = actual_data[actual_data['ds'] == date]['y'].iloc[0]
                ax.scatter(date, step_value, s=150, color=color, 
                          edgecolor='black', linewidth=2, alpha=0.9, zorder=5)
    
    # Create custom legend for holidays
    from matplotlib.lines import Line2D
    holiday_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, markeredgecolor='black', label='Vacation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, markeredgecolor='black', label='Sick Days'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
               markersize=10, markeredgecolor='black', label='Marathon'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', 
               markersize=10, markeredgecolor='black', label='Marathon Recovery')
    ]
    
    # Main legend
    main_legend = ax.legend(loc='upper left', fontsize=10)
    # Holiday legend
    holiday_legend = ax.legend(handles=holiday_legend_elements, 
                              loc='upper right', fontsize=9, title='Holiday Events')
    ax.add_artist(main_legend)  # Add both legends
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Step Count', fontsize=12, fontweight='bold')
    ax.set_title('Step Count Forecast - MODEL WITH HOLIDAYS\n' + 
                'Note: Model accounts for vacation, sick days, and marathon effects', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot WITH holidays saved to {save_path}")
    
    plt.close()


def plot_side_by_side_comparison(models, actual_data, results, holidays_df, 
                                 save_path='outputs/plots/side_by_side_comparison.png'):
    """
    Bonus: Side-by-side comparison plot (for extra clarity)
    
    Args:
        models (dict): Both trained models
        actual_data (pd.DataFrame): Historical data
        results (dict): Forecast results
        holidays_df (pd.DataFrame): Holidays dataframe
        save_path (str): Path to save the plot
    """
    print("Creating Side-by-Side comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    forecast_no_holidays = results['forecast_no_holidays']
    forecast_with_holidays = results['forecast_with_holidays']
    
    # LEFT PLOT: WITHOUT holidays
    ax1.plot(actual_data['ds'], actual_data['y'], 'ko-', 
             markersize=3, linewidth=1.5, label='Actual', alpha=0.8)
    ax1.plot(forecast_no_holidays['ds'], forecast_no_holidays['yhat'], 
             'r-', linewidth=2, label='Forecast (No Holidays)')
    ax1.fill_between(forecast_no_holidays['ds'], 
                     forecast_no_holidays['yhat_lower'], 
                     forecast_no_holidays['yhat_upper'],
                     alpha=0.3, color='red', label='95% CI')
    
    forecast_start = actual_data['ds'].max()
    ax1.axvline(x=forecast_start, color='green', linestyle='--', 
                linewidth=2, alpha=0.7)
    
    ax1.set_title('WITHOUT Holidays', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontweight='bold')
    ax1.set_ylabel('Step Count', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # RIGHT PLOT: WITH holidays
    ax2.plot(actual_data['ds'], actual_data['y'], 'ko-', 
             markersize=3, linewidth=1.5, label='Actual', alpha=0.8)
    ax2.plot(forecast_with_holidays['ds'], forecast_with_holidays['yhat'], 
             'b-', linewidth=2, label='Forecast (With Holidays)')
    ax2.fill_between(forecast_with_holidays['ds'], 
                     forecast_with_holidays['yhat_lower'], 
                     forecast_with_holidays['yhat_upper'],
                     alpha=0.3, color='blue', label='95% CI')
    
    ax2.axvline(x=forecast_start, color='green', linestyle='--', 
                linewidth=2, alpha=0.7)
    
    # Mark holidays on right plot
    holiday_colors = {'vacation': 'orange', 'sick': 'red', 'marathon': 'lime', 'marathon_recovery': 'yellow'}
    for holiday_type, color in holiday_colors.items():
        holiday_dates = holidays_df[holidays_df['holiday'] == holiday_type]['ds']
        for date in holiday_dates:
            if date in actual_data['ds'].values:
                step_value = actual_data[actual_data['ds'] == date]['y'].iloc[0]
                ax2.scatter(date, step_value, s=80, color=color, 
                           edgecolor='black', linewidth=1, alpha=0.9, zorder=5)
    
    ax2.set_title('WITH Holidays', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Step Count', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Step Count Forecast Comparison: Impact of Holiday Effects', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Side-by-side comparison saved to {save_path}")
    
    plt.close()


def create_step_comparison_plots(models, results, actual_data):
    """
    Create all required plots for Task 3
    
    Args:
        models (dict): Both trained models
        results (dict): Forecast results
        actual_data (pd.DataFrame): Historical data
    """
    print("=" * 50)
    print("STEP 4: STEP VISUALIZATION - CORRECTED")
    print("=" * 50 + "\n")
    
    # Get holidays dataframe
    from step_data_preparation import create_holidays_dataframe
    holidays_df = create_holidays_dataframe()
    
    print("Creating the TWO REQUIRED PLOTS:\n")
    
    # Create the TWO SEPARATE plots as required by assignment
    plot_without_holidays(models, actual_data, results)
    plot_with_holidays(models, actual_data, results, holidays_df)
    
    # Bonus: Create side-by-side comparison
    plot_side_by_side_comparison(models, actual_data, results, holidays_df)
    
    # Keep the existing holiday effects plot (it's good supplementary material)
    print("Creating supplementary analysis plot...")
    plot_holiday_effects_analysis(actual_data, holidays_df, results)
    
    print("\n✓ All required plots created successfully!")
    print("  📁 ASSIGNMENT PLOTS:")
    print("     ├── plot_without_holidays.png    [Required Plot 1]")
    print("     └── plot_with_holidays.png       [Required Plot 2]")
    print("  📁 BONUS PLOTS:")
    print("     ├── side_by_side_comparison.png  [Clear comparison]") 
    print("     └── holiday_effects_analysis.png [Detailed breakdown]")
    print("\n")


def plot_holiday_effects_analysis(actual_data, holidays_df, results, 
                                 save_path='outputs/plots/holiday_effects_analysis.png'):
    """Keep your existing detailed analysis plot"""
    print("Creating detailed holiday effects analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Timeline with holiday markers
    ax1 = axes[0]
    ax1.plot(actual_data['ds'], actual_data['y'], 'ko-', 
             markersize=4, linewidth=1.5, alpha=0.7, label='Daily Steps')
    
    holiday_colors = {'vacation': 'orange', 'sick': 'red', 'marathon': 'green', 'marathon_recovery': 'yellow'}
    
    for holiday_type, color in holiday_colors.items():
        holiday_dates = holidays_df[holidays_df['holiday'] == holiday_type]['ds']
        for date in holiday_dates:
            if date in actual_data['ds'].values:
                step_value = actual_data[actual_data['ds'] == date]['y'].iloc[0]
                ax1.scatter(date, step_value, s=100, color=color, 
                           edgecolor='black', linewidth=2, alpha=0.8, zorder=5)
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                             markersize=8, markeredgecolor='black', label=holiday_type.title())
                      for holiday_type, color in holiday_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    ax1.set_title('Step Count Timeline with Holiday Events', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Step Count')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Holiday impact bar chart
    ax2 = axes[1]
    impacts = results['holiday_impacts']
    holidays = list(impacts.keys())
    effects = [impacts[h]['effect'] for h in holidays]
    colors_list = [holiday_colors.get(h, 'gray') for h in holidays]
    
    bars = ax2.bar(holidays, effects, color=colors_list, alpha=0.8, edgecolor='black')
    
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{effect:+.0f}', ha='center', va='center', 
                fontweight='bold', fontsize=11)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Holiday Impact on Step Count', fontweight='bold')
    ax2.set_xlabel('Holiday Type')
    ax2.set_ylabel('Step Count Effect')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Model performance comparison
    ax3 = axes[2]
    models_names = ['Without Holidays', 'With Holidays']
    mae_values = [results['mae_no_holidays'], results['mae_with_holidays']]
    
    bars = ax3.bar(models_names, mae_values, color=['lightcoral', 'lightblue'], 
                   alpha=0.8, edgecolor='black')
    
    improvement = results['mae_improvement']
    ax3.annotate(f'Improvement:\n{improvement:.0f} steps\n({results["mae_improvement_percent"]:.1f}%)',
                xy=(0.5, max(mae_values) * 0.8), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                fontweight='bold', fontsize=10)
    
    for bar, mae in zip(bars, mae_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 20,
                f'{mae:.0f}', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_title('Model Performance Comparison (MAE)', fontweight='bold')
    ax3.set_ylabel('Mean Absolute Error (steps)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Forecast difference over time
    ax4 = axes[3]
    future_comparison = results['forecast_comparison']
    
    ax4.plot(future_comparison['ds'], future_comparison['difference'], 
             'purple', linewidth=3, marker='o', markersize=6)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.fill_between(future_comparison['ds'], 0, future_comparison['difference'],
                     alpha=0.3, color='purple')
    
    ax4.set_title('Forecast Difference (With Holidays - Without)', fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Step Count Difference')
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Holiday Effects Analysis - Detailed Breakdown', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Holiday effects analysis saved to {save_path}")
    
    plt.close()


if __name__ == "__main__":
    from step_data_preparation import prepare_step_data
    from step_model_training import train_step_models_comparison
    from step_forecasting import make_step_forecast_comparison
    
    df, holidays_df = prepare_step_data()
    models = train_step_models_comparison(df, holidays_df)
    results = make_step_forecast_comparison(models, df, holidays_df)
    create_step_comparison_plots(models, results, df)
