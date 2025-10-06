"""
Step Model Training Module
Trains Prophet models with and without holidays for comparison
"""

from prophet import Prophet
import pandas as pd
import pickle
import os


def create_model_without_holidays():
    """Create Prophet model without holidays"""
    print("Initializing Prophet model WITHOUT holidays...")
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        interval_width=0.95
    )
    
    print("✓ Model without holidays initialized")
    return model


def create_model_with_holidays(holidays_df):
    """Create Prophet model with holidays"""
    print("Initializing Prophet model WITH holidays...")
    
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        interval_width=0.95,
        holidays=holidays_df  # ← KEY: Adding holidays here
    )
    
    print("✓ Model with holidays initialized")
    print(f"  - Holidays included: {len(holidays_df)} events")
    print(f"  - Holiday types: {list(holidays_df['holiday'].unique())}")
    
    return model


def train_step_models_comparison(df, holidays_df):
    """
    Train both models (with and without holidays) for comparison
    
    Args:
        df (pd.DataFrame): Formatted step count data
        holidays_df (pd.DataFrame): Holidays dataframe
    
    Returns:
        dict: Dictionary containing both trained models
    """
    print("=" * 50)
    print("STEP 2: STEP MODEL TRAINING COMPARISON")
    print("=" * 50 + "\n")
    
    print(f"Training models on {len(df)} days of step count data...")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Step range: {df['y'].min():,} - {df['y'].max():,} steps\n")
    
    # Train model WITHOUT holidays
    print("Training Model 1: WITHOUT holidays...")
    model_no_holidays = create_model_without_holidays()
    model_no_holidays.fit(df)
    print("✓ Model without holidays trained\n")
    
    # Train model WITH holidays
    print("Training Model 2: WITH holidays...")
    model_with_holidays = create_model_with_holidays(holidays_df)
    model_with_holidays.fit(df)
    print("✓ Model with holidays trained\n")
    
    # Save both models
    save_models(model_no_holidays, model_with_holidays)
    
    models = {
        'no_holidays': model_no_holidays,
        'with_holidays': model_with_holidays
    }
    
    print("✓ Both models training completed successfully!\n")
    
    return models


def save_models(model_no_holidays, model_with_holidays):
    """Save both trained models"""
    os.makedirs('outputs/models', exist_ok=True)
    
    # Save model without holidays
    with open('outputs/models/step_model_no_holidays.pkl', 'wb') as f:
        pickle.dump(model_no_holidays, f)
    
    # Save model with holidays
    with open('outputs/models/step_model_with_holidays.pkl', 'wb') as f:
        pickle.dump(model_with_holidays, f)
    
    print("✓ Both models saved to outputs/models/\n")


if __name__ == "__main__":
    # Test the module
    from step_data_preparation import prepare_step_data
    
    df, holidays_df = prepare_step_data()
    models = train_step_models_comparison(df, holidays_df)
    print("Step model training test completed!")
