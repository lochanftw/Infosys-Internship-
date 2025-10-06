"""
Sleep Model Training Module
Handles Prophet model initialization and training with weekly seasonality
"""

from prophet import Prophet
import pandas as pd
import pickle
import os


def initialize_sleep_prophet_model():
    """
    Initialize Prophet model with weekly seasonality for sleep analysis
    
    Returns:
        Prophet: Initialized Prophet model with sleep-specific parameters
    """
    print("Initializing Prophet model for sleep analysis...")
    
    # Initialize Prophet with parameters optimized for sleep data
    model = Prophet(
        daily_seasonality=False,        # No daily pattern (using daily sleep totals)
        weekly_seasonality=True,        # Strong weekly patterns in sleep
        yearly_seasonality=False,       # Not enough data for yearly patterns  
        seasonality_mode='additive',    # Weekly patterns add to base sleep
        changepoint_prior_scale=0.1,    # Allow for some trend changes
        interval_width=0.95,            # 95% confidence intervals
        seasonality_prior_scale=10.0    # Strong weekly seasonality
    )
    
    print("✓ Prophet model initialized for sleep analysis:")
    print("  - Weekly seasonality: ENABLED (strong)")
    print("  - Daily seasonality: Disabled")  
    print("  - Changepoint flexibility: Moderate")
    print("  - Confidence interval: 95%")
    print("  - Seasonality strength: High (captures weekly patterns)\n")
    
    return model


def train_sleep_prophet_model(df):
    """
    Train Prophet model on sleep data
    
    Args:
        df (pd.DataFrame): Formatted dataframe with 'ds' and 'y' columns
    
    Returns:
        Prophet: Trained Prophet model
    """
    print("=" * 50)
    print("STEP 2: SLEEP MODEL TRAINING")
    print("=" * 50 + "\n")
    
    # Initialize model
    model = initialize_sleep_prophet_model()
    
    print(f"Training sleep model on {len(df)} days of data...")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"Sleep range: {df['y'].min():.1f} - {df['y'].max():.1f} hours\n")
    
    # Fit the model
    model.fit(df)
    
    print("✓ Sleep model training completed successfully!")
    print("  - Weekly patterns learned")
    print("  - Trend components identified") 
    print("  - Model ready for forecasting\n")
    
    # Save the trained model
    save_sleep_model(model)
    
    return model


def save_sleep_model(model, file_path='outputs/models/sleep_prophet_model.pkl'):
    """
    Save trained Prophet sleep model to file
    
    Args:
        model (Prophet): Trained Prophet model
        file_path (str): Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Sleep model saved to {file_path}\n")


def load_sleep_model(file_path='outputs/models/sleep_prophet_model.pkl'):
    """
    Load trained Prophet sleep model from file
    
    Args:
        file_path (str): Path to the saved model
    
    Returns:
        Prophet: Loaded Prophet model
    """
    print(f"Loading sleep model from {file_path}...")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print("✓ Sleep model loaded successfully\n")
    return model


if __name__ == "__main__":
    # Test the module
    from sleep_data_preparation import prepare_sleep_data
    
    df = prepare_sleep_data()
    model = train_sleep_prophet_model(df)
    print("Sleep model training test completed!")
