"""
Model Training Module
Handles Prophet model initialization and training
"""

from prophet import Prophet
import pandas as pd
import pickle
import os


def initialize_prophet_model():
    """
    Initialize Prophet model with custom parameters
    
    Returns:
        Prophet: Initialized Prophet model
    """
    print("Initializing Prophet model...")
    
    # Initialize Prophet with custom parameters
    model = Prophet(
        daily_seasonality=False,      # No daily pattern (using daily averages)
        weekly_seasonality=True,       # Weekly patterns (exercise routines)
        yearly_seasonality=False,      # Not enough data for yearly patterns
        seasonality_mode='additive',   # Additive seasonality
        changepoint_prior_scale=0.05,  # Flexibility in trend changes
        interval_width=0.95            # 95% confidence intervals
    )
    
    print("✓ Prophet model initialized with parameters:")
    print("  - Weekly seasonality: Enabled")
    print("  - Daily seasonality: Disabled")
    print("  - Confidence interval: 95%")
    print("  - Changepoint prior scale: 0.05\n")
    
    return model


def train_prophet_model(df):
    """
    Train Prophet model on the provided data
    
    Args:
        df (pd.DataFrame): Formatted dataframe with 'ds' and 'y' columns
    
    Returns:
        Prophet: Trained Prophet model
    """
    print("=" * 50)
    print("STEP 2: MODEL TRAINING")
    print("=" * 50 + "\n")
    
    # Initialize model
    model = initialize_prophet_model()
    
    print(f"Training model on {len(df)} days of data...")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}\n")
    
    # Fit the model
    model.fit(df)
    
    print("✓ Model training completed successfully!\n")
    
    # Save the trained model
    save_model(model)
    
    return model


def save_model(model, file_path='outputs/models/prophet_model.pkl'):
    """
    Save trained Prophet model to file
    
    Args:
        model (Prophet): Trained Prophet model
        file_path (str): Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✓ Model saved to {file_path}\n")


def load_model(file_path='outputs/models/prophet_model.pkl'):
    """
    Load trained Prophet model from file
    
    Args:
        file_path (str): Path to the saved model
    
    Returns:
        Prophet: Loaded Prophet model
    """
    print(f"Loading model from {file_path}...")
    
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    
    print("✓ Model loaded successfully\n")
    return model


if __name__ == "__main__":
    # Test the module
    from data_preparation import prepare_data
    
    df = prepare_data()
    model = train_prophet_model(df)
    print("Model training test completed!")
