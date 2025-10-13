"""
Time Series Analysis Module
Prophet-based trend forecasting and anomaly detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import streamlit as st
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class TrendForecastingEngine:
    """Advanced time series forecasting using Facebook Prophet"""
    
    def __init__(self, confidence_interval=0.90):
        self.confidence_interval = confidence_interval
        self.trained_models = {}
        self.forecast_results = {}
        self.performance_metrics = {}
        self.anomaly_records = {}
    
    def forecast_all_metrics(self, data_dict, forecast_periods=120):
        """Forecast all metrics in the dataset"""
        results = {}
        
        for metric_name, df in data_dict.items():
            st.subheader(f"Forecasting {metric_name.title()}")
            
            forecast_df, metrics = self.train_and_predict(
                df, metric_name, forecast_periods
            )
            
            if not forecast_df.empty:
                anomalies = self.identify_anomalies(metric_name, threshold_sigma=2.5)
                
                results[metric_name] = {
                    'forecast': forecast_df,
                    'metrics': metrics,
                    'anomalies': anomalies,
                    'model': self.trained_models.get(metric_name)
                }
        
        return results
    
    def train_and_predict(self, df, metric_name, periods=120):
        """Train Prophet model and generate forecast"""
        
        st.info(f"🔄 Training Prophet model for {metric_name}...")
        
        metrics = {
            'metric_name': metric_name,
            'training_size': len(df),
            'forecast_horizon': periods,
            'timestamp': datetime.now()
        }
        
        try:
            value_col = self._get_value_column(metric_name, df)
            
            if value_col is None:
                st.warning(f"Could not identify value column for {metric_name}")
                st.write(f"Available columns: {list(df.columns)}")
                return pd.DataFrame(), metrics
            
            prophet_input = pd.DataFrame({
                'ds': pd.to_datetime(df['timestamp']),
                'y': df[value_col]
            })
            
            prophet_input = prophet_input.dropna()
            
            if len(prophet_input) < 10:
                st.warning(f"Insufficient data points ({len(prophet_input)}) for {metric_name}")
                return pd.DataFrame(), metrics
            
            with st.spinner("Configuring and training Prophet model..."):
                model = Prophet(
                    interval_width=self.confidence_interval,
                    daily_seasonality=True,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.10,
                    seasonality_prior_scale=15.0,
                    seasonality_mode='additive'
                )
                
                model.fit(prophet_input)
            
            future_df = model.make_future_dataframe(
                periods=periods,
                freq='min',
                include_history=True
            )
            
            forecast = model.predict(future_df)
            
            training_forecast = forecast[forecast['ds'].isin(prophet_input['ds'])]
            merged = prophet_input.merge(
                training_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                on='ds',
                how='left'
            )
            
            merged['residual'] = merged['y'] - merged['yhat']
            merged['abs_error'] = np.abs(merged['residual'])
            merged['squared_error'] = merged['residual'] ** 2
            merged['percentage_error'] = (merged['abs_error'] / merged['y'].abs()) * 100
            
            metrics.update({
                'mae': merged['abs_error'].mean(),
                'rmse': np.sqrt(merged['squared_error'].mean()),
                'mape': merged['percentage_error'].mean(),
                'r_squared': self._calculate_r_squared(merged['y'], merged['yhat']),
                'residual_mean': merged['residual'].mean(),
                'residual_std': merged['residual'].std(),
                'status': 'success'
            })
            
            self.trained_models[metric_name] = model
            self.forecast_results[metric_name] = {
                'forecast': forecast,
                'actual': prophet_input,
                'residuals': merged
            }
            self.performance_metrics[metric_name] = metrics
            
            st.success(
                f"✅ Model trained successfully! "
                f"MAE: {metrics['mae']:.2f}, "
                f"RMSE: {metrics['rmse']:.2f}, "
                f"MAPE: {metrics['mape']:.1f}%"
            )
            
            return forecast, metrics
            
        except Exception as e:
            metrics['status'] = 'failed'
            metrics['error'] = str(e)
            st.error(f"❌ Forecasting failed: {str(e)}")
            return pd.DataFrame(), metrics
    
    def _get_value_column(self, metric_name, df):
        """Identify the value column for a metric - FIXED VERSION"""
        
        # Debug: Show what columns we have
        st.write(f"DEBUG: DataFrame columns: {list(df.columns)}")
        
        # FIRST: Check if dataframe has exactly 2 columns (timestamp + value)
        if len(df.columns) == 2:
            # Return the second column (the value column)
            value_col = df.columns[1]
            st.info(f"Using column '{value_col}' as value column")
            return value_col
        
        # SECOND: Try to find specific column names
        column_map = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'activity': 'activity_level'
        }
        
        value_col = column_map.get(metric_name)
        
        if value_col and value_col in df.columns:
            return value_col
        
        # THIRD: Look for any numeric column that's not timestamp
        for col in df.columns:
            if col.lower() not in ['timestamp', 'time', 'date', 'datetime']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.info(f"Auto-detected numeric column: '{col}'")
                    return col
        
        # FALLBACK: Return second column if available
        if len(df.columns) >= 2:
            return df.columns[1]
        
        return None
    
    def _calculate_r_squared(self, actual, predicted):
        """Calculate R-squared metric"""
        ss_res = ((actual - predicted) ** 2).sum()
        ss_tot = ((actual - actual.mean()) ** 2).sum()
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, min(1.0, r_squared))
    
    def identify_anomalies(self, metric_name, threshold_sigma=2.5):
        """Identify anomalies using residual analysis"""
        
        if metric_name not in self.forecast_results:
            return pd.DataFrame()
        
        residuals_df = self.forecast_results[metric_name]['residuals']
        
        residual_mean = residuals_df['residual'].mean()
        residual_std = residuals_df['residual'].std()
        
        upper_threshold = residual_mean + (threshold_sigma * residual_std)
        lower_threshold = residual_mean - (threshold_sigma * residual_std)
        
        anomalies = residuals_df[
            (residuals_df['residual'] > upper_threshold) |
            (residuals_df['residual'] < lower_threshold)
        ].copy()
        
        if len(anomalies) > 0:
            anomalies['anomaly_score'] = np.abs(
                (anomalies['residual'] - residual_mean) / residual_std
            )
            
            anomalies['anomaly_type'] = anomalies['residual'].apply(
                lambda x: 'High' if x > upper_threshold else 'Low'
            )
            
            anomalies = anomalies.sort_values('anomaly_score', ascending=False)
            
            self.anomaly_records[metric_name] = anomalies
        
        return anomalies
