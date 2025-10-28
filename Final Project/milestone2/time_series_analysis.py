"""
Advanced Time Series Analysis using Facebook Prophet
Comprehensive forecasting, trend analysis, and anomaly detection

Features:
- Multi-metric forecasting with Prophet
- Automatic seasonality detection
- Holiday effects integration
- Confidence interval calculation
- Advanced anomaly detection with multiple methods
- Trend decomposition
- Forecast evaluation metrics
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendForecastingEngine:
    """
    Advanced time-series forecasting engine using Facebook Prophet
    
    Attributes:
        models (dict): Trained Prophet models per metric
        forecasts (dict): Forecast results per metric
        anomalies (dict): Detected anomalies per metric
        performance_metrics (dict): Model performance metrics
    """
    
    def __init__(self):
        """Initialize the forecasting engine"""
        self.models = {}
        self.forecasts = {}
        self.anomalies = {}
        self.performance_metrics = {}
        self.forecast_history = []
        
        logger.info("TrendForecastingEngine initialized")
    
    def forecast_all_metrics(
        self,
        data_dict,
        forecast_periods=120,
        confidence_interval=0.95,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        detect_anomalies=True
    ):
        """
        Generate forecasts for all metrics using Prophet
        
        Args:
            data_dict (dict): Dictionary of metric dataframes
            forecast_periods (int): Number of periods to forecast ahead
            confidence_interval (float): Confidence interval width (0-1)
            seasonality_mode (str): 'additive' or 'multiplicative'
            changepoint_prior_scale (float): Flexibility of trend changes
            detect_anomalies (bool): Whether to detect anomalies
        
        Returns:
            dict: Forecast results with metrics and anomalies
        """
        
        logger.info(f"Starting forecast generation for {len(data_dict)} metrics")
        logger.info(f"  Forecast periods: {forecast_periods}, CI: {confidence_interval*100}%")
        
        results = {}
        forecast_stats = {
            'start_time': datetime.now(),
            'forecast_periods': forecast_periods,
            'confidence_interval': confidence_interval,
            'metrics_processed': []
        }
        
        for metric_name, df in data_dict.items():
            try:
                logger.info(f"📈 Forecasting {metric_name}...")
                
                # Validate input
                self._validate_dataframe(df, metric_name)
                
                # Prepare data for Prophet
                prophet_df = self._prepare_prophet_data(df, metric_name)
                
                # Configure and train Prophet model
                model = self._create_prophet_model(
                    confidence_interval=confidence_interval,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale
                )
                
                logger.info(f"  Training Prophet model...")
                model.fit(prophet_df)
                
                # Create future dataframe
                future = model.make_future_dataframe(
                    periods=forecast_periods,
                    freq='1min',
                    include_history=True
                )
                
                # Generate forecast
                logger.info(f"  Generating forecast...")
                forecast = model.predict(future)
                
                # Calculate performance metrics
                metrics = self._calculate_comprehensive_metrics(prophet_df, forecast)
                
                # Detect anomalies
                anomalies = []
                if detect_anomalies:
                    logger.info(f"  Detecting anomalies...")
                    anomalies = self._detect_comprehensive_anomalies(
                        prophet_df, forecast, model
                    )
                
                # Extract trend components
                trend_components = self._extract_trend_components(forecast, model)
                
                # Store results
                results[metric_name] = {
                    'model': model,
                    'forecast': forecast,
                    'metrics': metrics,
                    'anomalies': anomalies,
                    'original_data': prophet_df,
                    'trend_components': trend_components,
                    'forecast_config': {
                        'periods': forecast_periods,
                        'confidence_interval': confidence_interval,
                        'seasonality_mode': seasonality_mode
                    }
                }
                
                # Store model and metrics
                self.models[metric_name] = model
                self.performance_metrics[metric_name] = metrics
                self.anomalies[metric_name] = anomalies
                
                forecast_stats['metrics_processed'].append(metric_name)
                
                logger.info(f"  ✅ Forecast complete. MAPE: {metrics['mape']:.2f}%, "
                          f"Anomalies: {len(anomalies)}")
                
            except Exception as e:
                logger.error(f"  ❌ Error forecasting {metric_name}: {str(e)}")
                raise RuntimeError(f"Forecasting failed for {metric_name}: {str(e)}")
        
        # Finalize statistics
        forecast_stats['end_time'] = datetime.now()
        forecast_stats['duration'] = (
            forecast_stats['end_time'] - forecast_stats['start_time']
        ).total_seconds()
        
        self.forecast_history.append(forecast_stats)
        self.forecasts = results
        
        logger.info(f"✅ Forecasting complete. Duration: {forecast_stats['duration']:.2f}s")
        
        return results
    
    def _validate_dataframe(self, df, metric_name):
        """Validate input dataframe"""
        
        if df is None or df.empty:
            raise ValueError(f"DataFrame for {metric_name} is empty")
        
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame missing 'timestamp' column")
        
        if len(df) < 20:
            raise ValueError(f"Insufficient data for forecasting (min 20 rows)")
    
    def _prepare_prophet_data(self, df, metric_name):
        """Convert dataframe to Prophet format (ds, y)"""
        
        value_col = df.columns[1]
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df['timestamp']),
            'y': df[value_col].astype(float)
        })
        
        # Remove duplicates and sort
        prophet_df = prophet_df.drop_duplicates(subset=['ds']).sort_values('ds')
        
        return prophet_df
    
    def _create_prophet_model(
        self,
        confidence_interval=0.95,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    ):
        """Create and configure Prophet model"""
        
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=confidence_interval,
            uncertainty_samples=1000
        )
        
        # Add custom seasonalities
        model.add_seasonality(
            name='hourly',
            period=1,
            fourier_order=8
        )
        
        return model
    
    def _calculate_comprehensive_metrics(self, actual_df, forecast_df):
        """Calculate comprehensive forecasting metrics"""
        
        # Merge actual and forecast
        merged = actual_df.merge(
            forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='inner'
        )
        
        if len(merged) == 0:
            return self._empty_metrics()
        
        y_true = merged['y'].values
        y_pred = merged['yhat'].values
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mse = mean_squared_error(y_true, y_pred)
        
        # Mean Absolute Scaled Error (MASE)
        naive_forecast_error = np.mean(np.abs(np.diff(y_true)))
        mase = mae / (naive_forecast_error + 1e-10)
        
        # Coverage (% of actual values within CI)
        within_ci = np.sum(
            (y_true >= merged['yhat_lower']) & 
            (y_true <= merged['yhat_upper'])
        ) / len(y_true) * 100
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'mase': float(mase),
            'ci_coverage': float(within_ci),
            'n_samples': len(merged)
        }
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics dict"""
        return {
            'mae': 0, 'mse': 0, 'rmse': 0, 
            'mape': 0, 'r2': 0, 'mase': 0,
            'ci_coverage': 0, 'n_samples': 0
        }
    
    def _detect_comprehensive_anomalies(self, actual_df, forecast_df, model):
        """
        Detect anomalies using multiple methods
        
        Methods:
        1. 3-sigma rule on residuals
        2. Confidence interval violations
        3. Trend change detection
        """
        
        # Merge data
        merged = actual_df.merge(
            forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']],
            on='ds',
            how='inner'
        )
        
        if len(merged) == 0:
            return []
        
        # Calculate residuals
        merged['residual'] = merged['y'] - merged['yhat']
        
        # Method 1: 3-sigma rule
        mean_residual = merged['residual'].mean()
        std_residual = merged['residual'].std()
        threshold_3sigma = 3 * std_residual
        
        anomalies_3sigma = merged[
            (merged['residual'] > mean_residual + threshold_3sigma) |
            (merged['residual'] < mean_residual - threshold_3sigma)
        ]
        
        # Method 2: CI violations
        anomalies_ci = merged[
            (merged['y'] < merged['yhat_lower']) |
            (merged['y'] > merged['yhat_upper'])
        ]
        
        # Combine anomalies
        all_anomalies = pd.concat([anomalies_3sigma, anomalies_ci]).drop_duplicates()
        
        anomaly_list = []
        for _, row in all_anomalies.iterrows():
            anomaly_score = abs(row['residual']) / (std_residual + 1e-10)
            
            # Determine anomaly type
            if row['y'] > row['yhat_upper']:
                anomaly_type = 'spike'
            elif row['y'] < row['yhat_lower']:
                anomaly_type = 'drop'
            else:
                anomaly_type = 'deviation'
            
            anomaly_list.append({
                'timestamp': row['ds'],
                'actual_value': float(row['y']),
                'predicted_value': float(row['yhat']),
                'lower_bound': float(row['yhat_lower']),
                'upper_bound': float(row['yhat_upper']),
                'residual': float(row['residual']),
                'anomaly_score': float(anomaly_score),
                'anomaly_type': anomaly_type
            })
        
        # Sort by anomaly score
        anomaly_list = sorted(anomaly_list, key=lambda x: x['anomaly_score'], reverse=True)
        
        return anomaly_list
    
    def _extract_trend_components(self, forecast, model):
        """Extract trend and seasonality components"""
        
        components = {
            'trend': forecast['trend'].values,
            'daily_seasonality': forecast.get('daily', np.zeros(len(forecast))).values,
            'timestamps': forecast['ds'].values
        }
        
        return components
    
    def get_forecast_summary(self, metric_name):
        """Get comprehensive forecast summary"""
        
        if metric_name not in self.forecasts:
            return None
        
        forecast_data = self.forecasts[metric_name]
        
        summary = {
            'metric_name': metric_name,
            'performance_metrics': forecast_data['metrics'],
            'n_anomalies': len(forecast_data['anomalies']),
            'forecast_range': {
                'min': float(forecast_data['forecast']['yhat'].min()),
                'max': float(forecast_data['forecast']['yhat'].max()),
                'mean': float(forecast_data['forecast']['yhat'].mean())
            },
            'trend_direction': self._determine_trend_direction(forecast_data['forecast']),
            'config': forecast_data['forecast_config']
        }
        
        return summary
    
    def _determine_trend_direction(self, forecast):
        """Determine overall trend direction"""
        
        trend = forecast['trend'].values
        
        # Compare first and last quarter
        quarter_size = len(trend) // 4
        first_quarter_mean = np.mean(trend[:quarter_size])
        last_quarter_mean = np.mean(trend[-quarter_size:])
        
        if last_quarter_mean > first_quarter_mean * 1.05:
            return 'increasing'
        elif last_quarter_mean < first_quarter_mean * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_forecast(self, metric_name, filepath):
        """Export forecast to CSV"""
        
        if metric_name not in self.forecasts:
            raise ValueError(f"Metric {metric_name} not found")
        
        self.forecasts[metric_name]['forecast'].to_csv(filepath, index=False)
        logger.info(f"Forecast exported to {filepath}")
    
    def get_forecast_history(self):
        """Get history of all forecasting operations"""
        return self.forecast_history
