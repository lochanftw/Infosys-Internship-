"""
Feature Engineering Module
Advanced statistical feature extraction using TSFresh
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import streamlit as st

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters, ComprehensiveFCParameters


class AdvancedFeatureEngine:
    """
    Advanced Feature Extraction Engine using TSFresh
    Extracts time-domain and frequency-domain features
    """
    
    def __init__(self, extraction_mode: str = 'optimized'):
        """
        Initialize feature engine
        
        Args:
            extraction_mode: 'basic', 'optimized', or 'comprehensive'
        """
        self.extraction_mode = extraction_mode
        self.feature_databases = {}
        self.extraction_logs = {}
        self.feature_importance_cache = {}
        
    def extract_all_features(self, data_dict: Dict[str, pd.DataFrame],
                            window_size: int = 45,
                            complexity: str = 'Standard') -> Dict:
        """
        Extract features from all metrics
        
        Args:
            data_dict: Dictionary of dataframes
            window_size: Window size in minutes
            complexity: Feature complexity level
            
        Returns:
            Dictionary containing all extracted features
        """
        results = {}
        
        for metric_name, df in data_dict.items():
            st.subheader(f"Processing {metric_name.title()}")
            
            feature_matrix, report = self.extract_metric_features(
                df, metric_name, window_size, complexity
            )
            
            if not feature_matrix.empty:
                results[metric_name] = {
                    'features': feature_matrix,
                    'report': report,
                    'top_features': self.rank_features(feature_matrix, top_n=12)
                }
        
        return results
    
    def extract_metric_features(self, df: pd.DataFrame, metric_name: str,
                                window_size: int, complexity: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Extract features for a single metric
        
        Args:
            df: Input dataframe
            metric_name: Name of the metric
            window_size: Window size for feature extraction
            complexity: Complexity level
            
        Returns:
            Feature matrix and extraction report
        """
        st.info(f"🔄 Extracting {complexity} features from {metric_name}...")
        
        report = {
            'metric': metric_name,
            'input_records': len(df),
            'window_size': window_size,
            'complexity': complexity,
            'start_time': datetime.now()
        }
        
        try:
            # Prepare windowed data
            windowed_data = self._create_feature_windows(df, metric_name, window_size)
            
            if windowed_data is None or len(windowed_data) == 0:
                st.warning(f"⚠️ No valid data for {metric_name}")
                return pd.DataFrame(), report
            
            # Select feature parameters
            feature_params = self._select_feature_parameters(complexity)
            
            # Progress tracking
            progress = st.progress(0, text="Initializing feature extraction...")
            
            # Extract features with TSFresh
            progress.progress(20, text="Extracting time-series features...")
            
            feature_matrix = extract_features(
                windowed_data,
                column_id='window_id',
                column_sort='timestamp',
                default_fc_parameters=feature_params,
                disable_progressbar=True,
                n_jobs=1  # Single job for Streamlit compatibility
            )
            
            progress.progress(60, text="Imputing missing values...")
            
            # Handle missing values
            feature_matrix = impute(feature_matrix)
            
            progress.progress(80, text="Cleaning feature matrix...")
            
            # Clean features
            feature_matrix = self._sanitize_features(feature_matrix)
            
            progress.progress(100, text="Feature extraction complete!")
            
            # Generate report
            elapsed_time = (datetime.now() - report['start_time']).total_seconds()
            
            report.update({
                'features_count': len(feature_matrix.columns),
                'window_count': len(feature_matrix),
                'extraction_time_sec': elapsed_time,
                'avg_time_per_window': elapsed_time / len(feature_matrix) if len(feature_matrix) > 0 else 0,
                'status': 'success'
            })
            
            # Cache results
            self.feature_databases[metric_name] = feature_matrix
            self.extraction_logs[metric_name] = report
            
            st.success(f"✅ Extracted {report['features_count']} features from {report['window_count']} windows")
            
            return feature_matrix, report
            
        except Exception as e:
            report['status'] = 'failed'
            report['error_message'] = str(e)
            st.error(f"❌ Feature extraction failed: {str(e)}")
            return pd.DataFrame(), report
    
    def _create_feature_windows(self, df: pd.DataFrame, metric_name: str,
                               window_size: int) -> Optional[pd.DataFrame]:
        """
        Create rolling windows for feature extraction
        
        Args:
            df: Input dataframe
            metric_name: Metric identifier
            window_size: Window size in minutes
            
        Returns:
            Windowed dataframe in TSFresh format
        """
        # Map metric names to column names
        column_mapping = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'activity': 'activity_level',
            'sleep': 'duration_minutes'
        }
        
        value_column = column_mapping.get(metric_name)
        
        if value_column not in df.columns:
            st.warning(f"Column '{value_column}' not found in dataframe")
            return None
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create overlapping windows (50% overlap)
        window_step = max(1, window_size // 2)
        windows = []
        window_id = 0
        
        for start_idx in range(0, len(df_sorted) - window_size + 1, window_step):
            end_idx = start_idx + window_size
            
            window_df = df_sorted.iloc[start_idx:end_idx].copy()
            window_df['window_id'] = window_id
            
            # Select only required columns
            windows.append(window_df[['window_id', 'timestamp', value_column]])
            window_id += 1
        
        if not windows:
            return None
        
        # Combine all windows
        windowed_df = pd.concat(windows, ignore_index=True)
        
        # Rename value column to generic name
        windowed_df = windowed_df.rename(columns={value_column: 'value'})
        
        return windowed_df
    
    def _select_feature_parameters(self, complexity: str) -> Dict:
        """
        Select TSFresh feature parameters based on complexity
        
        Args:
            complexity: Complexity level
            
        Returns:
            Feature parameter dictionary
        """
        if complexity == 'Basic':
            return {
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "range_count": [{"min": 0, "max": 1}]
            }
        
        elif complexity == 'Standard':
            return {
                "mean": None,
                "median": None,
                "standard_deviation": None,
                "variance": None,
                "minimum": None,
                "maximum": None,
                "sum_values": None,
                "abs_energy": None,
                "quantile": [{"q": 0.1}, {"q": 0.25}, {"q": 0.75}, {"q": 0.9}],
                "skewness": None,
                "kurtosis": None,
                "count_above_mean": None,
                "count_below_mean": None,
                "mean_change": None,
                "absolute_sum_of_changes": None,
                "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}],
                "autocorrelation": [{"lag": 1}, {"lag": 2}]
            }
        
        else:  # Advanced - Comprehensive
            return ComprehensiveFCParameters()
    
    def _sanitize_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and sanitize feature matrix
        
        Args:
            feature_matrix: Raw feature matrix
            
        Returns:
            Cleaned feature matrix
        """
        # Remove features with zero variance
        zero_var_cols = [col for col in feature_matrix.columns 
                        if feature_matrix[col].std() == 0]
        
        if zero_var_cols:
            st.info(f"Removing {len(zero_var_cols)} constant features")
            feature_matrix = feature_matrix.drop(columns=zero_var_cols)
        
        # Remove features with excessive NaN values (>50%)
        nan_threshold = 0.5
        high_nan_cols = [col for col in feature_matrix.columns 
                        if feature_matrix[col].isna().sum() / len(feature_matrix) > nan_threshold]
        
        if high_nan_cols:
            st.info(f"Removing {len(high_nan_cols)} features with excessive missing values")
            feature_matrix = feature_matrix.drop(columns=high_nan_cols)
        
        # Remove infinite values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        return feature_matrix
    
    def rank_features(self, feature_matrix: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Rank features by variance and statistical properties
        
        Args:
            feature_matrix: Feature matrix
            top_n: Number of top features to return
            
        Returns:
            DataFrame with ranked features
        """
        if feature_matrix.empty:
            return pd.DataFrame()
        
        # Calculate feature statistics
        feature_stats = pd.DataFrame({
            'Feature': feature_matrix.columns,
            'Variance': feature_matrix.var().values,
            'Mean': feature_matrix.mean().values,
            'Std': feature_matrix.std().values,
            'Min': feature_matrix.min().values,
            'Max': feature_matrix.max().values,
            'Range': feature_matrix.max().values - feature_matrix.min().values
        })
        
        # Rank by variance
        feature_stats = feature_stats.sort_values('Variance', ascending=False)
        
        # Get top N
        top_features = feature_stats.head(top_n).reset_index(drop=True)
        
        # Simplify feature names for display
        top_features['Feature_Short'] = top_features['Feature'].apply(
            lambda x: x.split('__')[1] if '__' in x else x
        )
        
        return top_features
    
    def get_feature_summary(self, metric_name: str) -> Dict:
        """
        Get comprehensive feature summary
        
        Args:
            metric_name: Metric identifier
            
        Returns:
            Summary dictionary
        """
        if metric_name not in self.feature_databases:
            return {}
        
        fm = self.feature_databases[metric_name]
        
        return {
            'total_features': len(fm.columns),
            'total_windows': len(fm),
            'feature_categories': self._categorize_features(fm.columns),
            'statistical_summary': {
                'mean_features': fm.mean().mean(),
                'std_features': fm.std().mean(),
                'sparsity': (fm == 0).sum().sum() / (fm.shape[0] * fm.shape[1])
            }
        }
    
    def _categorize_features(self, feature_names: pd.Index) -> Dict[str, int]:
        """Categorize features by type"""
        categories = {
            'statistical': 0,
            'temporal': 0,
            'frequency': 0,
            'complexity': 0,
            'other': 0
        }
        
        for feat in feature_names:
            feat_lower = feat.lower()
            
            if any(x in feat_lower for x in ['mean', 'std', 'var', 'median', 'quantile', 'min', 'max']):
                categories['statistical'] += 1
            elif any(x in feat_lower for x in ['trend', 'slope', 'change', 'diff']):
                categories['temporal'] += 1
            elif any(x in feat_lower for x in ['fft', 'spectral', 'frequency', 'cwt']):
                categories['frequency'] += 1
            elif any(x in feat_lower for x in ['entropy', 'complexity', 'approximate']):
                categories['complexity'] += 1
            else:
                categories['other'] += 1
        
        return categories
