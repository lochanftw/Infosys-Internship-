"""
Advanced Feature Engineering Engine using TSFresh
Comprehensive statistical feature extraction from time-series health data

Features:
- Multi-window feature extraction with configurable overlap
- Three complexity levels (Basic, Standard, Advanced)
- Automatic feature cleaning and selection
- Feature importance ranking
- Correlation analysis
- Export capabilities
"""

import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters, 
    EfficientFCParameters, 
    MinimalFCParameters
)
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngine:
    """
    Advanced feature extraction engine using TSFresh
    
    Attributes:
        extraction_mode (str): Complexity level for feature extraction
        features (dict): Extracted features per metric
        feature_metadata (dict): Metadata about extraction process
        scaler (StandardScaler): Feature scaler for normalization
    """
    
    def __init__(self):
        """Initialize the feature extraction engine"""
        self.extraction_mode = "Standard"
        self.features = {}
        self.feature_metadata = {}
        self.scaler = StandardScaler()
        self.extraction_history = []
        
        logger.info("AdvancedFeatureEngine initialized")
    
    def extract_all_features(
        self, 
        data_dict, 
        window_size=60, 
        complexity="Standard",
        overlap_percentage=50,
        normalize=False
    ):
        """
        Extract comprehensive features from all metrics
        
        Args:
            data_dict (dict): Dictionary of {metric_name: DataFrame} pairs
            window_size (int): Rolling window size in minutes
            complexity (str): Feature complexity ('Basic', 'Standard', 'Advanced')
            overlap_percentage (int): Window overlap percentage (0-100)
            normalize (bool): Whether to normalize features
        
        Returns:
            dict: Extracted features per metric with metadata
        
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If feature extraction fails
        """
        
        logger.info(f"Starting feature extraction - Complexity: {complexity}, Window: {window_size}min")
        
        if not data_dict:
            raise ValueError("Input data dictionary is empty")
        
        results = {}
        extraction_stats = {
            'start_time': datetime.now(),
            'window_size': window_size,
            'complexity': complexity,
            'overlap_percentage': overlap_percentage,
            'metrics_processed': []
        }
        
        for metric_name, df in data_dict.items():
            try:
                logger.info(f"🔬 Processing metric: {metric_name}")
                
                # Validate input data
                self._validate_dataframe(df, metric_name)
                
                # Prepare time-series data with rolling windows
                prepared_data = self._prepare_timeseries_data(
                    df, window_size, metric_name, overlap_percentage
                )
                
                # Select feature parameters based on complexity
                fc_params = self._get_feature_parameters(complexity)
                
                # Extract features
                logger.info(f"  Extracting features with {complexity} complexity...")
                extracted = extract_features(
                    prepared_data,
                    column_id="window_id",
                    column_sort="timestamp",
                    column_value=df.columns[1],
                    default_fc_parameters=fc_params,
                    impute_function=impute,
                    disable_progressbar=False,
                    n_jobs=1
                )
                
                # Clean and process features
                extracted = self._clean_features(extracted)
                
                # Normalize if requested
                if normalize:
                    extracted = pd.DataFrame(
                        self.scaler.fit_transform(extracted),
                        columns=extracted.columns,
                        index=extracted.index
                    )
                
                # Calculate feature statistics
                feature_stats = self._calculate_feature_statistics(extracted, metric_name)
                
                # Store results
                results[metric_name] = extracted
                
                # Store metadata
                self.feature_metadata[metric_name] = {
                    'n_features': len(extracted.columns),
                    'n_windows': len(extracted),
                    'window_size': window_size,
                    'overlap': overlap_percentage,
                    'complexity': complexity,
                    'normalized': normalize,
                    'extraction_time': datetime.now(),
                    'statistics': feature_stats
                }
                
                extraction_stats['metrics_processed'].append(metric_name)
                
                logger.info(f"  ✅ Extracted {len(extracted.columns)} features from {len(extracted)} windows")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing {metric_name}: {str(e)}")
                raise RuntimeError(f"Feature extraction failed for {metric_name}: {str(e)}")
        
        # Finalize statistics
        extraction_stats['end_time'] = datetime.now()
        extraction_stats['duration'] = (
            extraction_stats['end_time'] - extraction_stats['start_time']
        ).total_seconds()
        
        self.extraction_history.append(extraction_stats)
        self.features = results
        
        logger.info(f"✅ Feature extraction complete. Duration: {extraction_stats['duration']:.2f}s")
        
        return results
    
    def _validate_dataframe(self, df, metric_name):
        """Validate input dataframe structure and content"""
        
        if df is None or df.empty:
            raise ValueError(f"DataFrame for {metric_name} is empty")
        
        if 'timestamp' not in df.columns:
            raise ValueError(f"DataFrame for {metric_name} missing 'timestamp' column")
        
        if len(df.columns) < 2:
            raise ValueError(f"DataFrame for {metric_name} must have at least 2 columns")
        
        # Check for sufficient data points
        if len(df) < 10:
            raise ValueError(f"DataFrame for {metric_name} has insufficient data (min 10 rows)")
        
        logger.debug(f"  Validation passed for {metric_name}: {len(df)} rows")
    
    def _prepare_timeseries_data(self, df, window_size, metric_name, overlap_percentage):
        """
        Prepare data in TSFresh format with rolling windows
        
        Args:
            df: Input dataframe
            window_size: Window size in minutes
            metric_name: Name of the metric
            overlap_percentage: Percentage of overlap between windows
        
        Returns:
            DataFrame formatted for TSFresh with window IDs
        """
        
        # Get value column (second column after timestamp)
        value_col = df.columns[1]
        
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate step size based on overlap
        step_size = max(1, int(window_size * (1 - overlap_percentage / 100)))
        
        logger.debug(f"  Window size: {window_size}, Step size: {step_size}")
        
        window_data = []
        window_id = 0
        
        # Create rolling windows
        for start_idx in range(0, len(df_sorted) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window = df_sorted.iloc[start_idx:end_idx].copy()
            window['window_id'] = window_id
            window_data.append(window)
            window_id += 1
        
        # Fallback if not enough data for windows
        if not window_data:
            logger.warning(f"  Insufficient data for windowing. Using full dataset.")
            df_sorted['window_id'] = 0
            return df_sorted[['window_id', 'timestamp', value_col]]
        
        combined = pd.concat(window_data, ignore_index=True)
        
        logger.debug(f"  Created {window_id} windows")
        
        return combined[['window_id', 'timestamp', value_col]]
    
    def _get_feature_parameters(self, complexity):
        """Get TSFresh feature parameters based on complexity level"""
        
        if complexity == "Basic":
            return MinimalFCParameters()
        elif complexity == "Advanced":
            return ComprehensiveFCParameters()
        else:  # Standard
            return EfficientFCParameters()
    
    def _clean_features(self, features):
        """
        Clean extracted features by removing invalid values
        
        Removes:
        - Columns with NaN values
        - Columns with infinite values
        - Constant columns (zero variance)
        - Highly correlated columns (>0.95)
        """
        
        initial_count = len(features.columns)
        
        # Remove NaN columns
        features = features.dropna(axis=1)
        after_nan = len(features.columns)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
        after_inf = len(features.columns)
        
        # Remove constant columns (zero variance)
        features = features.loc[:, features.std() > 1e-10]
        after_const = len(features.columns)
        
        # Remove highly correlated features
        if len(features.columns) > 1:
            corr_matrix = features.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper_triangle.columns 
                      if any(upper_triangle[column] > 0.95)]
            features = features.drop(columns=to_drop)
        
        final_count = len(features.columns)
        
        logger.debug(f"  Feature cleaning: {initial_count} → {final_count} features")
        logger.debug(f"    Removed: NaN({initial_count-after_nan}), "
                    f"Inf({after_nan-after_inf}), "
                    f"Const({after_inf-after_const}), "
                    f"Corr({after_const-final_count})")
        
        return features
    
    def _calculate_feature_statistics(self, features, metric_name):
        """Calculate comprehensive statistics for extracted features"""
        
        stats = {
            'mean': features.mean().to_dict(),
            'std': features.std().to_dict(),
            'min': features.min().to_dict(),
            'max': features.max().to_dict(),
            'variance': features.var().to_dict(),
            'top_features_by_variance': features.var().nlargest(10).to_dict()
        }
        
        return stats
    
    def get_top_features(self, metric_name, n=10, method='variance'):
        """
        Get top N important features
        
        Args:
            metric_name: Name of the metric
            n: Number of top features to return
            method: Selection method ('variance', 'mean', 'std')
        
        Returns:
            Series of top features with scores
        """
        
        if metric_name not in self.features:
            logger.error(f"Metric {metric_name} not found in extracted features")
            return None
        
        features = self.features[metric_name]
        
        if method == 'variance':
            scores = features.var().sort_values(ascending=False)
        elif method == 'mean':
            scores = features.mean().abs().sort_values(ascending=False)
        elif method == 'std':
            scores = features.std().sort_values(ascending=False)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return scores.head(n)
    
    def get_feature_summary(self, metric_name):
        """
        Get comprehensive summary of extracted features
        
        Returns:
            Dictionary with complete feature information
        """
        
        if metric_name not in self.features:
            return None
        
        features = self.features[metric_name]
        metadata = self.feature_metadata.get(metric_name, {})
        
        summary = {
            'metric_name': metric_name,
            'n_features': len(features.columns),
            'n_windows': len(features),
            'feature_names': list(features.columns),
            'metadata': metadata,
            'statistics': metadata.get('statistics', {}),
            'extraction_date': metadata.get('extraction_time', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return summary
    
    def export_features(self, metric_name, filepath):
        """Export features to CSV file"""
        
        if metric_name not in self.features:
            raise ValueError(f"Metric {metric_name} not found")
        
        self.features[metric_name].to_csv(filepath, index=True)
        logger.info(f"Features exported to {filepath}")
    
    def get_feature_correlation_matrix(self, metric_name):
        """Get correlation matrix for features"""
        
        if metric_name not in self.features:
            return None
        
        return self.features[metric_name].corr()
    
    def get_extraction_history(self):
        """Get history of all feature extractions"""
        return self.extraction_history
