"""
Timestamp Validation and Quality Assessment for Task 3
Validates timestamp conversions and data quality
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class TimestampValidator:
    """Comprehensive timestamp validation and quality assessment"""
    
    def __init__(self):
        self.validation_rules = {
            'future_timestamps': True,      # Check for future dates
            'reasonable_range': True,       # Check for reasonable date range
            'duplicate_detection': True,    # Find duplicate timestamps
            'gap_analysis': True,          # Analyze time gaps
            'timezone_consistency': True    # Check timezone consistency
        }
    
    def validate_timestamp_conversion(self, original_df: pd.DataFrame, 
                                    converted_df: pd.DataFrame) -> Dict:
        """
        Comprehensive validation of timestamp conversion
        
        Parameters:
        -----------
        original_df : pd.DataFrame
            Original data before conversion
        converted_df : pd.DataFrame
            Data after timezone conversion
        
        Returns:
        --------
        Dict with validation results
        """
        
        logger.info("🔍 Validating timestamp conversion")
        
        validation_report = {
            'conversion_summary': {},
            'data_quality': {},
            'issues_found': [],
            'recommendations': [],
            'statistics': {},
            'validation_passed': True
        }
        
        # Find timestamp columns
        orig_ts_col = self._find_timestamp_column(original_df)
        conv_ts_col = self._find_timestamp_column(converted_df)
        
        if not orig_ts_col or not conv_ts_col:
            validation_report['validation_passed'] = False
            validation_report['issues_found'].append('Timestamp column not found')
            return validation_report
        
        # Basic conversion statistics
        validation_report['conversion_summary'] = self._analyze_conversion_summary(
            original_df, converted_df, orig_ts_col, conv_ts_col
        )
        
        # Data quality checks
        validation_report['data_quality'] = self._analyze_data_quality(
            converted_df, conv_ts_col
        )
        
        # Specific validation tests
        validation_tests = [
            self._validate_no_data_loss,
            self._validate_timezone_awareness,
            self._validate_temporal_order,
            self._validate_reasonable_dates,
            self._validate_duplicate_timestamps
        ]
        
        for test_func in validation_tests:
            test_result = test_func(original_df, converted_df, orig_ts_col, conv_ts_col)
            
            if not test_result['passed']:
                validation_report['validation_passed'] = False
                validation_report['issues_found'].extend(test_result['issues'])
            
            validation_report['statistics'].update(test_result.get('statistics', {}))
        
        # Generate recommendations
        validation_report['recommendations'] = self._generate_recommendations(validation_report)
        
        logger.info(f"📊 Validation completed: {'✅ PASSED' if validation_report['validation_passed'] else '❌ FAILED'}")
        
        return validation_report
    
    def analyze_timestamp_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in timestamp data
        """
        
        logger.info("📈 Analyzing timestamp patterns")
        
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            return {'error': 'No timestamp column found'}
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        patterns = {
            'temporal_distribution': self._analyze_temporal_distribution(timestamps),
            'frequency_analysis': self._analyze_frequency_patterns(timestamps),
            'gap_analysis': self._analyze_time_gaps(timestamps),
            'seasonality': self._analyze_seasonality_patterns(timestamps),
            'outlier_detection': self._detect_timestamp_outliers(timestamps)
        }
        
        return patterns
    
    def create_validation_report(self, validation_results: Dict) -> str:
        """
        Create a formatted validation report
        """
        
        report = []
        report.append("📊 TIMESTAMP VALIDATION REPORT")
        report.append("=" * 50)
        
        # Summary
        status = "✅ PASSED" if validation_results['validation_passed'] else "❌ FAILED"
        report.append(f"\n🎯 Overall Status: {status}")
        
        # Conversion Summary
        if 'conversion_summary' in validation_results:
            summary = validation_results['conversion_summary']
            report.append(f"\n📋 Conversion Summary:")
            report.append(f"   • Original records: {summary.get('original_count', 'N/A')}")
            report.append(f"   • Converted records: {summary.get('converted_count', 'N/A')}")
            report.append(f"   • Data loss: {summary.get('data_loss', 'N/A')}")
            report.append(f"   • Timezone conversion: {summary.get('timezone_conversion', 'N/A')}")
        
        # Issues Found
        if validation_results['issues_found']:
            report.append(f"\n❌ Issues Found ({len(validation_results['issues_found'])}):")
            for issue in validation_results['issues_found']:
                report.append(f"   • {issue}")
        
        # Data Quality
        if 'data_quality' in validation_results:
            quality = validation_results['data_quality']
            report.append(f"\n📊 Data Quality Metrics:")
            for metric, value in quality.items():
                report.append(f"   • {metric}: {value}")
        
        # Recommendations
        if validation_results['recommendations']:
            report.append(f"\n💡 Recommendations:")
            for rec in validation_results['recommendations']:
                report.append(f"   • {rec}")
        
        return "\n".join(report)
    
    def visualize_timestamp_quality(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create visualizations for timestamp quality assessment
        """
        
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            logger.error("No timestamp column found for visualization")
            return
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Timestamp Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. Hourly distribution
        hourly_dist = timestamps.dt.hour.value_counts().sort_index()
        axes[0, 0].bar(hourly_dist.index, hourly_dist.values)
        axes[0, 0].set_title('Data Distribution by Hour')
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Records')
        
        # 2. Daily distribution
        daily_dist = timestamps.dt.date.value_counts().sort_index()
        axes[0, 1].plot(daily_dist.index, daily_dist.values, marker='o')
        axes[0, 1].set_title('Data Distribution by Day')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Number of Records')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Time gaps analysis
        time_diffs = timestamps.diff().dt.total_seconds() / 60  # Minutes
        time_diffs = time_diffs.dropna()
        
        axes[1, 0].hist(time_diffs[time_diffs <= 120], bins=30, alpha=0.7)  # Up to 2 hours
        axes[1, 0].set_title('Time Gaps Between Records (≤2 hours)')
        axes[1, 0].set_xlabel('Gap Duration (minutes)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Weekly pattern
        weekly_pattern = timestamps.dt.day_of_week.value_counts().sort_index()
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 1].bar(range(7), weekly_pattern.values)
        axes[1, 1].set_title('Data Distribution by Day of Week')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Number of Records')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(days)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Visualization saved to {save_path}")
        
        plt.show()
    
    # Helper methods
    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in DataFrame"""
        timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'created_at', 'recorded_at']
        
        for col in df.columns:
            if col.lower() in timestamp_candidates:
                return col
        
        for col in df.columns:
            if any(candidate in col.lower() for candidate in timestamp_candidates):
                return col
        
        return None
    
    def _analyze_conversion_summary(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                                  orig_col: str, conv_col: str) -> Dict:
        """Analyze conversion summary statistics"""
        
        orig_count = len(original_df)
        conv_count = len(converted_df)
        
        # Check for timezone awareness
        conv_timestamps = pd.to_datetime(converted_df[conv_col], errors='coerce')
        is_tz_aware = conv_timestamps.dt.tz is not None if len(conv_timestamps) > 0 else False
        
        return {
            'original_count': orig_count,
            'converted_count': conv_count,
            'data_loss': orig_count - conv_count,
            'timezone_conversion': 'UTC' if is_tz_aware else 'Naive',
            'timezone_aware': is_tz_aware
        }
    
    def _analyze_data_quality(self, df: pd.DataFrame, timestamp_col: str) -> Dict:
        """Analyze data quality metrics"""
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        quality_metrics = {
            'total_records': len(df),
            'valid_timestamps': len(timestamps.dropna()),
            'invalid_timestamps': timestamps.isna().sum(),
            'duplicate_timestamps': timestamps.duplicated().sum(),
            'future_timestamps': (timestamps > datetime.now()).sum() if len(timestamps) > 0 else 0,
            'time_range_days': (timestamps.max() - timestamps.min()).days if len(timestamps) > 0 else 0
        }
        
        return quality_metrics
    
    def _validate_no_data_loss(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                              orig_col: str, conv_col: str) -> Dict:
        """Validate no data was lost during conversion"""
        
        orig_count = len(original_df)
        conv_count = len(converted_df)
        
        passed = orig_count == conv_count
        issues = [] if passed else [f'Data loss detected: {orig_count - conv_count} records']
        
        return {
            'test_name': 'Data Loss Check',
            'passed': passed,
            'issues': issues,
            'statistics': {'original_count': orig_count, 'converted_count': conv_count}
        }
    
    def _validate_timezone_awareness(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                                   orig_col: str, conv_col: str) -> Dict:
        """Validate timezone awareness after conversion"""
        
        conv_timestamps = pd.to_datetime(converted_df[conv_col], errors='coerce')
        is_tz_aware = conv_timestamps.dt.tz is not None if len(conv_timestamps) > 0 else False
        
        passed = is_tz_aware
        issues = [] if passed else ['Converted timestamps are not timezone-aware']
        
        return {
            'test_name': 'Timezone Awareness Check',
            'passed': passed,
            'issues': issues,
            'statistics': {'timezone_aware': is_tz_aware}
        }
    
    def _validate_temporal_order(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                                orig_col: str, conv_col: str) -> Dict:
        """Validate temporal order is preserved"""
        
        conv_timestamps = pd.to_datetime(converted_df[conv_col], errors='coerce').dropna()
        
        # Check if timestamps are in chronological order
        is_ordered = conv_timestamps.is_monotonic_increasing
        
        passed = is_ordered or len(conv_timestamps) <= 1
        issues = [] if passed else ['Temporal order not preserved after conversion']
        
        return {
            'test_name': 'Temporal Order Check',
            'passed': passed,
            'issues': issues,
            'statistics': {'chronologically_ordered': is_ordered}
        }
    
    def _validate_reasonable_dates(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                                 orig_col: str, conv_col: str) -> Dict:
        """Validate dates are within reasonable range"""
        
        conv_timestamps = pd.to_datetime(converted_df[conv_col], errors='coerce').dropna()
        
        # Check for unreasonable dates
        min_reasonable = datetime(2000, 1, 1)
        max_reasonable = datetime(2030, 12, 31)
        
        unreasonable_count = ((conv_timestamps < min_reasonable) | 
                             (conv_timestamps > max_reasonable)).sum()
        
        passed = unreasonable_count == 0
        issues = [] if passed else [f'{unreasonable_count} timestamps outside reasonable range (2000-2030)']
        
        return {
            'test_name': 'Reasonable Date Range Check',
            'passed': passed,
            'issues': issues,
            'statistics': {'unreasonable_dates': unreasonable_count}
        }
    
    def _validate_duplicate_timestamps(self, original_df: pd.DataFrame, converted_df: pd.DataFrame,
                                     orig_col: str, conv_col: str) -> Dict:
        """Validate duplicate timestamp handling"""
        
        conv_timestamps = pd.to_datetime(converted_df[conv_col], errors='coerce')
        duplicate_count = conv_timestamps.duplicated().sum()
        
        # Duplicates might be acceptable depending on use case
        passed = duplicate_count < len(conv_timestamps) * 0.05  # Less than 5%
        issues = [] if passed else [f'{duplicate_count} duplicate timestamps found']
        
        return {
            'test_name': 'Duplicate Timestamp Check',
            'passed': passed,
            'issues': issues,
            'statistics': {'duplicate_count': duplicate_count}
        }
    
    def _analyze_temporal_distribution(self, timestamps: pd.Series) -> Dict:
        """Analyze temporal distribution patterns"""
        
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) == 0:
            return {'error': 'No valid timestamps'}
        
        return {
            'hourly_distribution': valid_timestamps.dt.hour.value_counts().to_dict(),
            'daily_distribution': valid_timestamps.dt.day_of_week.value_counts().to_dict(),
            'monthly_distribution': valid_timestamps.dt.month.value_counts().to_dict()
        }
    
    def _analyze_frequency_patterns(self, timestamps: pd.Series) -> Dict:
        """Analyze data recording frequency patterns"""
        
        valid_timestamps = timestamps.dropna().sort_values()
        
        if len(valid_timestamps) < 2:
            return {'error': 'Insufficient data for frequency analysis'}
        
        # Calculate time differences
        time_diffs = valid_timestamps.diff().dt.total_seconds()
        
        return {
            'median_interval_seconds': time_diffs.median(),
            'mean_interval_seconds': time_diffs.mean(),
            'min_interval_seconds': time_diffs.min(),
            'max_interval_seconds': time_diffs.max(),
            'regular_intervals': time_diffs.std() < time_diffs.mean() * 0.1  # Low variation = regular
        }
    
    def _analyze_time_gaps(self, timestamps: pd.Series) -> Dict:
        """Analyze gaps in timestamp data"""
        
        valid_timestamps = timestamps.dropna().sort_values()
        
        if len(valid_timestamps) < 2:
            return {'error': 'Insufficient data for gap analysis'}
        
        time_diffs = valid_timestamps.diff().dt.total_seconds() / 3600  # Hours
        
        # Define gap thresholds
        large_gaps = time_diffs[time_diffs > 6].count()  # Gaps > 6 hours
        very_large_gaps = time_diffs[time_diffs > 24].count()  # Gaps > 24 hours
        
        return {
            'total_gaps': len(time_diffs) - 1,
            'large_gaps_6h': large_gaps,
            'very_large_gaps_24h': very_large_gaps,
            'max_gap_hours': time_diffs.max(),
            'mean_gap_hours': time_diffs.mean()
        }
    
    def _analyze_seasonality_patterns(self, timestamps: pd.Series) -> Dict:
        """Analyze seasonal patterns in timestamps"""
        
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) == 0:
            return {'error': 'No valid timestamps'}
        
        # Simple seasonality analysis
        return {
            'weekend_vs_weekday': {
                'weekday_count': valid_timestamps[valid_timestamps.dt.day_of_week < 5].count(),
                'weekend_count': valid_timestamps[valid_timestamps.dt.day_of_week >= 5].count()
            },
            'business_hours_vs_off_hours': {
                'business_hours': valid_timestamps[(valid_timestamps.dt.hour >= 9) & 
                                                 (valid_timestamps.dt.hour <= 17)].count(),
                'off_hours': valid_timestamps[(valid_timestamps.dt.hour < 9) | 
                                            (valid_timestamps.dt.hour > 17)].count()
            }
        }
    
    def _detect_timestamp_outliers(self, timestamps: pd.Series) -> Dict:
        """Detect outlier timestamps"""
        
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) < 10:
            return {'error': 'Insufficient data for outlier detection'}
        
        # Statistical outlier detection based on time differences
        time_diffs = valid_timestamps.sort_values().diff().dt.total_seconds()
        
        q1 = time_diffs.quantile(0.25)
        q3 = time_diffs.quantile(0.75)
        iqr = q3 - q1
        
        outlier_threshold = q3 + 1.5 * iqr
        outliers = time_diffs[time_diffs > outlier_threshold]
        
        return {
            'outlier_count': len(outliers),
            'outlier_threshold_seconds': outlier_threshold,
            'outlier_positions': outliers.index.tolist()
        }
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        if not validation_results['validation_passed']:
            recommendations.append("Address validation issues before proceeding with analysis")
        
        issues = validation_results.get('issues_found', [])
        
        for issue in issues:
            if 'Data loss' in issue:
                recommendations.append("Review data conversion process to prevent data loss")
            elif 'timezone-aware' in issue:
                recommendations.append("Ensure timestamps are properly timezone-aware after conversion")
            elif 'duplicate' in issue:
                recommendations.append("Consider deduplication strategy for timestamp data")
            elif 'unreasonable' in issue:
                recommendations.append("Review and clean timestamp data for invalid dates")
        
        # Data quality recommendations
        quality = validation_results.get('data_quality', {})
        
        if quality.get('invalid_timestamps', 0) > 0:
            recommendations.append("Clean invalid timestamp data before analysis")
        
        if quality.get('duplicate_timestamps', 0) > quality.get('total_records', 1) * 0.1:
            recommendations.append("High duplicate rate - consider timestamp resolution improvement")
        
        return recommendations

def test_timestamp_validator():
    """Test the timestamp validator"""
    
    print("🔍 Testing Timestamp Validator")
    print("=" * 40)
    
    # Create test data
    dates = pd.date_range('2024-03-10 06:00:00', '2024-03-12 18:00:00', freq='H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'heart_rate': np.random.normal(75, 10, len(dates)),
        'steps': np.random.poisson(100, len(dates))
    })
    
    # Create "converted" data (simulate timezone conversion)
    converted_data = test_data.copy()
    converted_data['timestamp'] = pd.to_datetime(converted_data['timestamp']).dt.tz_localize('UTC')
    
    # Test validator
    validator = TimestampValidator()
    validation_result = validator.validate_timestamp_conversion(test_data, converted_data)
    
    # Print report
    report = validator.create_validation_report(validation_result)
    print(report)
    
    # Test pattern analysis
    patterns = validator.analyze_timestamp_patterns(converted_data)
    print(f"\n📈 Pattern Analysis Keys: {list(patterns.keys())}")

if __name__ == "__main__":
    test_timestamp_validator()
