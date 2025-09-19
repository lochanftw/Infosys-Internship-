"""
Automatic Timezone Detection for Task 3
Smart detection based on data patterns and user behavior
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TimezoneDetector:
    """Smart timezone detection from fitness data patterns"""
    
    def __init__(self):
        self.common_timezones = [
            'UTC', 'America/New_York', 'America/Los_Angeles', 'Europe/London',
            'Europe/Berlin', 'Asia/Tokyo', 'Asia/Shanghai', 'Australia/Sydney',
            'America/Chicago', 'Asia/Kolkata', 'Europe/Paris'
        ]
        
        self.business_hours = (9, 17)  # 9 AM to 5 PM
        self.sleep_hours = (22, 6)     # 10 PM to 6 AM
    
    def detect_timezone_from_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect timezone based on activity patterns
        
        Returns:
        --------
        Dict with detection results and confidence scores
        """
        
        logger.info("🔍 Analyzing data patterns for timezone detection")
        
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'default'}
        
        # Convert to datetime
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce').dropna()
        
        if len(timestamps) == 0:
            return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'no_data'}
        
        # Multiple detection methods
        methods = {
            'business_hours': self._detect_from_business_hours(df, timestamp_col),
            'sleep_patterns': self._detect_from_sleep_patterns(df, timestamp_col),
            'activity_peaks': self._detect_from_activity_peaks(df, timestamp_col),
            'data_frequency': self._detect_from_frequency_patterns(df, timestamp_col)
        }
        
        # Combine results with weighted scoring
        final_result = self._combine_detection_results(methods)
        
        logger.info(f"🎯 Detected timezone: {final_result['timezone']} "
                   f"(confidence: {final_result['confidence']:.2f})")
        
        return final_result
    
    def detect_travel_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Detect if user was traveling based on timezone shifts
        """
        
        logger.info("✈️ Analyzing travel patterns")
        
        timestamp_col = self._find_timestamp_column(df)
        if not timestamp_col:
            return []
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col)
        timestamps = pd.to_datetime(df_sorted[timestamp_col], errors='coerce')
        
        # Analyze hour patterns over time windows
        travel_periods = []
        window_size = timedelta(days=2)
        
        for i in range(0, len(timestamps), 48):  # Check every 2 days
            window_start = timestamps.iloc[i] if i < len(timestamps) else timestamps.iloc[-1]
            window_end = window_start + window_size
            
            window_mask = (timestamps >= window_start) & (timestamps <= window_end)
            window_data = df_sorted[window_mask]
            
            if len(window_data) > 10:  # Enough data points
                tz_result = self.detect_timezone_from_patterns(window_data)
                
                travel_periods.append({
                    'period_start': window_start,
                    'period_end': window_end,
                    'detected_timezone': tz_result['timezone'],
                    'confidence': tz_result['confidence'],
                    'data_points': len(window_data)
                })
        
        # Identify timezone changes (potential travel)
        travel_events = []
        prev_tz = None
        
        for period in travel_periods:
            current_tz = period['detected_timezone']
            
            if prev_tz and prev_tz != current_tz and period['confidence'] > 0.6:
                travel_events.append({
                    'travel_start': period['period_start'],
                    'from_timezone': prev_tz,
                    'to_timezone': current_tz,
                    'confidence': period['confidence']
                })
            
            prev_tz = current_tz
        
        logger.info(f"🗺️ Detected {len(travel_events)} potential travel events")
        return travel_events
    
    def _detect_from_business_hours(self, df: pd.DataFrame, timestamp_col: str) -> Dict:
        """Detect timezone based on business hours activity"""
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        hours = timestamps.dt.hour
        
        # Count activity during typical business hours for each timezone
        scores = {}
        
        for tz_name in self.common_timezones:
            try:
                tz = pytz.timezone(tz_name)
                
                # Convert to this timezone and check business hours
                tz_times = timestamps.dt.tz_localize('UTC').dt.tz_convert(tz)
                tz_hours = tz_times.dt.hour
                
                business_activity = len(tz_hours[(tz_hours >= 9) & (tz_hours <= 17)])
                total_activity = len(tz_hours.dropna())
                
                if total_activity > 0:
                    score = business_activity / total_activity
                    scores[tz_name] = score
                
            except Exception:
                continue
        
        if scores:
            best_tz = max(scores, key=scores.get)
            confidence = scores[best_tz]
            
            return {
                'timezone': best_tz,
                'confidence': confidence,
                'method': 'business_hours',
                'scores': scores
            }
        
        return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'business_hours'}
    
    def _detect_from_sleep_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict:
        """Detect timezone based on sleep/inactive periods"""
        
        # Look for heart rate or step patterns that indicate sleep
        sleep_indicators = ['heart_rate', 'steps', 'activity_level']
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Find low activity periods (likely sleep)
        low_activity_mask = pd.Series([False] * len(df))
        
        if 'heart_rate' in df.columns:
            hr_mean = df['heart_rate'].mean()
            low_activity_mask |= df['heart_rate'] < (hr_mean * 0.8)
        
        if 'steps' in df.columns:
            low_activity_mask |= df['steps'] < 10
        
        if 'activity_level' in df.columns:
            low_activity_mask |= df['activity_level'].isin(['sleep', 'sedentary', 'rest'])
        
        sleep_times = timestamps[low_activity_mask]
        
        if len(sleep_times) == 0:
            return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'sleep_patterns'}
        
        # Check which timezone puts sleep times in typical sleep hours (10 PM - 6 AM)
        scores = {}
        
        for tz_name in self.common_timezones:
            try:
                tz = pytz.timezone(tz_name)
                tz_sleep_times = sleep_times.dt.tz_localize('UTC').dt.tz_convert(tz)
                tz_sleep_hours = tz_sleep_times.dt.hour
                
                # Count sleep during typical sleep hours
                night_sleep = len(tz_sleep_hours[tz_sleep_hours >= 22]) + \
                             len(tz_sleep_hours[tz_sleep_hours <= 6])
                
                total_sleep = len(tz_sleep_hours)
                
                if total_sleep > 0:
                    scores[tz_name] = night_sleep / total_sleep
                
            except Exception:
                continue
        
        if scores:
            best_tz = max(scores, key=scores.get)
            confidence = scores[best_tz]
            
            return {
                'timezone': best_tz,
                'confidence': confidence,
                'method': 'sleep_patterns',
                'scores': scores
            }
        
        return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'sleep_patterns'}
    
    def _detect_from_activity_peaks(self, df: pd.DataFrame, timestamp_col: str) -> Dict:
        """Detect timezone based on daily activity peaks"""
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Use steps or heart rate as activity indicator
        activity_col = None
        if 'steps' in df.columns:
            activity_col = 'steps'
        elif 'heart_rate' in df.columns:
            activity_col = 'heart_rate'
        
        if not activity_col:
            return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'activity_peaks'}
        
        # Group by hour and find peak activity times
        df_temp = df.copy()
        df_temp['hour'] = timestamps.dt.hour
        hourly_activity = df_temp.groupby('hour')[activity_col].mean()
        
        # Find peak activity hours
        peak_hours = hourly_activity.nlargest(6).index.tolist()  # Top 6 hours
        
        # Score timezones based on whether peaks align with typical active hours
        typical_active_hours = list(range(8, 20))  # 8 AM to 8 PM
        
        scores = {}
        for tz_name in self.common_timezones:
            # Convert peak hours to this timezone
            overlap = len(set(peak_hours) & set(typical_active_hours))
            scores[tz_name] = overlap / len(peak_hours) if peak_hours else 0
        
        if scores:
            best_tz = max(scores, key=scores.get)
            confidence = scores[best_tz]
            
            return {
                'timezone': best_tz,
                'confidence': confidence,
                'method': 'activity_peaks',
                'scores': scores
            }
        
        return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'activity_peaks'}
    
    def _detect_from_frequency_patterns(self, df: pd.DataFrame, timestamp_col: str) -> Dict:
        """Detect timezone based on data recording frequency patterns"""
        
        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        
        # Analyze gaps in data (might indicate sleep or timezone issues)
        time_diffs = timestamps.diff().dt.total_seconds() / 3600  # Hours
        
        # Find typical recording intervals
        common_intervals = time_diffs.value_counts().head(3)
        
        # This is a simplified heuristic
        # In practice, you'd look for patterns like:
        # - Regular gaps during night hours
        # - Consistent recording intervals during day
        
        return {'timezone': 'UTC', 'confidence': 0.3, 'method': 'frequency_patterns'}
    
    def _combine_detection_results(self, methods: Dict) -> Dict:
        """Combine results from multiple detection methods"""
        
        # Weighted scoring based on method reliability
        weights = {
            'business_hours': 0.4,
            'sleep_patterns': 0.3,
            'activity_peaks': 0.2,
            'data_frequency': 0.1
        }
        
        timezone_scores = {}
        
        for method_name, result in methods.items():
            if 'scores' in result:
                weight = weights.get(method_name, 0.1)
                
                for tz, score in result['scores'].items():
                    if tz not in timezone_scores:
                        timezone_scores[tz] = 0
                    
                    timezone_scores[tz] += score * weight
        
        if timezone_scores:
            best_timezone = max(timezone_scores, key=timezone_scores.get)
            final_confidence = timezone_scores[best_timezone]
            
            return {
                'timezone': best_timezone,
                'confidence': min(final_confidence, 1.0),  # Cap at 1.0
                'method': 'combined',
                'individual_methods': methods,
                'all_scores': timezone_scores
            }
        
        # Fallback
        return {'timezone': 'UTC', 'confidence': 0.0, 'method': 'fallback'}
    
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

def test_timezone_detector():
    """Test the timezone detector"""
    
    print("🔍 Testing Timezone Detector")
    print("=" * 40)
    
    # Create test data with business hours pattern (NYC timezone)
    ny_tz = pytz.timezone('America/New_York')
    base_date = datetime(2024, 3, 10, 9, 0)  # 9 AM NYC time
    
    test_dates = []
    for i in range(72):  # 3 days of hourly data
        test_dates.append(ny_tz.localize(base_date + timedelta(hours=i)))
    
    test_data = pd.DataFrame({
        'timestamp': test_dates,
        'heart_rate': np.concatenate([
            np.random.normal(75, 10, 48),  # Day 1-2: normal activity
            np.random.normal(65, 8, 24)   # Night: lower heart rate
        ]),
        'steps': np.concatenate([
            np.random.poisson(120, 48),   # Day activity
            np.random.poisson(5, 24)     # Night: minimal steps
        ])
    })
    
    # Test detector
    detector = TimezoneDetector()
    result = detector.detect_timezone_from_patterns(test_data)
    
    print(f"🎯 Detection Result:")
    print(f"   Timezone: {result['timezone']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Method: {result['method']}")
    
    # Test travel detection
    travel_events = detector.detect_travel_patterns(test_data)
    print(f"\n✈️ Travel Events: {len(travel_events)}")

if __name__ == "__main__":
    test_timezone_detector()
