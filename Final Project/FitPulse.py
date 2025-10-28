import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import os
import glob
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

warnings.filterwarnings('ignore')

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / 'milestone2'))

# Import milestone modules
try:
    from milestone2.feature_engineering import AdvancedFeatureEngine
    from milestone2.time_series_analysis import TrendForecastingEngine
    from milestone2.pattern_clustering import BehaviorPatternAnalyzer
    from milestone2.data_generator import generate_comprehensive_dataset
    from milestone2 import utils
except ImportError as e:
    st.error(f"Missing milestone2 modules: {e}")

# Import visualization functions
try:
    from fitpulse_visualizations import (
        M2Visualizations,
        M4Visualizations,
        apply_purple_theme
    )
    VISUALS_AVAILABLE = True
except ImportError:
    st.warning("fitpulse_visualizations.py not found - visualizations disabled")
    VISUALS_AVAILABLE = False

# Import PDF exporter
try:
    from fitpulse_pdf_exporter import EnhancedPDFExporter
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ============================================================================
# UNIFIED SESSION STATE MANAGEMENT
# ============================================================================

def initialize_unified_session():
    """Initialize session state for all 4 milestones"""
    
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {}
    
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = AdvancedFeatureEngine()
    if 'forecast_engine' not in st.session_state:
        st.session_state.forecast_engine = TrendForecastingEngine()
    if 'pattern_analyzer' not in st.session_state:
        st.session_state.pattern_analyzer = BehaviorPatternAnalyzer()
    if 'm2_results' not in st.session_state:
        st.session_state.m2_results = {}
    
    if 'anomaly_results' not in st.session_state:
        st.session_state.anomaly_results = {}
    if 'anomalies_detected' not in st.session_state:
        st.session_state.anomalies_detected = {}
    
    if 'dashboard_ready' not in st.session_state:
        st.session_state.dashboard_ready = False
    
    if 'config' not in st.session_state:
        st.session_state.config = {
            'feature_complexity': 'Standard',
            'window_size': 60,
            'window_overlap': 50,
            'forecast_periods': 120,
            'confidence_interval': 0.95,
            'detect_anomalies': True,
            'cluster_method': 'kmeans',
            'n_clusters': 3,
            'scaling_method': 'standard',
            'auto_tune': True
        }

# ============================================================================
# DATA LOADER
# ============================================================================

class UnifiedDataLoader:
    """Unified data loader for all milestones"""
    
    def __init__(self, data_folder='data/input'):
        self.data_folder = data_folder
    
    def load_all_data(self):
        """Load all CSV files"""
        loaded_data = {}
        
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder, exist_ok=True)
            return loaded_data
        
        csv_files = glob.glob(os.path.join(self.data_folder, '*.csv'))
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            try:
                df = pd.read_csv(filepath)
                
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                data_type = self._detect_data_type(filename, df)
                if data_type:
                    loaded_data[data_type] = df
                    
            except Exception as e:
                st.error(f"Error loading {filename}: {e}")
        
        return loaded_data
    
    def _detect_data_type(self, filename, df):
        """Auto-detect data type"""
        filename_lower = filename.lower()
        
        if 'heart' in filename_lower or 'hr' in filename_lower:
            return 'heart_rate'
        elif 'step' in filename_lower:
            return 'steps'
        elif 'activity' in filename_lower:
            return 'activity'
        elif 'sleep' in filename_lower:
            return 'sleep'
        elif 'pros' in filename_lower:
            return 'pros'
        else:
            return filename.replace('.csv', '')
        

# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class EnhancedAnomalyDetector:
    """Enhanced anomaly detection using M2 results"""
    
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {'min': 40, 'max': 120, 'sustained_minutes': 10},
            'steps': {'min': 0, 'max': 1000, 'sustained_minutes': 5},
            'activity': {'min': 0, 'max': 100, 'sustained_minutes': 5},
            'sleep': {'min': 180, 'max': 720, 'sustained_minutes': 0},
            'pros': {'min': 40, 'max': 120, 'sustained_minutes': 10}
        }
    
    def detect_enhanced_anomalies(self, data, m2_features=None, m2_forecasts=None):
        """Enhanced detection"""
        results = {}
        
        for data_type, df in data.items():
            st.info(f"🔍 Detecting anomalies: {data_type}...")
            
            threshold_anomalies = self._detect_threshold_anomalies(df, data_type)
            statistical_anomalies = self._detect_statistical_anomalies(df, data_type)
            
            feature_anomalies = []
            if m2_features and data_type in m2_features:
                feature_anomalies = self._detect_feature_anomalies(
                    df, m2_features[data_type], data_type
                )
            
            forecast_anomalies = []
            if m2_forecasts and data_type in m2_forecasts:
                forecast_anomalies = self._detect_forecast_anomalies(
                    df, m2_forecasts[data_type], data_type
                )
            
            all_anomalies = (
                threshold_anomalies + statistical_anomalies + 
                feature_anomalies + forecast_anomalies
            )
            
            results[data_type] = {
                'anomalies': all_anomalies,
                'counts': {
                    'threshold': len(threshold_anomalies),
                    'statistical': len(statistical_anomalies),
                    'feature_based': len(feature_anomalies),
                    'forecast_based': len(forecast_anomalies),
                    'total': len(all_anomalies)
                },
                'data_with_anomalies': self._merge_anomalies_with_data(df, all_anomalies)
            }
        
        return results
    
    def _detect_threshold_anomalies(self, df, data_type):
        anomalies = []
        if data_type not in self.threshold_rules:
            return anomalies
        
        rule = self.threshold_rules[data_type]
        metric_col = self._get_metric_column(data_type)
        
        if metric_col not in df.columns:
            return anomalies
        
        high = df[df[metric_col] > rule['max']]
        low = df[df[metric_col] < rule['min']]
        
        for _, row in high.iterrows():
            anomalies.append({
                'timestamp': row['timestamp'],
                'value': row[metric_col],
                'type': 'threshold_high',
                'severity': 'High',
                'description': f'{metric_col} above {rule["max"]}'
            })
        
        for _, row in low.iterrows():
            anomalies.append({
                'timestamp': row['timestamp'],
                'value': row[metric_col],
                'type': 'threshold_low',
                'severity': 'Medium',
                'description': f'{metric_col} below {rule["min"]}'
            })
        
        return anomalies
    
    def _detect_statistical_anomalies(self, df, data_type):
        anomalies = []
        metric_col = self._get_metric_column(data_type)
        
        if metric_col not in df.columns:
            return anomalies
        
        window = min(50, len(df) // 10) if len(df) > 50 else 5
        df_copy = df.copy()
        df_copy['rolling_mean'] = df_copy[metric_col].rolling(window=window, center=True).mean()
        df_copy['rolling_std'] = df_copy[metric_col].rolling(window=window, center=True).std()
        
        df_copy['rolling_mean'].fillna(df_copy[metric_col].mean(), inplace=True)
        df_copy['rolling_std'].fillna(df_copy[metric_col].std(), inplace=True)
        
        df_copy['z_score'] = (df_copy[metric_col] - df_copy['rolling_mean']) / (df_copy['rolling_std'] + 1e-6)
        
        stat_anomalies = df_copy[np.abs(df_copy['z_score']) > 3]
        
        for _, row in stat_anomalies.iterrows():
            anomalies.append({
                'timestamp': row['timestamp'],
                'value': row[metric_col],
                'type': 'statistical',
                'severity': 'Medium',
                'description': f'Statistical deviation (Z: {row["z_score"]:.2f})'
            })
        
        return anomalies
    
    def _detect_feature_anomalies(self, df, features, data_type):
        anomalies = []
        
        if features.empty:
            return anomalies
        
        feature_variances = features.var()
        top_features = feature_variances.nlargest(5).index
        
        for feature in top_features:
            feature_data = features[feature]
            z_scores = np.abs(stats.zscore(feature_data))
            anomalous_windows = feature_data[z_scores > 2.5]
            
            for idx in anomalous_windows.index:
                window_start = idx * (st.session_state.config['window_size'] // 2)
                if window_start < len(df):
                    anomalies.append({
                        'timestamp': df.iloc[window_start]['timestamp'],
                        'value': df.iloc[window_start][self._get_metric_column(data_type)],
                        'type': 'feature_based',
                        'severity': 'Low',
                        'description': f'Feature anomaly: {feature[:30]}...'
                    })
        
        return anomalies[:10]
    
    def _detect_forecast_anomalies(self, df, forecast_data, data_type):
        anomalies = []
        
        if 'anomalies' not in forecast_data:
            return anomalies
        
        m2_anomalies = forecast_data['anomalies']
        
        for anomaly in m2_anomalies[:20]:
            anomalies.append({
                'timestamp': anomaly['timestamp'],
                'value': anomaly['actual_value'],
                'type': 'forecast_based',
                'severity': 'High' if anomaly['anomaly_score'] > 3 else 'Medium',
                'description': f'Forecast deviation: {anomaly["anomaly_score"]:.2f}'
            })
        
        return anomalies
    
    def _get_metric_column(self, data_type):
        metric_cols = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'activity': 'activity_level',
            'sleep': 'duration_minutes',
            'pros': 'heart_rate'
        }
        return metric_cols.get(data_type, 'value')
    
    def _merge_anomalies_with_data(self, df, anomalies):
        df_result = df.copy()
        df_result['is_anomaly'] = False
        df_result['anomaly_type'] = ''
        df_result['anomaly_severity'] = 'Normal'
        df_result['anomaly_description'] = ''
        
        for anomaly in anomalies:
            mask = df_result['timestamp'] == anomaly['timestamp']
            df_result.loc[mask, 'is_anomaly'] = True
            df_result.loc[mask, 'anomaly_type'] = anomaly['type']
            df_result.loc[mask, 'anomaly_severity'] = anomaly['severity']
            df_result.loc[mask, 'anomaly_description'] = anomaly['description']
        
        return df_result

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main unified application"""
    
    st.set_page_config(
        page_title="🏥 FitPulse Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    initialize_unified_session()
    
    # Apply theme
    if VISUALS_AVAILABLE:
        apply_purple_theme()
    elif 'milestone2' in sys.modules:
        utils.apply_custom_styling()
    
    # Header
    st.markdown("""
        <div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:10px;color:white;margin-bottom:2rem">
            <h1>🏥 FitPulse Unified Analytics Platform</h1>
            <p style="font-size:1.2rem;margin-top:0.5rem">
                Complete Pipeline: Data Processing → Analytics → Anomaly Detection → Insights
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        render_pipeline_status()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 M1: Data Processing",
        "🧬 M2: Advanced Analytics", 
        "🚨 M3: Anomaly Detection",
        "📋 M4: Dashboard"
    ])
    
    with tab1:
        handle_milestone1()
    
    with tab2:
        handle_milestone2()
    
    with tab3:
        handle_milestone3()
    
    with tab4:
        handle_milestone4()

def render_pipeline_status():
    """Render pipeline progress"""
    
    st.header("🔄 Pipeline Status")
    
    if st.session_state.processed_data:
        st.success("✅ M1: Data Loaded")
        st.write(f"📁 {len(st.session_state.processed_data)} datasets")
    else:
        st.info("⬜ M1: Load Data")
    
    if st.session_state.m2_results:
        completed = []
        if 'features' in st.session_state.m2_results:
            completed.append("Features")
        if 'forecasts' in st.session_state.m2_results:
            completed.append("Forecasts")
        if 'clusters' in st.session_state.m2_results:
            completed.append("Clusters")
        
        if completed:
            st.success("✅ M2: Analytics")
            st.write(f"📈 {', '.join(completed)}")
        else:
            st.info("⬜ M2: Run Analytics")
    else:
        st.info("⬜ M2: Run Analytics")
    
    if st.session_state.anomaly_results:
        total = sum([r['counts']['total'] for r in st.session_state.anomaly_results.values()])
        st.success("✅ M3: Anomalies")
        st.write(f"🚨 {total} detected")
    else:
        st.info("⬜ M3: Detect Anomalies")
    
    if st.session_state.dashboard_ready:
        st.success("✅ M4: Dashboard Ready")
    else:
        st.info("⬜ M4: Generate Insights")
    
    st.divider()
    st.markdown("### ⚙️ Config")
    cfg = st.session_state.config
    st.write(f"🪟 Window: {cfg['window_size']}min")
    st.write(f"📊 Forecast: {cfg['forecast_periods']}")
    st.write(f"🎯 Clusters: {cfg['n_clusters']}")

def handle_milestone1():
    """Milestone 1: COMPLETE Data Processing & Cleaning"""
    
    st.header("📊 Milestone 1: Data Processing & Quality Control")
    st.info("📋 Complete data pipeline: Upload → Validate → Clean → Align → Visualize")
    
    loader = UnifiedDataLoader()
    
    # Create tabs for different preprocessing stages
    upload_tab, clean_tab, align_tab, viz_tab, summary_tab = st.tabs([
        "📤 Upload Data",
        "🧹 Data Cleaning",
        "⏱️ Time Alignment",
        "📊 Visualizations",
        "📋 Summary Report"
    ])
    
    # ==================== TAB 1: UPLOAD ====================
    with upload_tab:
        st.subheader("📤 Data Upload & Loading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔄 Load Existing Data")
            
            if st.button("📂 Load from data/input/", type="primary", use_container_width=True):
                with st.spinner("Loading data files..."):
                    data = loader.load_all_data()
                    
                    if data:
                        st.session_state.raw_data = data
                        st.session_state.processed_data = data.copy()
                        st.success(f"✅ Loaded {len(data)} datasets")
                        
                        # Display data overview
                        for data_type, df in data.items():
                            with st.expander(f"📋 {data_type.replace('_', ' ').title()} ({len(df)} records)", expanded=False):
                                st.dataframe(df.head(10), use_container_width=True)
                                
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    if len(df.columns) > 1:
                                        val_col = df.columns[1]
                                        st.metric("Mean", f"{df[val_col].mean():.2f}")
                                with col_b:
                                    if len(df.columns) > 1:
                                        st.metric("Std Dev", f"{df[val_col].std():.2f}")
                                with col_c:
                                    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                                    st.metric("Missing %", f"{missing_pct:.1f}%")
                    else:
                        st.warning("⚠️ No CSV files found in data/input/")
        
        with col2:
            st.markdown("### 🎲 Generate Sample Data")
            
            days = st.slider("Days of data", 1, 7, 3)
            profile = st.selectbox("User profile", ["normal", "active", "athlete"])
            
            if st.button("🚀 Generate", use_container_width=True):
                with st.spinner("Generating sample data..."):
                    try:
                        data = generate_comprehensive_dataset(
                            days=days, 
                            user_profile=profile,
                            noise_level=0.1,
                            anomaly_rate=0.02
                        )
                        st.session_state.raw_data = data
                        st.session_state.processed_data = data.copy()
                        st.success(f"✅ Generated {days}-day dataset!")
                        
                    except Exception as e:
                        st.error(f"❌ Generation failed: {e}")
    
    # ==================== TAB 2: DATA CLEANING ====================
    with clean_tab:
        st.subheader("🧹 Data Cleaning & Quality Control")
        
        if not st.session_state.processed_data:
            st.warning("⚠️ Please load or generate data first (Upload Data tab)")
        else:
            st.info("🔍 Apply data validation, outlier detection, and missing value handling")
            
            # Cleaning options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                remove_duplicates = st.checkbox("Remove Duplicates", value=True)
            with col2:
                handle_missing = st.selectbox("Missing Values", 
                    ["Interpolate", "Forward Fill", "Backward Fill", "Drop"]),
                    
            with col3:
                detect_outliers = st.checkbox("Detect Outliers", value=True)
            
            if st.button("🚀 Clean Data", type="primary", use_container_width=True):
                with st.spinner("Cleaning data..."):
                    cleaned_data = {}
                    cleaning_reports = {}
                    
                    for data_type, df in st.session_state.processed_data.items():
                        st.info(f"Cleaning {data_type}...")
                        
                        df_clean = df.copy()
                        report = {
                            'original_rows': len(df),
                            'issues_found': [],
                            'rows_removed': 0,
                            'missing_handled': 0,
                            'outliers_detected': 0
                        }
                        
                        # 1. Remove duplicates
                        if remove_duplicates:
                            initial_len = len(df_clean)
                            if 'timestamp' in df_clean.columns:
                                df_clean = df_clean.drop_duplicates(subset=['timestamp'])
                            else:
                                df_clean = df_clean.drop_duplicates()
                            duplicates_removed = initial_len - len(df_clean)
                            if duplicates_removed > 0:
                                report['issues_found'].append(f"Removed {duplicates_removed} duplicate rows")
                                report['rows_removed'] += duplicates_removed
                        
                        # 2. Handle missing values
                        missing_before = df_clean.isnull().sum().sum()
                        if missing_before > 0:
                            if handle_missing == "Interpolate":
                                df_clean = df_clean.interpolate(method='linear', limit_direction='both')
                            elif handle_missing == "Forward Fill":
                                df_clean = df_clean.fillna(method='ffill')
                            elif handle_missing == "Backward Fill":
                                df_clean = df_clean.fillna(method='bfill')
                            elif handle_missing == "Drop":
                                df_clean = df_clean.dropna()
                            
                            missing_after = df_clean.isnull().sum().sum()
                            report['missing_handled'] = missing_before - missing_after
                            if report['missing_handled'] > 0:
                                report['issues_found'].append(f"Handled {report['missing_handled']} missing values")
                        
                        # 3. Detect outliers (Z-score method)
                        if detect_outliers:
                            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                            for col in numeric_cols:
                                if col != 'timestamp' and len(df_clean[col]) > 0:
                                    z_scores = np.abs(stats.zscore(df_clean[col].fillna(df_clean[col].mean())))
                                    outlier_mask = z_scores > 3
                                    outlier_count = outlier_mask.sum()
                                    
                                    if outlier_count > 0:
                                        df_clean[f'{col}_outlier'] = outlier_mask
                                        report['outliers_detected'] += outlier_count
                                        report['issues_found'].append(f"{col}: {outlier_count} outliers flagged")
                        
                        report['final_rows'] = len(df_clean)
                        report['quality_score'] = round(100 * (1 - report['rows_removed'] / report['original_rows']), 1)
                        
                        cleaned_data[data_type] = df_clean
                        cleaning_reports[data_type] = report
                    
                    # Update session state
                    st.session_state.processed_data = cleaned_data
                    st.session_state.cleaning_reports = cleaning_reports
                    
                    st.success("✅ Data cleaning complete!")
                    
                    # Display cleaning reports
                    st.subheader("📋 Cleaning Reports")
                    
                    for data_type, report in cleaning_reports.items():
                        with st.expander(f"🔍 {data_type.title()} Cleaning Report"):
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Original Rows", f"{report['original_rows']:,}")
                            with col2:
                                st.metric("Final Rows", f"{report['final_rows']:,}")
                            with col3:
                                st.metric("Rows Removed", f"{report['rows_removed']:,}")
                            with col4:
                                st.metric("Quality Score", f"{report['quality_score']}%")
                            
                            if report['issues_found']:
                                st.markdown("**Issues Found:**")
                                for issue in report['issues_found']:
                                    st.write(f"• {issue}")
                            else:
                                st.success("✅ No issues found - data is clean!")
    
    # ==================== TAB 3: TIME ALIGNMENT ====================
    
    with align_tab:
        st.subheader("⏱️ Time Alignment & Resampling")
        
        if not st.session_state.processed_data:
            st.warning("⚠️ Please load data first")
        else:
            st.info("🔄 Resample data to consistent time intervals and fill gaps")
            
            col1, col2 = st.columns(2)
            
            with col1:
                target_freq = st.selectbox("Target Frequency", 
                    ["1min", "5min", "15min", "30min", "1hour"], 
                    index=0,
                    key="m1_align_target_freq")  # ← UNIQUE KEY
            
            with col2:
                fill_method = st.selectbox("Gap Filling Method",
                    ["Interpolate", "Forward Fill", "Backward Fill", "Zero"], 
                    index=0,
                    key="m1_align_fill_method")  # ← UNIQUE KEY
            
            if st.button("🚀 Align & Resample", type="primary", use_container_width=True, key="m1_align_btn"):
                with st.spinner("Aligning timestamps..."):
                    aligned_data = {}
                    alignment_reports = {}
                    
                    for data_type, df in st.session_state.processed_data.items():
                        st.info(f"Aligning {data_type}...")
                        
                        if 'timestamp' not in df.columns:
                            st.warning(f"⚠️ {data_type}: No timestamp column, skipping")
                            aligned_data[data_type] = df
                            continue
                        
                        try:
                            df_copy = df.copy()
                            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
                            df_indexed = df_copy.set_index('timestamp')
                            
                            # Resample frequency mapping
                            freq_map = {
                                "1min": "1T", "5min": "5T", "15min": "15T",
                                "30min": "30T", "1hour": "1H"
                            }
                            freq_str = freq_map[target_freq]
                            
                            # Select ONLY numeric columns for resampling
                            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns
                            
                            if len(numeric_cols) == 0:
                                st.warning(f"⚠️ {data_type}: No numeric columns to resample, skipping")
                                aligned_data[data_type] = df
                                continue
                            
                            # Resample ONLY numeric columns
                            df_numeric = df_indexed[numeric_cols]
                            df_resampled = df_numeric.resample(freq_str).mean()
                            
                            # Fill gaps
                            initial_missing = df_resampled.isnull().sum().sum()
                            
                            if fill_method == "Interpolate":
                                df_resampled = df_resampled.interpolate(method='linear')
                            elif fill_method == "Forward Fill":
                                df_resampled = df_resampled.fillna(method='ffill')
                            elif fill_method == "Backward Fill":
                                df_resampled = df_resampled.fillna(method='bfill')
                            elif fill_method == "Zero":
                                df_resampled = df_resampled.fillna(0)
                            
                            final_missing = df_resampled.isnull().sum().sum()
                            gaps_filled = initial_missing - final_missing
                            
                            df_final = df_resampled.reset_index()
                            
                            aligned_data[data_type] = df_final
                            alignment_reports[data_type] = {
                                'original_rows': len(df),
                                'resampled_rows': len(df_final),
                                'gaps_filled': gaps_filled,
                                'target_frequency': target_freq,
                                'fill_method': fill_method
                            }
                            
                        except Exception as e:
                            st.error(f"❌ {data_type}: Alignment failed - {e}")
                            aligned_data[data_type] = df
                    
                    st.session_state.processed_data = aligned_data
                    st.session_state.alignment_reports = alignment_reports
                    
                    st.success("✅ Time alignment complete!")
                    
                    # Display alignment reports
                    st.subheader("📋 Alignment Reports")
                    
                    for data_type, report in alignment_reports.items():
                        with st.expander(f"⏱️ {data_type.title()} Alignment Report"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Original Rows", f"{report['original_rows']:,}")
                            with col2:
                                st.metric("Resampled Rows", f"{report['resampled_rows']:,}")
                            with col3:
                                st.metric("Gaps Filled", f"{report['gaps_filled']:,}")
                            
                            st.write(f"**Target Frequency:** {report['target_frequency']}")
                            st.write(f"**Fill Method:** {report['fill_method']}")

    
   # ==================== TAB 4: VISUALIZATIONS ====================
    with viz_tab:
        st.subheader("📊 Data Quality Visualizations")
        
        if not st.session_state.processed_data:
            st.warning("⚠️ Please load data first")
        else:
            st.info("📈 Visual analysis of your health data quality and patterns")
            
            # Create visualizations for each dataset
            for data_type, df in st.session_state.processed_data.items():
                with st.expander(f"📈 {data_type.replace('_', ' ').title()} Visualizations", expanded=True):
                    
                    if 'timestamp' not in df.columns:
                        st.warning(f"⚠️ No timestamp column found in {data_type}")
                        continue
                    
                    if len(df.columns) < 2:
                        st.warning(f"⚠️ No data columns found in {data_type}")
                        continue
                    
                    # Get the main metric column (first numeric column after timestamp)
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) == 0:
                        st.warning(f"⚠️ No numeric columns found in {data_type}")
                        continue
                    
                    metric_col = numeric_cols[0]
                    
                    # Create two columns for layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Timeline Plot
                        st.markdown("**📈 Time Series Plot**")
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=df['timestamp'],
                            y=df[metric_col],
                            mode='lines+markers',
                            name=metric_col.replace('_', ' ').title(),
                            line=dict(color='#7c3aed', width=2),
                            marker=dict(size=4)
                        ))
                        
                        # Mark outliers if they exist
                        if 'is_anomaly' in df.columns:
                            outliers = df[df['is_anomaly'] == True]
                            if len(outliers) > 0:
                                fig.add_trace(go.Scatter(
                                    x=outliers['timestamp'],
                                    y=outliers[metric_col],
                                    mode='markers',
                                    name='Anomalies',
                                    marker=dict(color='red', size=10, symbol='x')
                                ))
                        
                        fig.update_layout(
                            title=f"{data_type.title()} Over Time",
                            xaxis_title="Time",
                            yaxis_title=metric_col.title(),
                            height=350,
                            hovermode='x unified',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribution Plot (Histogram)
                        st.markdown("**📊 Distribution Analysis**")
                        fig2 = go.Figure()
                        
                        fig2.add_trace(go.Histogram(
                            x=df[metric_col],
                            nbinsx=30,
                            name=metric_col,
                            marker=dict(color='#7c3aed', line=dict(color='white', width=1))
                        ))
                        
                        fig2.update_layout(
                            title=f"{metric_col.title()} Distribution",
                            xaxis_title=metric_col.title(),
                            yaxis_title="Frequency",
                            height=350,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Statistics Summary
                    st.markdown("**📋 Statistical Summary**")
                    
                    col_a, col_b, col_c, col_d, col_e = st.columns(5)
                    
                    with col_a:
                        st.metric("Count", f"{len(df):,}")
                    with col_b:
                        st.metric("Mean", f"{df[metric_col].mean():.2f}")
                    with col_c:
                        st.metric("Median", f"{df[metric_col].median():.2f}")
                    with col_d:
                        st.metric("Std Dev", f"{df[metric_col].std():.2f}")
                    with col_e:
                        st.metric("Range", f"{df[metric_col].max() - df[metric_col].min():.2f}")
                    
                    # Box Plot for outlier visualization
                    st.markdown("**📦 Box Plot (Outlier Detection)**")
                    
                    fig3 = go.Figure()
                    
                    fig3.add_trace(go.Box(
                        y=df[metric_col],
                        name=metric_col,
                        marker=dict(color='#7c3aed'),
                        boxmean='sd'
                    ))
                    
                    fig3.update_layout(
                        title=f"{metric_col.title()} Outlier Analysis",
                        yaxis_title=metric_col.title(),
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True)
            
            # Overall Summary
            st.markdown("---")
            st.subheader("📊 Overall Data Quality Summary")
            
            total_records = sum([len(df) for df in st.session_state.processed_data.values()])
            total_datasets = len(st.session_state.processed_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Datasets", total_datasets)
            with col2:
                st.metric("Total Records", f"{total_records:,}")
            with col3:
                if hasattr(st.session_state, 'cleaning_reports'):
                    avg_quality = np.mean([r.get('quality_score', 100) for r in st.session_state.cleaning_reports.values()])
                    st.metric("Avg Quality Score", f"{avg_quality:.1f}%")
                else:
                    st.metric("Data Quality", "Good ✅")

    
    # ==================== TAB 5: SUMMARY ====================
    with summary_tab:
        st.subheader("📋 Complete Processing Summary")
        
        if st.session_state.processed_data:
            # Overall metrics
            total_records = sum([len(df) for df in st.session_state.processed_data.values()])
            total_datasets = len(st.session_state.processed_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Datasets", total_datasets)
            with col2:
                st.metric("Total Records", f"{total_records:,}")
            with col3:
                avg_quality = 100
                if hasattr(st.session_state, 'cleaning_reports'):
                    avg_quality = np.mean([r['quality_score'] for r in st.session_state.cleaning_reports.values()])
                st.metric("Avg Quality", f"{avg_quality:.1f}%")
            with col4:
                st.metric("Pipeline Status", "✅ Complete")
            
            st.markdown("---")
            
            # Pipeline stages
            st.markdown("### 📊 Pipeline Stages Completed")
            
            stages = []
            if st.session_state.raw_data:
                stages.append("✅ **Data Loading**: Loaded successfully")
            if hasattr(st.session_state, 'cleaning_reports'):
                stages.append("✅ **Data Cleaning**: Quality checks passed")
            if hasattr(st.session_state, 'alignment_reports'):
                stages.append("✅ **Time Alignment**: Resampling complete")
            
            if stages:
                for stage in stages:
                    st.markdown(stage)
            else:
                st.info("⏳ Run the preprocessing pipeline")
            
            # Export processed data
            st.markdown("---")
            st.markdown("### 💾 Export Processed Data")
            
            if st.button("📥 Download Processed Data", use_container_width=True):
                # Create CSV for each dataset
                for data_type, df in st.session_state.processed_data.items():
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download {data_type}.csv",
                        data=csv,
                        file_name=f"processed_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("⏳ No data processed yet. Load data in the Upload Data tab.")


def handle_milestone2():
    """Milestone 2: Advanced Analytics"""
    
    st.header("🧬 Milestone 2: Advanced Analytics")
    
    if not st.session_state.processed_data:
        st.warning("⚠️ Load data in M1 first")
        return
    
    st.info("Extract features, generate forecasts, discover patterns")
    
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.session_state.config['feature_complexity'] = st.selectbox(
                "Complexity", ["Basic", "Standard", "Advanced"], index=1
            )
            st.session_state.config['window_size'] = st.slider(
                "Window (min)", 15, 120, 60, 15
            )
        
        with col2:
            st.session_state.config['forecast_periods'] = st.slider(
                "Forecast Periods", 50, 300, 120, 10
            )
            st.session_state.config['confidence_interval'] = st.slider(
                "Confidence", 0.80, 0.99, 0.95, 0.01
            )
        
        with col3:
            st.session_state.config['cluster_method'] = st.selectbox(
                "Clustering", ["kmeans", "dbscan", "gmm"], index=0
            )
            st.session_state.config['n_clusters'] = st.slider(
                "Clusters", 2, 8, 3
            )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔬 Extract Features", use_container_width=True):
            run_feature_extraction()
    
    with col2:
        if st.button("📈 Generate Forecasts", use_container_width=True):
            run_forecasting()
    
    with col3:
        if st.button("🎯 Discover Patterns", use_container_width=True):
            run_clustering()
    
    display_m2_results()

def run_feature_extraction():
    """Run feature extraction"""
    
    with st.spinner("🔬 Extracting features..."):
        try:
            cfg = st.session_state.config
            features = st.session_state.feature_engine.extract_all_features(
                st.session_state.processed_data,
                window_size=cfg['window_size'],
                complexity=cfg['feature_complexity'],
                overlap_percentage=cfg['window_overlap']
            )
            
            if 'features' not in st.session_state.m2_results:
                st.session_state.m2_results['features'] = {}
            st.session_state.m2_results['features'].update(features)
            
            total = sum([len(f.columns) for f in features.values()])
            st.success(f"✅ Extracted {total} features!")
            
        except Exception as e:
            st.error(f"❌ Failed: {e}")

def run_forecasting():
    """Run forecasting"""
    
    with st.spinner("📈 Training Prophet..."):
        try:
            cfg = st.session_state.config
            forecasts = st.session_state.forecast_engine.forecast_all_metrics(
                st.session_state.processed_data,
                forecast_periods=cfg['forecast_periods'],
                confidence_interval=cfg['confidence_interval'],
                detect_anomalies=True
            )
            
            if 'forecasts' not in st.session_state.m2_results:
                st.session_state.m2_results['forecasts'] = {}
            st.session_state.m2_results['forecasts'].update(forecasts)
            
            total = sum([len(f['anomalies']) for f in forecasts.values()])
            st.success(f"✅ Forecasts generated! {total} anomalies found")
            
        except Exception as e:
            st.error(f"❌ Failed: {e}")

def run_clustering():
    """Run clustering"""
    
    if 'features' not in st.session_state.m2_results:
        st.error("❌ Extract features first!")
        return
    
    with st.spinner("🎯 Discovering patterns..."):
        try:
            cfg = st.session_state.config
            clusters = st.session_state.pattern_analyzer.analyze_patterns(
                st.session_state.m2_results['features'],
                method=cfg['cluster_method'],
                n_clusters=cfg['n_clusters'],
                scaling_method=cfg['scaling_method'],
                auto_tune=cfg['auto_tune']
            )
            
            if 'clusters' not in st.session_state.m2_results:
                st.session_state.m2_results['clusters'] = {}
            st.session_state.m2_results['clusters'].update(clusters)
            
            st.success(f"✅ Discovered {cfg['n_clusters']} patterns!")
            
        except Exception as e:
            st.error(f"❌ Failed: {e}")

def display_m2_results():
    """Display M2 results with visualizations"""
    
    results = st.session_state.m2_results
    
    if not results:
        return
    
    st.subheader("📊 Analysis Results")
    
    if 'features' in results and VISUALS_AVAILABLE:
        M2Visualizations.display_feature_results(results['features'])
    elif 'features' in results:
        with st.expander(f"🔬 Features ({len(results['features'])} datasets)"):
            for metric, features in results['features'].items():
                st.write(f"**{metric.title()}:** {len(features.columns)} features, {len(features)} windows")
    
    if 'forecasts' in results and VISUALS_AVAILABLE:
        M2Visualizations.display_forecast_results(results['forecasts'], st.session_state.processed_data)
    elif 'forecasts' in results:
        with st.expander(f"📈 Forecasts ({len(results['forecasts'])} datasets)"):
            for metric, forecast in results['forecasts'].items():
                mape = forecast['metrics']['mape']
                anomalies = len(forecast['anomalies'])
                st.write(f"**{metric.title()}:** MAPE {mape:.2f}%, {anomalies} anomalies")
    
    if 'clusters' in results and VISUALS_AVAILABLE:
        M2Visualizations.display_cluster_results(results['clusters'])
    elif 'clusters' in results:
        with st.expander(f"🎯 Patterns ({len(results['clusters'])} datasets)"):
            for metric, cluster in results['clusters'].items():
                silhouette = cluster['quality_metrics']['silhouette_score']
                n_clusters = cluster['quality_metrics']['n_clusters']
                st.write(f"**{metric.title()}:** {n_clusters} clusters, Silhouette {silhouette:.3f}")

def handle_milestone3():
    """Milestone 3: Enhanced Anomaly Detection"""
    
    st.header("🚨 Milestone 3: Enhanced Anomaly Detection")
    
    if not st.session_state.processed_data:
        st.warning("⚠️ Load data in M1 first")
        return
    
    st.info("🔍 Uses traditional methods + M2 features/forecasts for superior detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_features = st.checkbox(
            "🔬 Use M2 Features", 
            value='features' in st.session_state.m2_results,
            disabled='features' not in st.session_state.m2_results
        )
    
    with col2:
        use_forecasts = st.checkbox(
            "📈 Use M2 Forecasts", 
            value='forecasts' in st.session_state.m2_results,
            disabled='forecasts' not in st.session_state.m2_results
        )
    
    if st.button("🚀 Run Enhanced Detection", type="primary", use_container_width=True):
        
        with st.spinner("🔍 Detecting anomalies..."):
            detector = EnhancedAnomalyDetector()
            
            m2_features = st.session_state.m2_results.get('features') if use_features else None
            m2_forecasts = st.session_state.m2_results.get('forecasts') if use_forecasts else None
            
            results = detector.detect_enhanced_anomalies(
                st.session_state.processed_data,
                m2_features=m2_features,
                m2_forecasts=m2_forecasts
            )
            
            st.session_state.anomaly_results = results
            
            total = sum([r['counts']['total'] for r in results.values()])
            st.success(f"✅ Detected {total} anomalies!")
            
            st.subheader("📊 Detection Summary")
            
            summary_data = []
            for data_type, result in results.items():
                counts = result['counts']
                summary_data.append({
                    'Dataset': data_type.title(),
                    'Threshold': counts['threshold'],
                    'Statistical': counts['statistical'],
                    'Feature-based': counts['feature_based'],
                    'Forecast-based': counts['forecast_based'],
                    'Total': counts['total']
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
            
            display_anomaly_visualizations(results)

def display_anomaly_visualizations(anomaly_results):
    """Display anomaly visualizations"""
    
    st.subheader("📈 Anomaly Visualizations")
    
    for data_type, result in anomaly_results.items():
        with st.expander(f"🔍 {data_type.title()} Anomalies"):
            
            df = result['data_with_anomalies']
            metric_col = df.columns[1]
            
            fig = go.Figure()
            
            normal = df[~df['is_anomaly']]
            fig.add_trace(go.Scatter(
                x=normal['timestamp'],
                y=normal[metric_col],
                mode='lines',
                name='Normal',
                line=dict(color='blue', width=1)
            ))
            
            anomaly_data = df[df['is_anomaly']]
            
            for anom_type, color in [
                ('threshold_high', 'red'),
                ('threshold_low', 'orange'),
                ('statistical', 'purple'),
                ('feature_based', 'green'),
                ('forecast_based', 'gold')
            ]:
                type_data = anomaly_data[anomaly_data['anomaly_type'] == anom_type]
                if not type_data.empty:
                    fig.add_trace(go.Scatter(
                        x=type_data['timestamp'],
                        y=type_data[metric_col],
                        mode='markers',
                        name=anom_type.replace('_', ' ').title(),
                        marker=dict(color=color, size=8)
                    ))
            
            fig.update_layout(
                title=f"{data_type.title()} - Enhanced Detection",
                xaxis_title="Time",
                yaxis_title=metric_col.title(),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            if not anomaly_data.empty:
                st.write("**Detected Anomalies:**")
                summary = anomaly_data[['timestamp', metric_col, 'anomaly_type', 'anomaly_severity', 'anomaly_description']].head(10)
                st.dataframe(summary, use_container_width=True)

def handle_milestone4():
    """Milestone 4: Insights Dashboard"""
    
    st.header("📋 Milestone 4: Health Insights Dashboard")
    
    if not st.session_state.anomaly_results:
        st.warning("⚠️ Run anomaly detection in M3 first")
        return
    
    st.success("🎉 Pipeline Complete! All milestones connected")
    
    st.session_state.dashboard_ready = True
    
    # Display with visualizations if available
    if VISUALS_AVAILABLE:
        M4Visualizations.display_executive_dashboard(
            st.session_state.anomaly_results,
            st.session_state.processed_data,
            st.session_state.m2_results
        )
    else:
        # Fallback: basic display
        display_basic_dashboard()
    
    # ==================== EXPORT BUTTONS - ALWAYS SHOW ====================
    st.markdown("---")
    st.subheader("💾 Export Complete Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export Text Report", use_container_width=True, key="m4_unified_btn_text"):
            generate_unified_report()
    
    with col2:
        if st.button("📊 Export Anomalies CSV", use_container_width=True, key="m4_unified_btn_csv"):
            export_anomaly_data()
    
    with col3:
        if st.button("📑 Download PDF Report", type="primary", use_container_width=True, key="m4_unified_btn_pdf"):
            generate_pdf_report()


def display_basic_dashboard():
    """Basic dashboard (fallback)"""
    
    st.subheader("🎯 Key Insights")
    
    total_datasets = len(st.session_state.processed_data)
    total_records = sum([len(df) for df in st.session_state.processed_data.values()])
    total_anomalies = sum([r['counts']['total'] for r in st.session_state.anomaly_results.values()])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Datasets", total_datasets)
    
    with col2:
        st.metric("📝 Data Points", f"{total_records:,}")
    
    with col3:
        st.metric("🚨 Anomalies", total_anomalies)
    
    with col4:
        rate = (total_anomalies / total_records) * 100 if total_records > 0 else 0
        st.metric("📊 Anomaly Rate", f"{rate:.2f}%")
    
    st.subheader("📈 Dataset Details")
    
    for data_type in st.session_state.processed_data.keys():
        with st.expander(f"🔍 {data_type.title()} Analysis"):
            
            df = st.session_state.processed_data[data_type]
            metric_col = df.columns[1]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**📊 Data Summary:**")
                st.write(f"• Records: {len(df):,}")
                st.write(f"• Mean: {df[metric_col].mean():.2f}")
                st.write(f"• Std Dev: {df[metric_col].std():.2f}")
                
                if data_type in st.session_state.m2_results.get('features', {}):
                    features = st.session_state.m2_results['features'][data_type]
                    st.write(f"• Features: {len(features.columns)}")
                
                if data_type in st.session_state.m2_results.get('forecasts', {}):
                    forecast = st.session_state.m2_results['forecasts'][data_type]
                    st.write(f"• Forecast MAPE: {forecast['metrics']['mape']:.2f}%")
            
            with col2:
                if data_type in st.session_state.anomaly_results:
                    result = st.session_state.anomaly_results[data_type]
                    counts = result['counts']
                    
                    st.write("**🚨 Anomalies:**")
                    st.write(f"• Threshold: {counts['threshold']}")
                    st.write(f"• Statistical: {counts['statistical']}")
                    st.write(f"• Feature: {counts['feature_based']}")
                    st.write(f"• Forecast: {counts['forecast_based']}")
                    st.write(f"• **Total: {counts['total']}**")


# ============================================================================
# EXPORT FUNCTIONS - Module Level (NOT inside any other function)
# ============================================================================


def generate_unified_report():
    """Generate comprehensive unified text report"""
    
    report_lines = [
        "=" * 80,
        "UNIFIED HEALTH DATA ANALYTICS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "EXECUTIVE SUMMARY",
        "-" * 20,
    ]
    
    total_datasets = len(st.session_state.processed_data)
    total_records = sum([len(df) for df in st.session_state.processed_data.values()])
    total_anomalies = sum([r['counts']['total'] for r in st.session_state.anomaly_results.values()])
    
    report_lines.extend([
        f"Total Datasets Analyzed: {total_datasets}",
        f"Total Health Records: {total_records:,}",
        f"Total Anomalies Detected: {total_anomalies}",
        f"Overall Anomaly Rate: {(total_anomalies/total_records)*100:.2f}%" if total_records > 0 else "Overall Anomaly Rate: 0%",
        ""
    ])
    
    for data_type in st.session_state.processed_data.keys():
        report_lines.extend([
            "",
            f"{data_type.upper().replace('_', ' ')} ANALYSIS",
            "-" * 30,
        ])
        
        df = st.session_state.processed_data[data_type]
        metric_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        report_lines.extend([
            f"Total Records: {len(df):,}",
            f"Mean {metric_col}: {df[metric_col].mean():.2f}",
            f"Standard Deviation: {df[metric_col].std():.2f}",
            f"Minimum Value: {df[metric_col].min():.2f}",
            f"Maximum Value: {df[metric_col].max():.2f}",
        ])
        
        if data_type in st.session_state.anomaly_results:
            counts = st.session_state.anomaly_results[data_type]['counts']
            report_lines.extend([
                "",
                "Anomaly Breakdown:",
                f"  - Threshold-based: {counts['threshold']}",
                f"  - Statistical: {counts['statistical']}",
                f"  - Feature-based: {counts['feature_based']}",
                f"  - Forecast-based: {counts['forecast_based']}",
                f"  - TOTAL: {counts['total']}",
            ])
            
            anomaly_rate = (counts['total'] / len(df) * 100) if len(df) > 0 else 0
            report_lines.append(f"  - Anomaly Rate: {anomaly_rate:.2f}%")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report_text = "\n".join(report_lines)
    
    st.download_button(
        label="📥 Download Text Report",
        data=report_text,
        file_name=f"fitpulse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.success("✅ Report generated!")


def export_anomaly_data():
    """Export all anomaly data as CSV"""
    
    all_anomalies = []
    
    for data_type, result in st.session_state.anomaly_results.items():
        df = result['data_with_anomalies']
        anomaly_rows = df[df['is_anomaly']].copy()
        anomaly_rows['dataset'] = data_type
        all_anomalies.append(anomaly_rows)
    
    if all_anomalies:
        combined = pd.concat(all_anomalies, ignore_index=True)
        
        priority_cols = ['dataset', 'timestamp', 'anomaly_severity', 'anomaly_type', 'anomaly_description']
        other_cols = [col for col in combined.columns if col not in priority_cols]
        ordered_cols = [col for col in (priority_cols + other_cols) if col in combined.columns]
        combined = combined[ordered_cols]
        
        csv = combined.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success(f"✅ {len(combined)} anomalies exported!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", len(combined))
        with col2:
            st.metric("High Risk", (combined['anomaly_severity'] == 'High').sum())
        with col3:
            st.metric("Datasets", combined['dataset'].nunique())
    else:
        st.error("No anomalies to export")


def generate_pdf_report():
    """Generate PDF with embedded charts"""
    
    if not PDF_AVAILABLE:
        st.error("❌ PDF unavailable - install fpdf")
        return
    
    if not VISUALS_AVAILABLE:
        st.warning("⚠️ Visualizations unavailable")
        return
    
    try:
        with st.spinner("Generating PDF..."):
            from fitpulse_visualizations import M4Visualizations
            from fitpulse_pdf_exporter import EnhancedPDFExporter
            
            insights = M4Visualizations._generate_insights(
                st.session_state.anomaly_results,
                st.session_state.processed_data
            )
            
            pdf_bytes = EnhancedPDFExporter.export_comprehensive_pdf(
                insights,
                st.session_state.anomaly_results
            )
            
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"fitpulse_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            
            st.success("✅ PDF ready!")
    except Exception as e:
        st.error(f"❌ Failed: {e}")


if __name__ == "__main__":
    main()


