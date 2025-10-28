# Milestone 3: Anomaly Detection and Visualization
# Detecting Unusual Health Patterns Using Fitness Watch Data
# COMPLETE WORKING VERSION - WITH PROS SUPPORT + FILE SAVING + ALL PURPLE BUTTONS



import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os
import glob
warnings.filterwarnings('ignore')



# For statistical calculations
from scipy import stats
from scipy.signal import find_peaks



# For saving results
import json
import io




# ============================================================================
# CUSTOM STYLING FOR ALL PURPLE BUTTONS
# ============================================================================

def apply_purple_button_style():
    """Apply custom purple styling to ALL buttons"""
    st.markdown("""
        <style>
        /* Purple gradient button styling for ALL buttons */
        div.stButton > button,
        div.stDownloadButton > button {
            background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 16px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(124, 58, 237, 0.3) !important;
        }
        
        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(124, 58, 237, 0.5) !important;
        }
        
        div.stButton > button:active,
        div.stDownloadButton > button:active {
            transform: translateY(0px);
            box-shadow: 0 2px 4px rgba(124, 58, 237, 0.3) !important;
        }
        </style>
    """, unsafe_allow_html=True)




# ============================================================================
# DATA LOADER - LOADS ALL CSV FILES
# ============================================================================



class AutoDataLoader:
    """Automatically loads ALL CSV files from data/input/ folder"""
    
    def __init__(self, data_folder='data/input'):
        self.data_folder = data_folder
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load ALL CSV files from data folder"""
        loaded_data = {}
        
        if not os.path.exists(self.data_folder):
            st.error(f"❌ Folder '{self.data_folder}' not found!")
            return loaded_data
        
        # Find ALL CSV files
        csv_files = glob.glob(os.path.join(self.data_folder, '*.csv'))
        
        if not csv_files:
            st.warning(f"⚠️ No CSV files found in {self.data_folder}")
            return loaded_data
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            
            try:
                df = pd.read_csv(filepath)
                
                # Process timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Auto-detect data type
                filename_lower = filename.lower()
                
                if 'heart' in filename_lower or 'hr' in filename_lower:
                    data_type = 'heart_rate'
                elif 'step' in filename_lower:
                    data_type = 'steps'
                elif 'activity' in filename_lower:
                    data_type = 'activity'
                elif 'sleep' in filename_lower:
                    data_type = 'sleep'
                elif 'pros' in filename_lower:
                    data_type = 'pros'
                else:
                    # Use filename as type
                    data_type = filename.replace('.csv', '').replace('.json', '')
                
                loaded_data[data_type] = df
                st.success(f"✅ Loaded {filename}: {len(df)} records")
                
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
        
        return loaded_data




# ============================================================================
# ANOMALY DETECTION METHODS
# ============================================================================



class ThresholdAnomalyDetector:
    """Rule-based anomaly detection using configurable thresholds"""
    
    def __init__(self):
        self.threshold_rules = {
            'heart_rate': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'Heart rate outside normal resting range'
            },
            'steps': {
                'metric_name': 'step_count',
                'min_threshold': 0,
                'max_threshold': 1000,
                'sustained_minutes': 5,
                'description': 'Unrealistic step count detected'
            },
            'activity': {
                'metric_name': 'activity_level',
                'min_threshold': 0,
                'max_threshold': 100,
                'sustained_minutes': 5,
                'description': 'Unusual activity level'
            },
            'sleep': {
                'metric_name': 'duration_minutes',
                'min_threshold': 180,
                'max_threshold': 720,
                'sustained_minutes': 0,
                'description': 'Unusual sleep duration'
            },
            'pros': {
                'metric_name': 'heart_rate',
                'min_threshold': 40,
                'max_threshold': 120,
                'sustained_minutes': 10,
                'description': 'Prosthetic sensor - heart rate anomaly'
            }
        }
        self.detected_anomalies = []
    
    def detect_anomalies(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect threshold-based anomalies"""
        
        st.info(f"🔍 Running threshold-based anomaly detection on {data_type}...")
        
        report = {
            'method': 'Threshold-Based',
            'data_type': data_type,
            'total_records': len(df),
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0,
            'threshold_info': {}
        }
        
        if data_type not in self.threshold_rules:
            st.warning(f"No threshold rules defined for {data_type}")
            return df, report
        
        rule = self.threshold_rules[data_type]
        metric_col = rule['metric_name']
        
        if metric_col not in df.columns:
            st.error(f"Metric column '{metric_col}' not found in data")
            return df, report
        
        df_result = df.copy()
        df_result['threshold_anomaly'] = False
        df_result['anomaly_reason'] = ''
        df_result['severity'] = 'Normal'
        
        # Detect violations
        too_high = df_result[metric_col] > rule['max_threshold']
        too_low = df_result[metric_col] < rule['min_threshold']
        
        # Apply sustained duration filter
        if rule['sustained_minutes'] > 0:
            window_size = rule['sustained_minutes']
            too_high_sustained = too_high.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            too_low_sustained = too_low.rolling(window=window_size, min_periods=window_size).sum() >= window_size
            
            df_result.loc[too_high_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_high_sustained, 'anomaly_reason'] = f'High {metric_col} (>{rule["max_threshold"]})'
            df_result.loc[too_high_sustained, 'severity'] = 'High'
            
            df_result.loc[too_low_sustained, 'threshold_anomaly'] = True
            df_result.loc[too_low_sustained, 'anomaly_reason'] = f'Low {metric_col} (<{rule["min_threshold"]})'
            df_result.loc[too_low_sustained, 'severity'] = 'Medium'
        else:
            df_result.loc[too_high, 'threshold_anomaly'] = True
            df_result.loc[too_high, 'anomaly_reason'] = f'Excessive {metric_col}'
            df_result.loc[too_high, 'severity'] = 'Medium'
            
            df_result.loc[too_low, 'threshold_anomaly'] = True
            df_result.loc[too_low, 'anomaly_reason'] = f'Insufficient {metric_col}'
            df_result.loc[too_low, 'severity'] = 'High'
        
        # Calculate statistics
        anomaly_count = df_result['threshold_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        report['threshold_info'] = {
            'min_threshold': rule['min_threshold'],
            'max_threshold': rule['max_threshold'],
            'sustained_minutes': rule['sustained_minutes']
        }
        
        if anomaly_count > 0:
            st.success(f"✅ Detected {anomaly_count} threshold anomalies ({report['anomaly_percentage']:.2f}%)")
        else:
            st.info("No threshold-based anomalies detected")
        
        return df_result, report




class ResidualAnomalyDetector:
    """Model-based anomaly detection using statistical residuals"""
    
    def __init__(self, threshold_std: float = 3.0):
        self.threshold_std = threshold_std
        self.detected_anomalies = []
    
    def detect_anomalies_statistical(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Detect anomalies using statistical methods (Z-score)"""
        
        st.info(f"🔍 Running statistical residual detection on {data_type}...")
        
        report = {
            'method': 'Statistical Residual-Based',
            'data_type': data_type,
            'threshold_std': self.threshold_std,
            'anomalies_detected': 0,
            'anomaly_percentage': 0.0
        }
        
        metric_columns = {
            'heart_rate': 'heart_rate',
            'steps': 'step_count',
            'activity': 'activity_level',
            'sleep': 'duration_minutes',
            'pros': 'heart_rate'
        }
        
        metric_col = metric_columns.get(data_type)
        
        if not metric_col or metric_col not in df.columns:
            st.warning(f"Cannot perform statistical detection on {data_type}")
            return df, report
        
        df_result = df.copy()
        
        # Calculate rolling statistics
        window = min(50, len(df) // 10)
        if window < 5:
            window = 5
        
        df_result['rolling_mean'] = df_result[metric_col].rolling(window=window, center=True).mean()
        df_result['rolling_std'] = df_result[metric_col].rolling(window=window, center=True).std()
        
        # Fill NaN values
        df_result['rolling_mean'].fillna(df_result[metric_col].mean(), inplace=True)
        df_result['rolling_std'].fillna(df_result[metric_col].std(), inplace=True)
        
        # Calculate Z-score
        df_result['z_score'] = (df_result[metric_col] - df_result['rolling_mean']) / (df_result['rolling_std'] + 1e-6)
        
        # Detect anomalies
        df_result['residual_anomaly'] = np.abs(df_result['z_score']) > self.threshold_std
        df_result['residual_anomaly_reason'] = ''
        df_result.loc[df_result['residual_anomaly'], 'residual_anomaly_reason'] = 'Statistical deviation detected'
        
        anomaly_count = df_result['residual_anomaly'].sum()
        report['anomalies_detected'] = int(anomaly_count)
        report['anomaly_percentage'] = (anomaly_count / len(df_result)) * 100
        
        if anomaly_count > 0:
            st.success(f"✅ Detected {anomaly_count} statistical anomalies ({report['anomaly_percentage']:.2f}%)")
        else:
            st.info("No statistical anomalies detected")
        
        return df_result, report




# ============================================================================
# ANOMALY VISUALIZATION
# ============================================================================



class AnomalyVisualizer:
    """Creates interactive visualizations highlighting detected anomalies"""
    
    def __init__(self):
        self.color_scheme = {
            'normal': '#1f77b4',
            'threshold': '#ff7f0e',
            'residual': '#d62728',
            'cluster': '#9467bd'
        }
    
    def plot_anomalies(self, df: pd.DataFrame, data_type: str, metric_col: str):
        """Universal anomaly plotter"""
        
        has_threshold = 'threshold_anomaly' in df.columns
        has_residual = 'residual_anomaly' in df.columns
        
        fig = go.Figure()
        
        # Normal data
        normal_df = df.copy()
        if has_threshold:
            normal_df = normal_df[~normal_df['threshold_anomaly']]
        
        # Choose plot type based on data
        if data_type in ['steps', 'activity']:
            fig.add_trace(go.Bar(
                x=normal_df['timestamp'],
                y=normal_df[metric_col],
                name='Normal',
                marker_color=self.color_scheme['normal'],
                opacity=0.7
            ))
        else:
            fig.add_trace(go.Scatter(
                x=normal_df['timestamp'],
                y=normal_df[metric_col],
                mode='lines',
                name='Normal',
                line=dict(color=self.color_scheme['normal'], width=2)
            ))
        
        # Rolling mean if available
        if 'rolling_mean' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['rolling_mean'],
                mode='lines',
                name='Trend',
                line=dict(color='lightgreen', width=2, dash='dash')
            ))
        
        # Threshold anomalies
        if has_threshold:
            threshold_anomalies = df[df['threshold_anomaly']]
            if len(threshold_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=threshold_anomalies['timestamp'],
                    y=threshold_anomalies[metric_col],
                    mode='markers',
                    name='Threshold Anomalies',
                    marker=dict(
                        color=self.color_scheme['threshold'],
                        size=12,
                        symbol='x',
                        line=dict(width=2)
                    )
                ))
        
        # Residual anomalies
        if has_residual:
            residual_anomalies = df[df['residual_anomaly']]
            if len(residual_anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=residual_anomalies['timestamp'],
                    y=residual_anomalies[metric_col],
                    mode='markers',
                    name='Statistical Anomalies',
                    marker=dict(
                        color=self.color_scheme['residual'],
                        size=12,
                        symbol='diamond',
                        line=dict(width=2)
                    )
                ))
        
        # Format title
        chart_title = data_type.replace('_', ' ').title() + " - Anomaly Detection"
        ylabel_text = metric_col.replace('_', ' ').title()
        
        fig.update_layout(
            title=chart_title,
            xaxis_title="Time",
            yaxis_title=ylabel_text,
            hovermode='x unified',
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_anomaly_summary_dashboard(self, all_reports: Dict):
        """Create comprehensive summary dashboard"""
        
        st.subheader("📊 Anomaly Detection Summary Dashboard")
        
        total_anomalies = 0
        detection_methods = []
        
        for data_type, reports in all_reports.items():
            for method, report in reports.items():
                if 'anomalies_detected' in report:
                    total_anomalies += report['anomalies_detected']
                    detection_methods.append({
                        'Data Type': data_type.replace('_', ' ').title(),
                        'Method': report.get('method', method),
                        'Anomalies': report['anomalies_detected'],
                        'Percentage': f"{report.get('anomaly_percentage', 0):.2f}%"
                    })
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies Detected", total_anomalies)
        with col2:
            methods_used = len(set([d['Method'] for d in detection_methods]))
            st.metric("Detection Methods Used", methods_used)
        with col3:
            data_types = len(set([d['Data Type'] for d in detection_methods]))
            st.metric("Data Types Analyzed", data_types)
        
        # Detailed table
        if detection_methods:
            st.subheader("Anomaly Detection Breakdown")
            summary_df = pd.DataFrame(detection_methods)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Bar chart
            fig = px.bar(
                summary_df,
                x='Data Type',
                y='Anomalies',
                color='Method',
                title="Anomalies Detected by Method and Data Type",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)




# ============================================================================
# MILESTONE 3 PIPELINE
# ============================================================================



class AnomalyDetectionPipeline:
    """Complete Milestone 3 pipeline"""
    
    def __init__(self):
        self.threshold_detector = ThresholdAnomalyDetector()
        self.residual_detector = ResidualAnomalyDetector(threshold_std=3.0)
        self.visualizer = AnomalyVisualizer()
        self.data_loader = AutoDataLoader()
        
        self.processed_data = {}
        self.anomaly_reports = {}
    
    def run_complete_milestone3(self, preprocessed_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run complete anomaly detection pipeline"""
        
        st.header("🚨 Milestone 3: Anomaly Detection and Visualization")
        st.markdown("**Detecting unusual patterns using multiple methods**")
        
        results = {
            'data_with_anomalies': {},
            'reports': {}
        }
        
        # Process each data type
        for data_type, df in preprocessed_data.items():
            st.markdown("---")
            st.subheader(f"🔍 Analyzing {data_type.replace('_', ' ').title()}")
            
            results['reports'][data_type] = {}
            
            # Identify metric column
            metric_cols = {
                'heart_rate': 'heart_rate',
                'steps': 'step_count',
                'activity': 'activity_level',
                'sleep': 'duration_minutes',
                'pros': 'heart_rate'
            }
            
            metric_col = metric_cols.get(data_type)
            if not metric_col or metric_col not in df.columns:
                st.warning(f"Cannot process {data_type} - metric column not found")
                continue
            
            # Method 1: Threshold detection
            with st.expander(f"🟡 Method 1: Threshold-Based Detection - {data_type}", expanded=True):
                df_with_threshold, threshold_report = self.threshold_detector.detect_anomalies(df, data_type)
                results['reports'][data_type]['threshold'] = threshold_report
                
                if 'threshold_info' in threshold_report:
                    info = threshold_report['threshold_info']
                    st.write(f"**Thresholds:** Min={info['min_threshold']}, Max={info['max_threshold']}, "
                           f"Sustained={info['sustained_minutes']} minutes")
            
            # Method 2: Statistical detection
            with st.expander(f"🔴 Method 2: Statistical Residual Detection - {data_type}", expanded=True):
                df_final, residual_report = self.residual_detector.detect_anomalies_statistical(df_with_threshold, data_type)
                results['reports'][data_type]['residual'] = residual_report
            
            results['data_with_anomalies'][data_type] = df_final
            self.processed_data[data_type] = df_final
            
            # Visualizations
            st.subheader(f"📈 Visualizations - {data_type.replace('_', ' ').title()}")
            self.visualizer.plot_anomalies(df_final, data_type, metric_col)
        
        # Summary dashboard
        st.markdown("---")
        self.visualizer.create_anomaly_summary_dashboard(results['reports'])
        
        # SAVE RESULTS BACK TO FILES
        st.markdown("---")
        st.header("💾 Saving Results to Files")
        
        saved_count = 0
        for data_type, df in results['data_with_anomalies'].items():
            try:
                # Determine filename
                filename_map = {
                    'heart_rate': 'heart_rate.csv',
                    'steps': 'steps.csv',
                    'activity': 'activity.csv',
                    'pros': 'pros.csv',
                    'sleep': 'sleep.csv'
                }
                
                filename = filename_map.get(data_type, f"{data_type}.csv")
                filepath = os.path.join(self.data_loader.data_folder, filename)
                
                # Save with anomaly columns
                df.to_csv(filepath, index=False)
                st.success(f"✅ Saved {filename} with anomaly columns ({len(df):,} records)")
                saved_count += 1
                
            except Exception as e:
                st.error(f"❌ Error saving {data_type}: {str(e)}")
        
        if saved_count > 0:
            st.success(f"🎉 Successfully saved {saved_count} file(s) with anomaly data! Milestone 4 can now read them.")
        
        # Export
        self._generate_export_options(results)
        
        self.anomaly_reports = results
        return results
    
    def _generate_export_options(self, results: Dict):
        """Generate export functionality with PURPLE BUTTONS"""
        
        st.markdown("---")
        st.subheader("📥 Export Additional Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export Anomaly Report (JSON)", use_container_width=True):
                report_json = json.dumps(results['reports'], indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=report_json,
                    file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_json"
                )
        
        with col2:
            if st.button("📊 Export Anomaly Data (CSV)", use_container_width=True):
                all_anomalies = []
                for data_type, df in results['data_with_anomalies'].items():
                    anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower()]
                    if anomaly_cols:
                        anomaly_mask = df[anomaly_cols].any(axis=1)
                        anomalies = df[anomaly_mask].copy()
                        anomalies['data_type'] = data_type
                        all_anomalies.append(anomalies)
                
                if all_anomalies:
                    combined_df = pd.concat(all_anomalies, ignore_index=True)
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv"
                    )




# ============================================================================
# MAIN APP
# ============================================================================



def main():
    st.set_page_config(
        page_title="Milestone 3: Anomaly Detection",
        page_icon="🚨",
        layout="wide"
    )
    
    # Apply purple button styling to ALL BUTTONS
    apply_purple_button_style()
    
    st.title("🚨 Milestone 3: Anomaly Detection and Visualization")
    st.markdown("**Detecting Unusual Health Patterns Using Fitness Watch Data**")
    
    # Sidebar
    st.sidebar.header("⚙️ Milestone 3 Configuration")
    
    threshold_std = st.sidebar.slider(
        "Statistical Detection Threshold (std)",
        min_value=1.0, max_value=5.0, value=3.0, step=0.5
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Detection Methods:
    1. **Threshold-Based**: Rule-based detection
    2. **Statistical**: Z-score based detection
    
    ### Supported Data Types:
    - Heart Rate
    - Steps
    - Activity
    - Sleep
    - **Pros** (Prosthetic Sensors)
    """)
    
    # Initialize pipeline
    if 'milestone3_pipeline' not in st.session_state:
        st.session_state.milestone3_pipeline = AnomalyDetectionPipeline()
    
    st.session_state.milestone3_pipeline.residual_detector.threshold_std = threshold_std
    
    # Info
    st.info("""
    **Milestone 3 Objectives:**
    - Implement multiple anomaly detection methods
    - Create interactive visualizations with anomaly highlights
    - Generate comprehensive anomaly reports
    - **Save results to CSV files** for Milestone 4
    """)
    
    # Load data
    st.markdown("---")
    with st.spinner("📂 Loading data from data/input/..."):
        health_data = st.session_state.milestone3_pipeline.data_loader.load_all_data()
    
    if len(health_data) == 0:
        st.error("❌ No data loaded! Please ensure CSV files exist in data/input/")
        st.stop()
    
    st.success(f"✅ Loaded {len(health_data)} dataset(s)")
    
    # Run detection with PURPLE BUTTON
    if st.button("🚀 Run Anomaly Detection", use_container_width=True):
        with st.spinner("Running anomaly detection pipeline..."):
            results = st.session_state.milestone3_pipeline.run_complete_milestone3(health_data)
            st.session_state.milestone3_results = results
            st.success("✅ Anomaly detection complete!")




if __name__ == "__main__":
    main()
