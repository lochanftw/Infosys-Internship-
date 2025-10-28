"""
FitPulse Analytics Pro - Milestone 2
Main Application Entry Point - FIXED VERSION WITH COMPLETE REPORT DISPLAY
"""


import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings('ignore')


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


from milestone2.feature_engineering import AdvancedFeatureEngine
from milestone2.time_series_analysis import TrendForecastingEngine
from milestone2.pattern_clustering import BehaviorPatternAnalyzer
from milestone2.data_generator import generate_comprehensive_dataset
from milestone2 import utils



def initialize_session():
    """Initialize all session state variables"""
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if 'feature_engine' not in st.session_state:
        st.session_state.feature_engine = AdvancedFeatureEngine()
    
    if 'forecast_engine' not in st.session_state:
        st.session_state.forecast_engine = TrendForecastingEngine()
    
    if 'pattern_analyzer' not in st.session_state:
        st.session_state.pattern_analyzer = BehaviorPatternAnalyzer()
    
    if 'results' not in st.session_state:
        st.session_state.results = {}
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False



def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="FitPulse Milestone 2 - Advanced Analytics",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    utils.apply_custom_styling()
    
    # Initialize session state
    initialize_session()
    
    # Main header
    st.markdown("""
        <div class="main-header">
            <h1>🧬 FitPulse Milestone 2: Advanced Analytics Platform</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                TSFresh Feature Engineering • Prophet Forecasting • Behavioral Clustering
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Management",
        "🔬 Feature Extraction", 
        "📈 Time Series Forecasting",
        "🎯 Pattern Clustering",
        "📋 Comprehensive Report"
    ])
    
    # Render each tab
    with tab1:
        handle_data_tab()
    
    with tab2:
        handle_features_tab()
    
    with tab3:
        handle_forecasting_tab()
    
    with tab4:
        handle_clustering_tab()
    
    with tab5:
        handle_summary_tab()



def render_sidebar():
    """Render sidebar configuration panel"""
    
    st.header("⚙️ Analysis Configuration")
    
    # Feature Extraction Settings
    with st.expander("🔬 Feature Extraction", expanded=True):
        st.session_state.feature_complexity = st.selectbox(
            "Complexity Level",
            options=["Basic", "Standard", "Advanced"],
            index=1
        )
        
        st.session_state.window_size = st.slider(
            "Window Size (minutes)",
            min_value=15,
            max_value=120,
            value=60,
            step=15
        )
        
        st.session_state.window_overlap = st.slider(
            "Window Overlap (%)",
            min_value=0,
            max_value=75,
            value=50,
            step=25
        )
    
    # Forecasting Settings
    with st.expander("📈 Forecasting", expanded=True):
        st.session_state.forecast_periods = st.slider(
            "Forecast Horizon (periods)",
            min_value=50,
            max_value=300,
            value=120,
            step=10
        )
        
        st.session_state.confidence_interval = st.slider(
            "Confidence Interval",
            min_value=0.80,
            max_value=0.99,
            value=0.95,
            step=0.01,
            format="%.2f"
        )
        
        st.session_state.detect_anomalies = st.checkbox(
            "Enable Anomaly Detection",
            value=True
        )
    
    # Clustering Settings
    with st.expander("🎯 Clustering", expanded=True):
        st.session_state.cluster_method = st.selectbox(
            "Clustering Algorithm",
            options=["kmeans", "dbscan", "gmm", "hierarchical"],
            index=0
        )
        
        if st.session_state.cluster_method in ['kmeans', 'gmm', 'hierarchical']:
            st.session_state.n_clusters = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=8,
                value=3
            )
        else:
            st.session_state.n_clusters = 3
        
        st.session_state.scaling_method = st.selectbox(
            "Feature Scaling",
            options=["standard", "minmax", "robust"],
            index=0
        )
        
        st.session_state.auto_tune = st.checkbox(
            "Auto-tune Parameters",
            value=True
        )
    
    # Analysis Status
    st.divider()
    
    if st.session_state.data:
        st.success("✅ Data Loaded")
        
        if 'features' in st.session_state.results:
            st.success("✅ Features Extracted")
        
        if 'forecasts' in st.session_state.results:
            st.success("✅ Forecasts Generated")
        
        if 'clusters' in st.session_state.results:
            st.success("✅ Patterns Clustered")
    else:
        st.info("📥 Load or generate data to begin")



def handle_data_tab():
    """Handle data loading and generation tab - FIXED VERSION"""
    
    st.header("📊 Health Data Management")
    
    utils.show_info_message(
        "📥 Upload your preprocessed health data or generate sample data for testing.",
        type="info"
    )
    
    col1, col2 = st.columns([2, 1])
    
    # Generate sample data
    with col1:
        st.subheader("🔄 Generate Sample Dataset")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            days = st.slider("Days of data", 1, 14, 7)
            user_profile = st.selectbox(
                "User Profile",
                ["sedentary", "normal", "active", "athlete"],
                index=1
            )
        
        with col_b:
            noise_level = st.slider("Noise Level", 0.0, 0.3, 0.1, 0.05)
            anomaly_rate = st.slider("Anomaly Rate", 0.0, 0.1, 0.02, 0.01)
        
        if st.button("🚀 Generate Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive dataset..."):
                try:
                    # Generate data
                    generated_data = generate_comprehensive_dataset(
                        days=days,
                        user_profile=user_profile,
                        noise_level=noise_level,
                        anomaly_rate=anomaly_rate
                    )
                    
                    # Store in session state
                    st.session_state.data = generated_data
                    st.session_state.data_loaded = True
                    
                    # Show success message
                    st.success(f"✅ Generated {days}-day dataset with {user_profile} profile!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating data: {str(e)}")
    
    # Upload custom data
    with col2:
        st.subheader("📤 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            key='file_uploader'
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                filename = uploaded_file.name.lower()
                
                st.info(f"📄 File: {uploaded_file.name} ({len(df):,} rows)")
                
                # Auto-detect metric type
                metric_type = detect_metric_type(filename, df)
                
                if metric_type:
                    # Process and store data
                    processed_df = process_uploaded_data(df, metric_type)
                    st.session_state.data = {metric_type: processed_df}
                    st.session_state.data_loaded = True
                    
                    emoji_map = {
                        'heart_rate': '❤️',
                        'steps': '👟',
                        'activity': '⚡',
                        'sleep': '😴'
                    }
                    
                    st.success(
                        f"{emoji_map.get(metric_type, '📊')} "
                        f"{metric_type.replace('_', ' ').title()} data loaded! "
                        f"({len(processed_df):,} records)"
                    )
                else:
                    st.warning("⚠️ Could not auto-detect data type")
                    
                    manual_type = st.selectbox(
                        "Select data type:",
                        ["heart_rate", "steps", "activity", "sleep"],
                        key='manual_type_select'
                    )
                    
                    if st.button("📥 Load Data", key='manual_load_btn'):
                        processed_df = process_uploaded_data(df, manual_type)
                        st.session_state.data = {manual_type: processed_df}
                        st.session_state.data_loaded = True
                        st.success(f"✅ Loaded as {manual_type}!")
                        
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Display loaded data
    if st.session_state.data:
        st.divider()
        utils.display_data_overview(st.session_state.data)



def detect_metric_type(filename, df):
    """Auto-detect metric type from filename or columns"""
    
    filename_lower = filename.lower()
    columns_str = ' '.join(df.columns).lower()
    
    # Check filename
    if 'heart' in filename_lower or 'hr' in filename_lower:
        return 'heart_rate'
    elif 'step' in filename_lower:
        return 'steps'
    elif 'activity' in filename_lower or 'active' in filename_lower:
        return 'activity'
    elif 'sleep' in filename_lower:
        return 'sleep'
    
    # Check columns
    if 'heart' in columns_str:
        return 'heart_rate'
    elif 'step' in columns_str:
        return 'steps'
    elif 'activity' in columns_str:
        return 'activity'
    elif 'sleep' in columns_str:
        return 'sleep'
    
    return None



def process_uploaded_data(df, metric_type):
    """Process uploaded data into standard format"""
    
    value_col_map = {
        'heart_rate': 'heart_rate',
        'steps': 'step_count',
        'activity': 'activity_level',
        'sleep': 'sleep_quality'
    }
    
    value_col_name = value_col_map[metric_type]
    
    # Assign proper column names if needed
    if len(df.columns) == 2:
        df.columns = ['timestamp', value_col_name]
    elif 'timestamp' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'timestamp'})
    
    # Ensure value column exists
    if value_col_name not in df.columns and len(df.columns) >= 2:
        df = df.rename(columns={df.columns[1]: value_col_name})
    
    # Clean dataframe
    clean_df = df[['timestamp', value_col_name]].copy()
    clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
    clean_df = clean_df.sort_values('timestamp').reset_index(drop=True)
    
    return clean_df



def handle_features_tab():
    """Handle feature extraction tab"""
    
    st.header("🔬 Advanced Feature Extraction with TSFresh")
    
    if not st.session_state.data:
        utils.show_info_message(
            "👈 Please load data from the Data Management tab first.",
            type="info"
        )
        return
    
    # Display current configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        utils.create_metric_card(
            "Complexity",
            st.session_state.feature_complexity,
            icon="🎚️"
        )
    
    with col2:
        utils.create_metric_card(
            "Window Size",
            f"{st.session_state.window_size} min",
            icon="⏱️"
        )
    
    with col3:
        utils.create_metric_card(
            "Overlap",
            f"{st.session_state.window_overlap}%",
            icon="🔄"
        )
    
    # Metric selection
    st.subheader("📊 Select Metric")
    metric = st.selectbox(
        "Choose metric for feature extraction",
        list(st.session_state.data.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    # Feature extraction button
    if st.button("▶️ Extract Features", type="primary", use_container_width=True):
        
        with st.spinner("🔬 Extracting statistical features..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preparing data...")
                progress_bar.progress(20)
                
                status_text.text("Extracting features with TSFresh...")
                progress_bar.progress(40)
                
                # Extract features
                results = st.session_state.feature_engine.extract_all_features(
                    {metric: st.session_state.data[metric]},
                    window_size=st.session_state.window_size,
                    complexity=st.session_state.feature_complexity,
                    overlap_percentage=st.session_state.window_overlap
                )
                
                progress_bar.progress(80)
                status_text.text("Processing results...")
                
                st.session_state.results['features'] = results
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                st.success(f"✅ Extracted {len(results[metric].columns)} features from {len(results[metric])} windows!")
                
            except Exception as e:
                st.error(f"❌ Feature extraction failed: {str(e)}")
                return
    
    # Display results if available
    if 'features' in st.session_state.results:
        st.divider()
        utils.visualize_feature_results(st.session_state.results['features'])



def handle_forecasting_tab():
    """Handle time series forecasting tab"""
    
    st.header("📈 Time Series Forecasting with Prophet")
    
    if not st.session_state.data:
        utils.show_info_message(
            "👈 Please load data from the Data Management tab first.",
            type="info"
        )
        return
    
    # Display configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        utils.create_metric_card(
            "Forecast Horizon",
            f"{st.session_state.forecast_periods} periods",
            icon="🔮"
        )
    
    with col2:
        utils.create_metric_card(
            "Confidence Interval",
            f"{st.session_state.confidence_interval*100:.0f}%",
            icon="📊"
        )
    
    with col3:
        anomaly_status = "Enabled" if st.session_state.detect_anomalies else "Disabled"
        utils.create_metric_card(
            "Anomaly Detection",
            anomaly_status,
            icon="🚨"
        )
    
    # Metric selection
    st.subheader("📊 Select Metric")
    metric = st.selectbox(
        "Choose metric for forecasting",
        list(st.session_state.data.keys()),
        format_func=lambda x: x.replace('_', ' ').title(),
        key='forecast_metric_select'
    )
    
    # Forecast button
    if st.button("▶️ Generate Forecast", type="primary", use_container_width=True):
        
        with st.spinner("📈 Training Prophet model and generating forecast..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preparing data for Prophet...")
                progress_bar.progress(15)
                
                status_text.text("Training Prophet model...")
                progress_bar.progress(30)
                
                # Generate forecast
                forecasts = st.session_state.forecast_engine.forecast_all_metrics(
                    {metric: st.session_state.data[metric]},
                    forecast_periods=st.session_state.forecast_periods,
                    confidence_interval=st.session_state.confidence_interval,
                    detect_anomalies=st.session_state.detect_anomalies
                )
                
                progress_bar.progress(70)
                status_text.text("Detecting anomalies...")
                progress_bar.progress(90)
                
                st.session_state.results['forecasts'] = forecasts
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                mape = forecasts[metric]['metrics']['mape']
                n_anomalies = len(forecasts[metric]['anomalies'])
                
                st.success(f"✅ Forecast generated! MAPE: {mape:.2f}%, Anomalies: {n_anomalies}")
                
            except Exception as e:
                st.error(f"❌ Forecasting failed: {str(e)}")
                return
    
    # Display results if available
    if 'forecasts' in st.session_state.results:
        st.divider()
        utils.visualize_forecast_results(
            st.session_state.results['forecasts'],
            st.session_state.data
        )



def handle_clustering_tab():
    """Handle pattern clustering tab"""
    
    st.header("🎯 Behavioral Pattern Clustering")
    
    if 'features' not in st.session_state.results:
        utils.show_info_message(
            "👈 Please extract features first from the Feature Extraction tab.",
            type="info"
        )
        return
    
    # Display configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        utils.create_metric_card(
            "Algorithm",
            st.session_state.cluster_method.UPPER(),
            icon="🤖"
        )
    
    with col2:
        utils.create_metric_card(
            "Target Clusters",
            str(st.session_state.n_clusters),
            icon="🎯"
        )
    
    with col3:
        utils.create_metric_card(
            "Scaling Method",
            st.session_state.scaling_method.title(),
            icon="⚖️"
        )
    
    # Info about standardization
    utils.show_info_message(
        "🔧 **Feature Standardization:** Features are automatically scaled to ensure optimal clustering performance.",
        type="info"
    )
    
    # Clustering button
    if st.button("▶️ Discover Patterns", type="primary", use_container_width=True):
        
        with st.spinner(f"🎯 Applying {st.session_state.cluster_method.upper()} clustering..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Preprocessing features...")
                progress_bar.progress(20)
                
                status_text.text("Applying clustering algorithm...")
                progress_bar.progress(40)
                
                # Apply clustering
                clusters = st.session_state.pattern_analyzer.analyze_patterns(
                    st.session_state.results['features'],
                    method=st.session_state.cluster_method,
                    n_clusters=st.session_state.n_clusters,
                    scaling_method=st.session_state.scaling_method,
                    auto_tune=st.session_state.auto_tune
                )
                
                progress_bar.progress(70)
                status_text.text("Calculating quality metrics...")
                progress_bar.progress(90)
                
                st.session_state.results['clusters'] = clusters
                
                progress_bar.progress(100)
                status_text.text("✅ Complete!")
                
                # Display results
                first_metric = list(clusters.keys())[0]
                silhouette = clusters[first_metric]['quality_metrics']['silhouette_score']
                n_clusters_found = clusters[first_metric]['quality_metrics']['n_clusters']
                
                if silhouette > 0.7:
                    st.success(f"✅ Excellent clustering! Silhouette Score: {silhouette:.3f}, Clusters: {n_clusters_found}")
                elif silhouette > 0.5:
                    st.success(f"✅ Good clustering quality. Silhouette Score: {silhouette:.3f}, Clusters: {n_clusters_found}")
                else:
                    st.warning(f"⚠️ Fair clustering quality. Silhouette Score: {silhouette:.3f}. Consider adjusting parameters.")
                
            except Exception as e:
                st.error(f"❌ Clustering failed: {str(e)}")
                import traceback
                with st.expander("🔍 Error Details"):
                    st.code(traceback.format_exc())
                return
    
    # Display results if available
    if 'clusters' in st.session_state.results:
        st.divider()
        utils.visualize_cluster_results(st.session_state.results['clusters'])



def handle_summary_tab():
    """Handle comprehensive summary report tab - ENHANCED VERSION"""
    
    st.header("📋 Comprehensive Analysis Report")
    
    # Always show progress tracker
    st.markdown("### ✅ Analysis Progress")
    
    checklist = {
        "Load Data": st.session_state.data is not None,
        "Extract Features": 'features' in st.session_state.results,
        "Generate Forecast": 'forecasts' in st.session_state.results,
        "Cluster Patterns": 'clusters' in st.session_state.results
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    for i, (task, completed) in enumerate(checklist.items()):
        col = [col1, col2, col3, col4][i]
        with col:
            if completed:
                st.success(f"✅ {task}")
            else:
                st.info(f"⬜ {task}")
    
    st.divider()
    
    # Show report if data exists
    if not st.session_state.results:
        utils.show_info_message(
            "⚠️ No analysis results yet. Complete the analysis steps above to generate a comprehensive report.",
            type="info"
        )
        return
    
    # Generate and display summary
    utils.generate_summary_report(st.session_state.results)



if __name__ == "__main__":
    main()
