"""
FitPulse Analytics Pro - Milestone 2
Main Application Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from milestone2.feature_engineering import AdvancedFeatureEngine
from milestone2.time_series_analysis import TrendForecastingEngine
from milestone2.pattern_clustering import BehaviorPatternAnalyzer
from milestone2.data_generator import generate_comprehensive_dataset
from milestone2 import utils


def initialize_session():
    """Initialize session state variables"""
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


def main():
    """Main application function"""
    
    # Page config
    st.set_page_config(
        page_title="Milestone 2 - Feature & Pattern Analysis",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply styling
    utils.apply_custom_styling()
    
    # Initialize session
    initialize_session()
    
    # Header
    st.title("🧬 Milestone 2: Advanced Feature Extraction & Modeling")
    st.markdown("**TSFresh Feature Engineering • Prophet Forecasting • Pattern Clustering**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Feature Extraction")
        feature_complexity = st.selectbox(
            "Complexity Level",
            options=["Basic", "Standard", "Advanced"],
            index=1
        )
        window_size = st.slider("Window Size (min)", 15, 120, 45, 15)
        
        st.subheader("Forecasting")
        forecast_periods = st.slider("Forecast Horizon", 50, 300, 120, 10)
        
        st.subheader("Clustering")
        cluster_method = st.selectbox("Method", ["kmeans", "dbscan"])
        if cluster_method == "kmeans":
            n_clusters = st.slider("Patterns", 2, 8, 4)
        else:
            n_clusters = 4
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data", "🔬 Features", "📈 Forecasting", "🎯 Clustering", "📋 Summary"
    ])
    
    # Tab implementations
        # Tab 1: Data
        # Tab 1: Data
    with tab1:
        st.header("Health Data Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🔄 Generate Sample Dataset", type="primary", use_container_width=True):
                with st.spinner("Generating data..."):
                    st.session_state.data = generate_comprehensive_dataset(days=7)
        
        with col2:
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                import pandas as pd
                
                # Read uploaded CSV
                df = pd.read_csv(uploaded)
                
                filename = uploaded.name.lower()
                
                st.info(f"📄 Uploaded: {uploaded.name}")
                st.write(f"**Original columns:** {list(df.columns)}")
                st.write(f"**Rows:** {len(df):,}")
                
                # Check if CSV has no proper headers (headerless CSV)
                has_no_headers = (
                    df.columns[0] == 0 or 
                    'Unnamed' in str(df.columns[0]) or 
                    df.columns[0].isdigit() if isinstance(df.columns[0], str) else False
                )
                
                # ====================================================================
                # SMART DETECTION: Based on filename OR column analysis
                # ====================================================================
                
                metric_type = None
                value_column_name = None
                
                # Method 1: Detect from FILENAME
                if 'heart' in filename or 'hr' in filename:
                    metric_type = 'heart_rate'
                    value_column_name = 'heart_rate'
                    st.info("🔍 Detected: **Heart Rate** data (from filename)")
                
                elif 'step' in filename:
                    metric_type = 'steps'
                    value_column_name = 'step_count'
                    st.info("🔍 Detected: **Steps** data (from filename)")
                
                elif 'activity' in filename or 'active' in filename:
                    metric_type = 'activity'
                    value_column_name = 'activity_level'
                    st.info("🔍 Detected: **Activity** data (from filename)")
                
                # Method 2: Detect from COLUMN NAMES (fallback)
                elif not has_no_headers:
                    if 'heart_rate' in df.columns or 'heart' in str(df.columns).lower():
                        metric_type = 'heart_rate'
                        value_column_name = 'heart_rate'
                        st.info("🔍 Detected: **Heart Rate** data (from columns)")
                    
                    elif 'step_count' in df.columns or 'steps' in df.columns:
                        metric_type = 'steps'
                        value_column_name = 'step_count'
                        st.info("🔍 Detected: **Steps** data (from columns)")
                    
                    elif 'activity_level' in df.columns or 'activity' in df.columns:
                        metric_type = 'activity'
                        value_column_name = 'activity_level'
                        st.info("🔍 Detected: **Activity** data (from columns)")
                
                # ====================================================================
                # PROCESS THE DETECTED DATA
                # ====================================================================
                
                if metric_type:
                    # Assign proper headers if needed
                    if has_no_headers or len(df.columns) == 2:
                        df.columns = ['timestamp', value_column_name]
                        st.success(f"✅ Assigned column names: ['timestamp', '{value_column_name}']")
                    
                    # Ensure we have the right columns
                    if 'timestamp' not in df.columns:
                        # Rename first column to timestamp
                        df = df.rename(columns={df.columns[0]: 'timestamp'})
                    
                    # Check if value column exists, if not use second column
                    if value_column_name not in df.columns:
                        if len(df.columns) >= 2:
                            # Use second column as value column
                            df = df.rename(columns={df.columns[1]: value_column_name})
                    
                    # Create clean dataframe with only needed columns
                    try:
                        clean_df = df[['timestamp', value_column_name]].copy()
                        
                        # Convert timestamp to datetime
                        clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
                        
                        # Store in session state
                        st.session_state.data = {metric_type: clean_df}
                        
                        # Success message
                        emoji_map = {
                            'heart_rate': '❤️',
                            'steps': '👟',
                            'activity': '⚡'
                        }
                        emoji = emoji_map.get(metric_type, '📊')
                        
                        st.success(
                            f"{emoji} **{metric_type.replace('_', ' ').title()} data loaded successfully!** "
                            f"({len(clean_df):,} records)"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error processing data: {str(e)}")
                        st.write("DataFrame info:")
                        st.write(df.head())
                
                else:
                    # Could not detect - show manual options
                    st.warning("⚠️ Could not auto-detect data type from filename or columns")
                    
                    # Manual selection
                    manual_type = st.selectbox(
                        "Please select the data type:",
                        options=["heart_rate", "steps", "activity"]
                    )
                    
                    if st.button("📥 Load as Selected Type"):
                        value_col_map = {
                            'heart_rate': 'heart_rate',
                            'steps': 'step_count',
                            'activity': 'activity_level'
                        }
                        
                        value_col_name = value_col_map[manual_type]
                        
                        # Assign columns
                        if len(df.columns) == 2:
                            df.columns = ['timestamp', value_col_name]
                        
                        clean_df = df[['timestamp', value_col_name]].copy()
                        clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
                        
                        st.session_state.data = {manual_type: clean_df}
                        st.success(f"✅ Loaded as {manual_type.replace('_', ' ').title()} data!")
                        st.rerun()
        
        # ====================================================================
        # DISPLAY LOADED DATA
        # ====================================================================
        
        if st.session_state.data:
            st.subheader("📊 Dataset Overview")
            
            for metric, df in st.session_state.data.items():
                # Create columns for display
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    emoji_map = {
                        'heart_rate': '❤️',
                        'steps': '👟',
                        'activity': '⚡'
                    }
                    emoji = emoji_map.get(metric, '📊')
                    st.metric(
                        label=f"{emoji} {metric.replace('_', ' ').title()}",
                        value=f"{len(df):,} records",
                        delta=f"{df.shape[1]} columns"
                    )
                
                with col2:
                    # Show date range
                    if 'timestamp' in df.columns:
                        min_date = df['timestamp'].min()
                        max_date = df['timestamp'].max()
                        duration = (max_date - min_date).days
                        st.metric("Duration", f"{duration} days", delta=f"{(max_date - min_date).total_seconds()/3600:.0f} hours")
                
                with col3:
                    # Show value statistics
                    value_col = df.columns[1]
                    st.metric(
                        "Value Range",
                        f"{df[value_col].min():.1f} - {df[value_col].max():.1f}",
                        delta=f"Avg: {df[value_col].mean():.1f}"
                    )
                
                # Preview data
                with st.expander(f"👁️ Preview {metric.replace('_', ' ').title()} Data"):
                    st.dataframe(df.head(15), use_container_width=True)
                    st.write(f"**Columns:** {list(df.columns)}")
                    st.write(f"**Data types:** {dict(df.dtypes)}")


    
    with tab2:
        handle_features_tab(feature_complexity, window_size)
    
    with tab3:
        handle_forecasting_tab(forecast_periods)
    
    with tab4:
        handle_clustering_tab(cluster_method, n_clusters)
    
    with tab5:
        handle_summary_tab()


def handle_data_tab():
    """Handle data generation/upload tab"""
    st.header("Health Data Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔄 Generate Sample Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating data..."):
                st.session_state.data = generate_comprehensive_dataset(days=7)
                st.success("✅ Dataset generated!")
    
    with col2:
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded:
            import pandas as pd
            st.session_state.data = {'heart_rate': pd.read_csv(uploaded)}
    
    if st.session_state.data:
        utils.display_data_overview(st.session_state.data)


def handle_features_tab(complexity, window_size):
    """Handle feature extraction tab"""
    st.header("🔬 Feature Extraction")
    
    if not st.session_state.data:
        st.info("👆 Load data first")
        return
    
    metric = st.selectbox("Select Metric", list(st.session_state.data.keys()))
    
    if st.button("▶️ Extract Features", type="primary"):
        st.session_state.feature_engine.extraction_mode = complexity
        
        with st.spinner("Extracting features..."):
            results = st.session_state.feature_engine.extract_all_features(
                {metric: st.session_state.data[metric]},
                window_size=window_size,
                complexity=complexity
            )
            st.session_state.results['features'] = results
    
    if 'features' in st.session_state.results:
        utils.visualize_feature_results(st.session_state.results['features'])


def handle_forecasting_tab(forecast_periods):
    """Handle forecasting tab"""
    st.header("📈 Time Series Forecasting")
    
    if not st.session_state.data:
        st.info("👆 Load data first")
        return
    
    metric = st.selectbox("Select Metric", list(st.session_state.data.keys()), key='forecast_metric')
    
    if st.button("▶️ Generate Forecast", type="primary"):
        with st.spinner("Training Prophet model..."):
            forecasts = st.session_state.forecast_engine.forecast_all_metrics(
                {metric: st.session_state.data[metric]},
                forecast_periods=forecast_periods
            )
            st.session_state.results['forecasts'] = forecasts
    
    if 'forecasts' in st.session_state.results:
        utils.visualize_forecast_results(
            st.session_state.results['forecasts'],
            st.session_state.data
        )


def handle_clustering_tab(method, n_clusters):
    """Handle clustering tab"""
    st.header("🎯 Pattern Clustering")
    
    if 'features' not in st.session_state.results:
        st.info("👆 Extract features first")
        return
    
    if st.button("▶️ Discover Patterns", type="primary"):
        with st.spinner(f"Applying {method} clustering..."):
            clusters = st.session_state.pattern_analyzer.analyze_patterns(
                st.session_state.results['features'],
                method=method,
                n_clusters=n_clusters
            )
            st.session_state.results['clusters'] = clusters
    
    if 'clusters' in st.session_state.results:
        utils.visualize_cluster_results(st.session_state.results['clusters'])


def handle_summary_tab():
    """Handle summary report tab"""
    st.header("📋 Milestone 2 Summary")
    
    if st.session_state.results:
        utils.generate_summary_report(st.session_state.results)
    else:
        st.info("Complete analysis steps to generate summary")


if __name__ == "__main__":
    main()
