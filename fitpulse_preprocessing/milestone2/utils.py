"""
Utility Functions
Visualization helpers and data processing utilities
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional


def apply_custom_styling():
    """Apply custom CSS styling - LIGHT DROPDOWN with DARK TEXT"""
    
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Metrics styling - DARK TEXT */
        div[data-testid="stMetricValue"] {
            font-size: 32px;
            font-weight: 700;
            color: #1a1a1a !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            font-weight: 600;
            color: #2c3e50 !important;
        }
        
        div[data-testid="stMetricDelta"] {
            font-size: 14px;
            color: #34495e !important;
        }
        
        /* Headers - ALL DARK */
        h1 {
            color: #1a1a1a !important;
            font-weight: 700;
            padding-bottom: 10px;
            border-bottom: 3px solid #3498db;
        }
        
        h2 {
            color: #2c3e50 !important;
            font-weight: 600;
            margin-top: 20px;
        }
        
        h3 {
            color: #34495e !important;
            font-weight: 500;
        }
        
        /* Regular text - DARK */
        p, div, span {
            color: #2c3e50 !important;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background-color: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #ecf0f1;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            color: #2c3e50 !important;
            border: 2px solid transparent;
            transition: all 0.3s;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #d5dbdb;
            border-color: #3498db;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3498db !important;
            color: white !important;
        }
        
        /* Buttons - WHITE TEXT */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            transition: transform 0.2s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Expanders - DARK TEXT */
        .streamlit-expanderHeader {
            background-color: #ecf0f1;
            border-radius: 8px;
            font-weight: 600;
            color: #1a1a1a !important;
        }
        
        /* Dataframes */
        .dataframe {
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Selectbox label - DARK TEXT */
        .stSelectbox label {
            color: #2c3e50 !important;
            font-weight: 600;
        }
        
        /* FIXED: Selectbox dropdown - LIGHT BACKGROUND + DARK TEXT */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #ecf0f1 !important;
            color: #1a1a1a !important;
            border-radius: 8px;
            border: 2px solid #bdc3c7;
            font-weight: 600;
        }
        
        /* Dropdown selected text - DARK */
        .stSelectbox div[data-baseweb="select"] span {
            color: #1a1a1a !important;
        }
        
        /* Dropdown icon - DARK */
        .stSelectbox svg {
            fill: #2c3e50 !important;
        }
        
        /* Dropdown hover effect */
        .stSelectbox div[data-baseweb="select"]:hover > div {
            background-color: #d5dbdb !important;
            border-color: #3498db;
        }
        
        /* Text input labels - DARK */
        .stTextInput label {
            color: #2c3e50 !important;
        }
        
        /* Markdown text - DARK */
        .stMarkdown {
            color: #2c3e50 !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #2c3e50;
        }
        
        section[data-testid="stSidebar"] label {
            color: white !important;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: white !important;
        }
        
        /* Progress bars */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Info boxes - DARK TEXT */
        .stAlert {
            border-radius: 8px;
            border-left: 4px solid #3498db;
            color: #1a1a1a !important;
        }
    </style>
    """, unsafe_allow_html=True)


def display_data_overview(data_dict: Dict[str, pd.DataFrame]):
    """Display comprehensive data overview with visualizations"""
    
    st.subheader("📊 Dataset Overview")
    
    # Summary metrics
    metric_cols = st.columns(len(data_dict))
    
    for idx, (metric_name, df) in enumerate(data_dict.items()):
        with metric_cols[idx]:
            value_col = df.columns[1]
            
            st.metric(
                label=metric_name.replace('_', ' ').title(),
                value=f"{len(df):,}",
                delta=f"{df.shape[1]} cols"
            )
    
    # Time series preview
    st.subheader("Time Series Preview")
    
    selected_metric = st.selectbox(
        "Select metric to visualize",
        options=list(data_dict.keys()),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    if selected_metric:
        df = data_dict[selected_metric]
        value_col = df.columns[1]
        
        # Plot time series
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[value_col],
            mode='lines',
            name=value_col.replace('_', ' ').title(),
            line=dict(color='#3498db', width=1.5)
        ))
        
        fig.update_layout(
            title=dict(
                text=f"{selected_metric.replace('_', ' ').title()} Over Time",
                font=dict(size=18, color='#1a1a1a')
            ),
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            height=400,
            template='plotly_white',
            font=dict(color='#2c3e50')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical summary
        st.subheader("Statistical Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("Mean", f"{df[value_col].mean():.2f}")
        col2.metric("Median", f"{df[value_col].median():.2f}")
        col3.metric("Std Dev", f"{df[value_col].std():.2f}")
        col4.metric("Min", f"{df[value_col].min():.2f}")
        col5.metric("Max", f"{df[value_col].max():.2f}")


def visualize_feature_results(features_dict: Dict):
    """Visualize feature extraction results"""
    
    st.subheader("🔬 Feature Extraction Results")
    
    for metric_name, feature_data in features_dict.items():
        if 'features' not in feature_data:
            continue
        
        with st.expander(f"📊 {metric_name.replace('_', ' ').title()} Features", expanded=True):
            
            feature_matrix = feature_data['features']
            report = feature_data.get('report', {})
            top_features = feature_data.get('top_features', pd.DataFrame())
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Features Extracted", report.get('features_count', 0))
            col2.metric("Windows Analyzed", report.get('window_count', 0))
            col3.metric("Extraction Time", f"{report.get('extraction_time_sec', 0):.2f}s")
            
            # Top features table
            if not top_features.empty:
                st.write("**Top Features by Variance:**")
                
                # Create a shortened version for display
                display_df = top_features.copy()
                if 'Feature' in display_df.columns:
                    display_df['Feature_Short'] = display_df['Feature'].apply(
                        lambda x: x.split('__')[1] if '__' in str(x) else str(x)[:30]
                    )
                
                st.dataframe(
                    display_df[['Feature_Short', 'Variance', 'Mean', 'Std']].head(10) 
                    if 'Feature_Short' in display_df.columns 
                    else top_features.head(10),
                    use_container_width=True
                )
            
            # Feature distributions
            if not feature_matrix.empty:
                visualize_feature_distributions_grid(feature_matrix, metric_name)


def visualize_feature_distributions_grid(feature_matrix: pd.DataFrame, 
                                         metric_name: str, n_features: int = 6):
    """Create grid of feature distribution plots"""
    
    # Get top features by variance
    feature_vars = feature_matrix.var().sort_values(ascending=False).head(n_features)
    
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[
            f.split('__')[1] if '__' in f else f[:20] 
            for f in feature_vars.index
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    colors = px.colors.qualitative.Set2
    
    for idx, feature in enumerate(feature_vars.index):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Histogram(
                x=feature_matrix[feature],
                name=feature,
                showlegend=False,
                marker=dict(
                    color=colors[idx % len(colors)],
                    opacity=0.7,
                    line=dict(color='white', width=1)
                ),
                nbinsx=30
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title=dict(
            text=f"Feature Distributions: {metric_name.title()}",
            font=dict(size=18, color='#1a1a1a')
        ),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white',
        font=dict(color='#2c3e50')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_forecast_results(forecasts_dict: Dict, original_data: Dict):
    """Visualize Prophet forecast results"""
    
    st.subheader("📈 Forecasting Results")
    
    for metric_name, forecast_data in forecasts_dict.items():
        if 'forecast' not in forecast_data:
            continue
        
        with st.expander(f"📈 {metric_name.replace('_', ' ').title()} Forecast", expanded=True):
            
            forecast_df = forecast_data['forecast']
            metrics = forecast_data.get('metrics', {})
            anomalies = forecast_data.get('anomalies', pd.DataFrame())
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("MAE", f"{metrics.get('mae', 0):.2f}")
            col2.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
            col3.metric("MAPE", f"{metrics.get('mape', 0):.1f}%")
            col4.metric("R²", f"{metrics.get('r_squared', 0):.3f}")
            
            # Forecast plot
            plot_prophet_forecast(forecast_df, original_data.get(metric_name), metric_name)
            
            # Anomalies
            if not anomalies.empty:
                st.write(f"**🚨 Detected {len(anomalies)} Anomalies**")
                st.dataframe(anomalies.head(10), use_container_width=True)


def plot_prophet_forecast(forecast: pd.DataFrame, actual_data: Optional[pd.DataFrame],
                          metric_name: str):
    """Plot Prophet forecast with confidence intervals"""
    
    fig = go.Figure()
    
    # Actual data
    if actual_data is not None:
        value_col = actual_data.columns[1]
        fig.add_trace(go.Scatter(
            x=actual_data['timestamp'],
            y=actual_data[value_col],
            mode='markers',
            name='Actual',
            marker=dict(size=4, color='#2ecc71', opacity=0.6)
        ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='#e74c3c', width=2)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper CI',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        name='90% CI',
        fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Prophet Forecast: {metric_name.title()}",
            font=dict(size=18, color='#1a1a1a')
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        font=dict(color='#2c3e50')
    )
    
    st.plotly_chart(fig, use_container_width=True)


def visualize_cluster_results(clusters_dict: Dict):
    """Visualize clustering results"""
    
    st.subheader("🎯 Clustering Results")
    
    for metric_name, cluster_data in clusters_dict.items():
        if 'labels' not in cluster_data:
            continue
        
        with st.expander(f"🎯 {metric_name.replace('_', ' ').title()} Patterns", expanded=True):
            
            labels = cluster_data['labels']
            report = cluster_data.get('report', {})
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Patterns Found", report.get('n_clusters_found', 0))
            
            if 'silhouette_score' in report:
                col2.metric("Silhouette Score", f"{report['silhouette_score']:.3f}")
            
            if 'davies_bouldin_index' in report:
                col3.metric("Davies-Bouldin", f"{report['davies_bouldin_index']:.3f}")
            
            # Cluster distribution
            plot_cluster_distribution(labels, metric_name)


def plot_cluster_distribution(labels: np.ndarray, metric_name: str):
    """Plot cluster size distribution"""
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    cluster_df = pd.DataFrame({
        'Pattern ID': [f"Pattern {l}" if l >= 0 else "Noise" for l in unique_labels],
        'Count': counts,
        'Percentage': (counts / len(labels) * 100).round(1)
    })
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(cluster_df, use_container_width=True)
    
    with col2:
        fig = px.pie(
            cluster_df,
            values='Count',
            names='Pattern ID',
            title="Pattern Distribution",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            height=300,
            font=dict(color='#1a1a1a'),
            title_font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, use_container_width=True)


def generate_summary_report(results: Dict):
    """Generate comprehensive summary report"""
    
    st.subheader("📋 Comprehensive Analysis Summary")
    
    # Overall progress
    st.success("✅ Milestone 2 Analysis Complete!")
    
    # Summary table
    summary_data = []
    
    for key in results.keys():
        if '_features' in str(key):
            metric = key.split('_features')[0] if isinstance(key, str) else key
            
            feature_status = '✅' if f'{metric}_features' in results else '❌'
            forecast_status = '✅' if f'{metric}_forecasts' in results else '❌'
            cluster_status = '✅' if f'{metric}_clusters' in results else '❌'
            
            summary_data.append({
                'Metric': metric.replace('_', ' ').title(),
                'Features': feature_status,
                'Forecast': forecast_status,
                'Clustering': cluster_status
            })
    
    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    st.info("""
    **Milestone 2 Deliverables:**
    - ✅ TSFresh statistical feature extraction
    - ✅ Prophet time series forecasting with confidence intervals
    - ✅ Behavioral pattern clustering (K-Means/DBSCAN)
    - ✅ Anomaly detection via residual analysis
    - ✅ PCA/t-SNE dimensionality reduction visualizations
    - ✅ Comprehensive performance metrics and reports
    """)
