"""
Utility Functions and Visualization Helpers
Comprehensive utilities for Milestone 2 analytics platform


Features:
- Custom Streamlit styling and themes
- Interactive Plotly visualizations
- Data export utilities
- Report generation
- Statistical analysis helpers
"""


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import io
import base64



def apply_custom_styling():
    """Apply custom CSS styling to Streamlit app"""
    
    st.markdown("""
        <style>
        /* Main color scheme */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #2ecc71;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --dark-bg: #1e1e2f;
            --light-bg: #f5f7fa;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        /* Metric cards - PURPLE WITH WHITE TEXT */
        .metric-card {
            background: linear-gradient(135deg, #7c3aed, #6d28d9);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(124, 58, 237, 0.3);
            border-left: 4px solid #5b21b6;
            margin-bottom: 1rem;
        }
        
        .metric-card h3 {
            color: white;
            margin-bottom: 0.5rem;
        }
        
        .metric-card h2 {
            color: white !important;
        }
        
        .metric-card p {
            color: rgba(255, 255, 255, 0.9);
        }
        
        /* Status badges */
        .status-excellent {
            background-color: var(--success-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .status-good {
            background-color: #3498db;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .status-fair {
            background-color: var(--warning-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        .status-poor {
            background-color: var(--danger-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-weight: bold;
            display: inline-block;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }
        
        /* Data tables */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, var(--dark-bg), #2d2d3f);
        }
        
        /* Info boxes - PURPLE WITH WHITE TEXT */
        .info-box {
            background-color: #7c3aed;
            border-left: 4px solid #5b21b6;
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
            color: white;
        }
        
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid var(--success-color);
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background-color: #fff3e0;
            border-left: 4px solid var(--warning-color);
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }
        
        /* Progress indicators */
        .progress-container {
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            height: 20px;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            transition: width 0.3s ease;
        }
        
        /* Chart containers */
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--light-bg);
            border-radius: 5px;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)



def display_data_overview(data_dict):
    """Display comprehensive overview of loaded data"""
    
    st.subheader("📊 Dataset Overview")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📈 Metrics", len(data_dict))
    
    total_records = sum(len(df) for df in data_dict.values())
    with col2:
        st.metric("📝 Total Records", f"{total_records:,}")
    
    if data_dict:
        first_df = list(data_dict.values())[0]
        duration = (first_df['timestamp'].max() - first_df['timestamp'].min())
        duration_hours = duration.total_seconds() / 3600
        
        with col3:
            st.metric("⏱️ Duration", f"{duration_hours:.1f} hours")
        
        with col4:
            st.metric("📅 Days", f"{duration.days}")
    
    # Detailed metric breakdown
    for metric_name, df in data_dict.items():
        with st.expander(f"👁️ {metric_name.replace('_', ' ').title()} Details"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                value_col = df.columns[1]
                st.write("**Statistics:**")
                st.write(f"- Count: {len(df):,}")
                st.write(f"- Mean: {df[value_col].mean():.2f}")
                st.write(f"- Std: {df[value_col].std():.2f}")
                st.write(f"- Min: {df[value_col].min():.2f}")
                st.write(f"- Max: {df[value_col].max():.2f}")
                
                # Quick visualization
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df[value_col],
                    name=metric_name,
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title="Distribution",
                    height=200,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)



def visualize_feature_results(feature_results):
    """Visualize feature extraction results"""
    
    st.subheader("🔬 Feature Extraction Results")
    
    for metric_name, features in feature_results.items():
        st.markdown(f"### {metric_name.replace('_', ' ').title()}")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Features Extracted", len(features.columns))
        
        with col2:
            st.metric("Windows Created", len(features))
        
        with col3:
            st.metric("Total Data Points", len(features) * len(features.columns))
        
        # Top features by variance
        st.markdown("#### 📊 Top Features by Variance")
        
        top_features = features.var().nlargest(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features.values,
            y=[name[:50] + '...' if len(name) > 50 else name for name in top_features.index],
            orientation='h',
            marker=dict(
                color=top_features.values,
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Feature Importance (by Variance)",
            xaxis_title="Variance",
            yaxis_title="Feature",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distribution
        with st.expander("📈 Feature Distribution Analysis"):
            selected_feature = st.selectbox(
                "Select feature to visualize",
                features.columns,
                key=f"feature_select_{metric_name}"
            )
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Distribution", "Time Series")
            )
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=features[selected_feature],
                    name="Distribution",
                    marker_color='#667eea'
                ),
                row=1, col=1
            )
            
            # Time series
            fig.add_trace(
                go.Scatter(
                    y=features[selected_feature],
                    mode='lines',
                    name="Values",
                    line=dict(color='#764ba2')
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)



def visualize_forecast_results(forecast_results, original_data):
    """Visualize Prophet forecasting results"""
    
    st.subheader("📈 Time Series Forecasting Results")
    
    for metric_name, forecast_data in forecast_results.items():
        st.markdown(f"### {metric_name.replace('_', ' ').title()}")
        
        # Performance metrics
        metrics = forecast_data['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mape = metrics['mape']
            status = 'excellent' if mape < 5 else 'good' if mape < 10 else 'fair'
            st.metric("MAPE", f"{mape:.2f}%", delta=f"{status}")
        
        with col2:
            st.metric("MAE", f"{metrics['mae']:.2f}")
        
        with col3:
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
        
        with col4:
            r2 = metrics['r2']
            st.metric("R²", f"{r2:.3f}", delta=f"{r2*100:.1f}% var explained")
        
        # Forecast visualization
        forecast = forecast_data['forecast']
        original = forecast_data['original_data']
        
        fig = go.Figure()
        
        # Actual data
        fig.add_trace(go.Scatter(
            x=original['ds'],
            y=original['y'],
            mode='markers',
            name='Actual',
            marker=dict(size=4, color='#2c3e50')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#667eea', width=2)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='95% CI',
            line=dict(width=0),
            fillcolor='rgba(102, 126, 234, 0.2)',
            fill='tonexty'
        ))
        
        fig.update_layout(
            title=f"{metric_name.replace('_', ' ').title()} Forecast with 95% Confidence Interval",
            xaxis_title="Time",
            yaxis_title="Value",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly detection results
        if forecast_data['anomalies']:
            with st.expander(f"🚨 Anomalies Detected ({len(forecast_data['anomalies'])})"):
                anomalies_df = pd.DataFrame(forecast_data['anomalies'])
                
                # Display top anomalies
                st.dataframe(
                    anomalies_df.head(10)[['timestamp', 'actual_value', 'predicted_value', 
                                           'anomaly_score', 'anomaly_type']],
                    use_container_width=True
                )
                
                # Anomaly visualization
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=anomalies_df['timestamp'],
                    y=anomalies_df['anomaly_score'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=anomalies_df['anomaly_score'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Score")
                    ),
                    text=anomalies_df['anomaly_type'],
                    hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<br>Time: %{x}'
                ))
                
                fig.update_layout(
                    title="Anomaly Scores Over Time",
                    xaxis_title="Time",
                    yaxis_title="Anomaly Score (σ)",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)



def visualize_cluster_results(cluster_results):
    """Visualize clustering analysis results"""
    
    st.subheader("🎯 Behavioral Pattern Clustering")
    
    for metric_name, cluster_data in cluster_results.items():
        st.markdown(f"### {metric_name.replace('_', ' ').title()}")
        
        # Quality metrics
        metrics = cluster_data['quality_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            silhouette = metrics['silhouette_score']
            if silhouette > 0.7:
                status = "Excellent"
                color = "success"
            elif silhouette > 0.5:
                status = "Good"
                color = "good"
            elif silhouette > 0.3:
                status = "Fair"
                color = "fair"
            else:
                status = "Poor"
                color = "poor"
            
            st.metric("Silhouette Score", f"{silhouette:.3f}")
            st.markdown(f'<span class="status-{color}">{status}</span>', unsafe_allow_html=True)
        
        with col2:
            st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
        
        with col3:
            st.metric("Clusters Found", metrics['n_clusters'])
        
        with col4:
            if metrics['n_noise_points'] > 0:
                st.metric("Noise Points", metrics['n_noise_points'])
        
        # Cluster visualization
        if 'pca' in cluster_data['visualization_data']:
            pca_data = cluster_data['visualization_data']['pca']
            labels = cluster_data['labels']
            
            fig = go.Figure()
            
            for label in np.unique(labels):
                mask = labels == label
                label_name = f"Cluster {label}" if label != -1 else "Noise"
                
                fig.add_trace(go.Scatter(
                    x=pca_data[mask, 0],
                    y=pca_data[mask, 1],
                    mode='markers',
                    name=label_name,
                    marker=dict(size=8),
                    text=[label_name] * np.sum(mask)
                ))
            
            variance_explained = cluster_data['visualization_data'].get('pca_variance_explained', [0, 0])
            
            fig.update_layout(
                title="Cluster Visualization (PCA Projection)",
                xaxis_title=f"PC1 ({variance_explained[0]*100:.1f}% variance)",
                yaxis_title=f"PC2 ({variance_explained[1]*100:.1f}% variance)",
                height=500,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster statistics
        with st.expander("📊 Cluster Statistics & Interpretation"):
            cluster_stats = cluster_data['cluster_stats']
            
            for cluster_name, stats in cluster_stats.items():
                st.markdown(f"#### {cluster_name}")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.write(f"**Size:** {stats['size']} ({stats['percentage']:.1f}%)")
                    
                    # Progress bar
                    progress_html = f"""
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {stats['percentage']}%;"></div>
                    </div>
                    """
                    st.markdown(progress_html, unsafe_allow_html=True)
                
                with col2:
                    if 'feature_importance' in stats and stats['feature_importance']:
                        st.write("**Top Features:**")
                        top_5 = sorted(
                            stats['feature_importance'].items(),
                            key=lambda x: abs(x[1]['mean']),
                            reverse=True
                        )[:5]
                        
                        for feature, values in top_5:
                            feature_short = feature[:40] + '...' if len(feature) > 40 else feature
                            st.write(f"- {feature_short}: {values['mean']:.2f} ± {values['std']:.2f}")
                
                st.divider()



def generate_summary_report(results):
    """Generate comprehensive summary report"""
    
    st.header("📋 Comprehensive Analysis Summary")
    
    # Overall statistics
    st.markdown("### 📊 Overall Statistics")
    
    summary_data = {
        'Component': [],
        'Status': [],
        'Key Metrics': []
    }
    
    # Feature extraction summary
    if 'features' in results:
        for metric, features in results['features'].items():
            summary_data['Component'].append(f"Features ({metric})")
            summary_data['Status'].append("✅ Complete")
            summary_data['Key Metrics'].append(f"{len(features.columns)} features, {len(features)} windows")
    
    # Forecasting summary
    if 'forecasts' in results:
        for metric, forecast_data in results['forecasts'].items():
            mape = forecast_data['metrics']['mape']
            summary_data['Component'].append(f"Forecast ({metric})")
            status = "✅ Excellent" if mape < 10 else "⚠️ Fair"
            summary_data['Status'].append(status)
            summary_data['Key Metrics'].append(f"MAPE: {mape:.2f}%, {len(forecast_data['anomalies'])} anomalies")
    
    # Clustering summary
    if 'clusters' in results:
        for metric, cluster_data in results['clusters'].items():
            silhouette = cluster_data['quality_metrics']['silhouette_score']
            summary_data['Component'].append(f"Clustering ({metric})")
            status = "✅ Excellent" if silhouette > 0.5 else "⚠️ Fair"
            summary_data['Status'].append(status)
            summary_data['Key Metrics'].append(
                f"Silhouette: {silhouette:.3f}, {cluster_data['quality_metrics']['n_clusters']} clusters"
            )
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Export option
    st.markdown("### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Export Summary as CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"milestone2_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Export Detailed Report"):
            report_text = generate_detailed_text_report(results)
            st.download_button(
                label="⬇️ Download Report",
                data=report_text,
                file_name=f"milestone2_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )



def generate_detailed_text_report(results):
    """Generate detailed text report"""
    
    report = []
    report.append("="*70)
    report.append("FITPULSE MILESTONE 2 - COMPREHENSIVE ANALYSIS REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Feature extraction section
    if 'features' in results:
        report.append("\n" + "="*70)
        report.append("FEATURE EXTRACTION RESULTS")
        report.append("="*70)
        
        for metric, features in results['features'].items():
            report.append(f"\n{metric.upper()}:")
            report.append(f"  - Features extracted: {len(features.columns)}")
            report.append(f"  - Windows created: {len(features)}")
            report.append(f"  - Top 5 features by variance:")
            
            top_features = features.var().nlargest(5)
            for i, (feature, variance) in enumerate(top_features.items(), 1):
                report.append(f"    {i}. {feature}: {variance:.4f}")
    
    # Forecasting section
    if 'forecasts' in results:
        report.append("\n" + "="*70)
        report.append("TIME SERIES FORECASTING RESULTS")
        report.append("="*70)
        
        for metric, forecast_data in results['forecasts'].items():
            metrics = forecast_data['metrics']
            report.append(f"\n{metric.upper()}:")
            report.append(f"  - MAPE: {metrics['mape']:.2f}%")
            report.append(f"  - MAE: {metrics['mae']:.2f}")
            report.append(f"  - RMSE: {metrics['rmse']:.2f}")
            report.append(f"  - R²: {metrics['r2']:.3f}")
            report.append(f"  - Anomalies detected: {len(forecast_data['anomalies'])}")
    
    # Clustering section
    if 'clusters' in results:
        report.append("\n" + "="*70)
        report.append("BEHAVIORAL PATTERN CLUSTERING RESULTS")
        report.append("="*70)
        
        for metric, cluster_data in results['clusters'].items():
            metrics = cluster_data['quality_metrics']
            report.append(f"\n{metric.upper()}:")
            report.append(f"  - Silhouette Score: {metrics['silhouette_score']:.3f}")
            report.append(f"  - Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f}")
            report.append(f"  - Number of clusters: {metrics['n_clusters']}")
            
            report.append(f"\n  Cluster Distribution:")
            for cluster_name, stats in cluster_data['cluster_stats'].items():
                report.append(f"    - {cluster_name}: {stats['size']} samples ({stats['percentage']:.1f}%)")
    
    report.append("\n" + "="*70)
    report.append("END OF REPORT")
    report.append("="*70)
    
    return "\n".join(report)



def create_metric_card(title, value, delta=None, icon="📊"):
    """Create a styled metric card with purple background"""
    
    delta_html = ""
    if delta:
        delta_html = f'<p style="color: rgba(255, 255, 255, 0.9); margin: 0;">{delta}</p>'
    
    card_html = f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <h2 style="margin: 0.5rem 0; color: white;">{value}</h2>
        {delta_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)



def show_info_message(message, type="info"):
    """Show styled info message"""
    
    box_class = f"{type}-box"
    
    message_html = f"""
    <div class="{box_class}">
        {message}
    </div>
    """
    
    st.markdown(message_html, unsafe_allow_html=True)