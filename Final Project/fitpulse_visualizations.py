# fitpulse_visualizations.py
# 🎨 RICH VISUALIZATIONS FOR MILESTONES 2 & 4
# All charts, graphs, heatmaps, 3D plots, sunbursts etc.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# THEME
# ============================================================================

def apply_purple_theme():
    """Apply purple theme with black sidebar"""
    st.markdown("""
        <style>
        h1, h2, h3 { color: #a78bfa !important; text-shadow: 0 0 20px rgba(124, 58, 237, 0.5); }
        div[data-testid="stMetricValue"] {
            background: linear-gradient(135deg, #7c3aed, #6d28d9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
            font-size: 2rem !important;
        }
        div[data-testid="stMetricLabel"] { color: #c4b5fd !important; font-weight: 600; }
        div.stButton > button, div.stDownloadButton > button {
            background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.75rem 2rem !important;
            box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
            transition: all 0.3s ease !important;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background: linear-gradient(135deg, #6d28d9, #5b21b6) !important;
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6) !important;
        }
        section[data-testid="stSidebar"] { background-color: #0e1117 !important; }
        section[data-testid="stSidebar"] * { color: #fafafa !important; }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# MILESTONE 2 VISUALIZATIONS
# ============================================================================

class M2Visualizations:
    """Rich visualizations for Milestone 2"""
    
    @staticmethod
    def display_feature_results(features: Dict):
        """Display feature extraction results with heatmap"""
        
        with st.expander(f"🔬 Features Extracted ({len(features)} datasets)", expanded=True):
            
            for metric, feat_df in features.items():
                st.markdown(f"### 📊 {metric.replace('_', ' ').title()}")
                st.write(f"**{len(feat_df.columns)} features** extracted from **{len(feat_df)} windows**")
                
                # Feature importance heatmap
                if len(feat_df.columns) > 0:
                    # Top 20 most important features by variance
                    feature_importance = feat_df.var().nlargest(20)
                    
                    if len(feature_importance) > 0:
                        fig = go.Figure(data=go.Bar(
                            x=feature_importance.values,
                            y=[f.split('__')[-1][:30] for f in feature_importance.index],
                            orientation='h',
                            marker=dict(
                                color=feature_importance.values,
                                colorscale='Purples',
                                showscale=True,
                                colorbar=dict(title="Variance")
                            )
                        ))
                        
                        fig.update_layout(
                            title=f"Top 20 Features by Variance - {metric.title()}",
                            xaxis_title="Variance",
                            yaxis_title="Feature",
                            height=600,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#fafafa')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation heatmap (top 15 features)
                    top_features = feat_df.var().nlargest(15).index
                    if len(top_features) > 1:
                        corr_matrix = feat_df[top_features].corr()
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=[f.split('__')[-1][:20] for f in corr_matrix.columns],
                            y=[f.split('__')[-1][:20] for f in corr_matrix.index],
                            colorscale='Purples',
                            zmid=0,
                            colorbar=dict(title="Correlation")
                        ))
                        
                        fig.update_layout(
                            title="Feature Correlation Matrix",
                            height=500,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#fafafa', size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
    
    @staticmethod
    def display_forecast_results(forecasts: Dict, original_data: Dict):
        """Display forecast results with plots"""
        
        with st.expander(f"📈 Forecasts Generated ({len(forecasts)} datasets)", expanded=True):
            
            for metric, forecast in forecasts.items():
                st.markdown(f"### 📊 {metric.replace('_', ' ').title()}")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAPE", f"{forecast['metrics']['mape']:.2f}%")
                with col2:
                    st.metric("RMSE", f"{forecast['metrics']['rmse']:.2f}")
                with col3:
                    st.metric("MAE", f"{forecast['metrics']['mae']:.2f}")
                with col4:
                    st.metric("Anomalies", len(forecast['anomalies']))
                
                # Forecast plot
                fig = go.Figure()
                
                # Historical data
                if metric in original_data:
                    df = original_data[metric]
                    metric_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df[metric_col],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#7c3aed', width=2)
                    ))
                
                # Forecast
                forecast_df = forecast['forecast']
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#10b981', width=2, dash='dash')
                ))
                
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(width=0),
                    fillcolor='rgba(16, 185, 129, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
                
                # Anomalies
                if forecast['anomalies']:
                    anomaly_times = [a['timestamp'] for a in forecast['anomalies']]
                    anomaly_values = [a['actual_value'] for a in forecast['anomalies']]
                    
                    fig.add_trace(go.Scatter(
                        x=anomaly_times,
                        y=anomaly_values,
                        mode='markers',
                        name='Forecast Anomalies',
                        marker=dict(color='#ef4444', size=10, symbol='x')
                    ))
                
                fig.update_layout(
                    title=f"Time Series Forecast - {metric.title()}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
    
    @staticmethod
    def display_cluster_results(clusters: Dict):
        """Display clustering results"""
        
        with st.expander(f"🎯 Patterns Discovered ({len(clusters)} datasets)", expanded=True):
            
            for metric, cluster_data in clusters.items():
                st.markdown(f"### 📊 {metric.replace('_', ' ').title()}")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Clusters", cluster_data['quality_metrics']['n_clusters'])
                with col2:
                    st.metric("Silhouette", f"{cluster_data['quality_metrics']['silhouette_score']:.3f}")
                with col3:
                    st.metric("Davies-Bouldin", f"{cluster_data['quality_metrics']['davies_bouldin_score']:.3f}")
                with col4:
                    st.metric("Calinski-Harabasz", f"{cluster_data['quality_metrics']['calinski_harabasz_score']:.1f}")
                
                # Cluster visualization (2D scatter with PCA)
                if 'cluster_labels' in cluster_data:
                    labels = cluster_data['cluster_labels']
                    
                    # Simple 2D projection using first 2 features
                    if 'features_scaled' in cluster_data:
                        features = cluster_data['features_scaled']
                        
                        # Take first 2 components for visualization
                        if features.shape[1] >= 2:
                            x = features[:, 0]
                            y = features[:, 1]
                            
                            fig = go.Figure()
                            
                            # Plot each cluster
                            unique_labels = np.unique(labels)
                            colors = px.colors.qualitative.Set3
                            
                            for i, label in enumerate(unique_labels):
                                mask = labels == label
                                fig.add_trace(go.Scatter(
                                    x=x[mask],
                                    y=y[mask],
                                    mode='markers',
                                    name=f'Cluster {label}',
                                    marker=dict(
                                        size=8,
                                        color=colors[i % len(colors)],
                                        line=dict(width=0.5, color='white')
                                    )
                                ))
                            
                            fig.update_layout(
                                title=f"Cluster Visualization - {metric.title()}",
                                xaxis_title="Feature Component 1",
                                yaxis_title="Feature Component 2",
                                height=500,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='#fafafa')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Cluster size distribution
                if 'cluster_labels' in cluster_data:
                    unique, counts = np.unique(cluster_data['cluster_labels'], return_counts=True)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Cluster {i}' for i in unique],
                            y=counts,
                            marker_color='#7c3aed',
                            text=counts,
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Cluster Size Distribution",
                        xaxis_title="Cluster",
                        yaxis_title="Number of Windows",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#fafafa')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()

# ============================================================================
# MILESTONE 4 VISUALIZATIONS
# ============================================================================

class M4Visualizations:
    """Rich visualizations for Milestone 4 Dashboard"""
    
    @staticmethod
    def display_executive_dashboard(anomaly_results: Dict, processed_data: Dict, m2_results: Dict):
        """Display comprehensive M4 dashboard with all visualizations"""
        
        # Generate insights
        insights = M4Visualizations._generate_insights(anomaly_results, processed_data)
        
        st.subheader("🎯 Executive Health Insights")
        
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Anomalies", f"{insights['summary']['total_anomalies']:,}")
        with col2:
            st.metric("Anomaly Rate", f"{insights['summary']['overall_anomaly_rate']:.2f}%")
        with col3:
            risk = insights['risk_assessment']
            st.markdown(f"**Risk Level**")
            st.markdown(f"## {risk['icon']} {risk['level']}")
        with col4:
            st.metric("High Risk", f"{insights['severity_distribution']['high']:,}")
        with col5:
            st.metric("Health Score", f"{insights['health_score']['score']}/100")
        
        st.markdown("---")
        
        # 1. Severity Distribution Pie Chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Severity Distribution")
            severity_df = pd.DataFrame({
                'Severity': ['High', 'Medium', 'Low'],
                'Count': [
                    insights['severity_distribution']['high'],
                    insights['severity_distribution']['medium'],
                    insights['severity_distribution']['low']
                ]
            })
            
            fig = px.pie(
                severity_df, values='Count', names='Severity',
                color='Severity',
                color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#10b981'},
                hole=0.4,
                height=400
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 Anomalies by Type")
            type_data = []
            for dt, info in insights['by_data_type'].items():
                type_data.append({
                    'Type': dt.replace('_', ' ').title(),
                    'Count': info['anomaly_count']
                })
            
            if type_data:
                type_df = pd.DataFrame(type_data)
                fig = px.bar(
                    type_df, x='Type', y='Count',
                    color='Type',
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                    text='Count',
                    height=400
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa'),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # 2. Timeline View
        M4Visualizations._create_timeline_view(anomaly_results)
        
        # 3. Heatmap View
        M4Visualizations._create_heatmap_view(anomaly_results)
        
        # 4. Hourly Patterns
        M4Visualizations._create_hourly_patterns(insights)
        
        # 5. Day of Week Patterns
        M4Visualizations._create_day_of_week_patterns(insights)
        
        # 6. 3D Visualization
        M4Visualizations._create_3d_scatter(anomaly_results)
        
        # 7. Sunburst Chart
        M4Visualizations._create_sunburst(insights)
        
        # 8. Recommendations
        st.subheader("💡 Key Recommendations")
        for i, rec in enumerate(insights['recommendations'][:10], 1):
            st.markdown(f"{i}. {rec}")
    
    @staticmethod
    def _generate_insights(anomaly_results: Dict, processed_data: Dict) -> Dict:
        """Generate comprehensive insights"""
        
        total_anomalies = sum([r['counts']['total'] for r in anomaly_results.values()])
        total_records = sum([len(df) for df in processed_data.values()])
        
        high_risk = sum([
            (r['data_with_anomalies']['anomaly_severity'] == 'High').sum() 
            for r in anomaly_results.values()
        ])
        medium_risk = sum([
            (r['data_with_anomalies']['anomaly_severity'] == 'Medium').sum() 
            for r in anomaly_results.values()
        ])
        low_risk = sum([
            (r['data_with_anomalies']['anomaly_severity'] == 'Low').sum() 
            for r in anomaly_results.values()
        ])
        
        insights = {
            'summary': {
                'total_anomalies': total_anomalies,
                'total_records': total_records,
                'overall_anomaly_rate': (total_anomalies / total_records * 100) if total_records > 0 else 0
            },
            'severity_distribution': {
                'high': high_risk,
                'medium': medium_risk,
                'low': low_risk
            },
            'by_data_type': {},
            'hourly_patterns': {},
            'day_of_week_patterns': {}
        }
        
        for data_type, result in anomaly_results.items():
            insights['by_data_type'][data_type] = {
                'anomaly_count': result['counts']['total'],
                'anomaly_percentage': (result['counts']['total'] / len(result['data_with_anomalies']) * 100) if len(result['data_with_anomalies']) > 0 else 0
            }
            
            # Hourly patterns
            df = result['data_with_anomalies']
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
                
                anomaly_df = df[df['is_anomaly']]
                if len(anomaly_df) > 0:
                    hourly = anomaly_df['hour'].value_counts().sort_index()
                    insights['hourly_patterns'][data_type] = {
                        'peak_hour': int(hourly.idxmax()) if len(hourly) > 0 else None,
                        'peak_count': int(hourly.max()) if len(hourly) > 0 else 0,
                        'distribution': hourly.to_dict()
                    }
                    
                    dow = anomaly_df['day_of_week'].value_counts()
                    insights['day_of_week_patterns'][data_type] = {
                        'peak_day': dow.idxmax() if len(dow) > 0 else None,
                        'peak_count': int(dow.max()) if len(dow) > 0 else 0,
                        'distribution': dow.to_dict()
                    }
        
        # Health score
        health_score = 100 - min((total_anomalies / total_records * 100) * 10, 100) if total_records > 0 else 100
        insights['health_score'] = {
            'score': round(health_score, 1),
            'grade': M4Visualizations._get_health_grade(health_score),
            'interpretation': M4Visualizations._interpret_health_score(health_score)
        }
        
        # Risk assessment
        if total_anomalies > 100 or high_risk > 20:
            insights['risk_assessment'] = {'level': 'High', 'icon': '🔴'}
        elif total_anomalies > 50 or high_risk > 5:
            insights['risk_assessment'] = {'level': 'Medium', 'icon': '🟠'}
        else:
            insights['risk_assessment'] = {'level': 'Low', 'icon': '🟢'}
        
        # Recommendations
        insights['recommendations'] = M4Visualizations._generate_recommendations(total_anomalies, high_risk, insights)
        
        return insights
    
    @staticmethod
    def _get_health_grade(score):
        if score >= 90: return 'A (Excellent)'
        elif score >= 80: return 'B (Good)'
        elif score >= 70: return 'C (Fair)'
        elif score >= 60: return 'D (Poor)'
        else: return 'F (Critical)'
    
    @staticmethod
    def _interpret_health_score(score):
        if score >= 90: return "Excellent health with minimal anomalies"
        elif score >= 80: return "Good health with minor irregularities"
        elif score >= 70: return "Fair status requiring attention"
        elif score >= 60: return "Poor metrics with significant anomalies"
        else: return "Critical status - consult healthcare provider"
    
    @staticmethod
    def _generate_recommendations(total_anomalies, high_risk, insights):
        recs = []
        if total_anomalies > 100:
            recs.append("🚨 HIGH ALERT: Significant anomalies detected")
            recs.append("👨‍⚕️ Schedule medical consultation within 24-48 hours")
            recs.append("🔍 Check device calibration")
        elif total_anomalies > 50:
            recs.append("⚠️ MODERATE: Notable anomalies - monitor closely")
            recs.append("📊 Track patterns over next 3-7 days")
            recs.append("🔧 Verify device is worn properly")
        elif total_anomalies > 0:
            recs.append("ℹ️ LOW: Minor anomalies detected")
            recs.append("📈 Continue regular monitoring")
        else:
            recs.append("✅ EXCELLENT: No anomalies detected")
            recs.append("✓ All metrics within healthy ranges")
        
        for data_type, pattern in insights.get('hourly_patterns', {}).items():
            if pattern['peak_hour'] is not None:
                recs.append(f"⏰ {data_type.title()}: Peak anomalies at {pattern['peak_hour']}:00")
        
        return recs
    
    @staticmethod
    def _create_timeline_view(anomaly_results: Dict):
        """Timeline visualization"""
        st.subheader("📅 Anomaly Timeline Analysis")
        
        for data_type, result in anomaly_results.items():
            df = result['data_with_anomalies']
            if 'timestamp' not in df.columns:
                continue
            
            anomaly_mask = df['is_anomaly']
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"💜 {data_type.replace('_', ' ').title()} Timeline", expanded=False):
                metric_col = df.columns[1]
                
                fig = go.Figure()
                
                normal = df[~anomaly_mask]
                fig.add_trace(go.Scatter(
                    x=normal['timestamp'],
                    y=normal[metric_col],
                    mode='lines',
                    name='Normal',
                    line=dict(color='#7c3aed', width=2)
                ))
                
                anomaly_df = df[anomaly_mask]
                fig.add_trace(go.Scatter(
                    x=anomaly_df['timestamp'],
                    y=anomaly_df[metric_col],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='#ef4444', size=10, symbol='x')
                ))
                
                fig.update_layout(
                    title=f"{data_type.title()} - Timeline View",
                    xaxis_title="Time",
                    yaxis_title=metric_col.title(),
                    hovermode='x unified',
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _create_heatmap_view(anomaly_results: Dict):
        """Heatmap visualization"""
        st.subheader("🔥 Anomaly Heatmap")
        
        for data_type, result in anomaly_results.items():
            df = result['data_with_anomalies']
            if 'timestamp' not in df.columns:
                continue
            
            anomaly_mask = df['is_anomaly']
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"🗓️ {data_type.replace('_', ' ').title()} Heatmap", expanded=False):
                anomaly_df = df[anomaly_mask].copy()
                anomaly_df['date'] = pd.to_datetime(anomaly_df['timestamp']).dt.date
                anomaly_df['hour'] = pd.to_datetime(anomaly_df['timestamp']).dt.hour
                
                heatmap_data = anomaly_df.groupby(['date', 'hour']).size().reset_index(name='count')
                pivot = heatmap_data.pivot(index='hour', columns='date', values='count').fillna(0)
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[str(d) for d in pivot.columns],
                    y=pivot.index,
                    colorscale='Purples',
                    colorbar=dict(title="Count"),
                    hovertemplate='Date: %{x}<br>Hour: %{y}<br>Anomalies: %{z}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Anomaly Concentration - {data_type.title()}",
                    xaxis_title="Date",
                    yaxis_title="Hour of Day",
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(20,20,30,0.8)',
                    font=dict(color='#fafafa')
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _create_hourly_patterns(insights: Dict):
        """Hourly distribution"""
        st.subheader("⏰ Hourly Anomaly Patterns")
        
        for data_type, pattern in insights.get('hourly_patterns', {}).items():
            if pattern.get('distribution'):
                with st.expander(f"📊 {data_type.replace('_', ' ').title()} Hourly", expanded=False):
                    hours = list(pattern['distribution'].keys())
                    counts = list(pattern['distribution'].values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=hours, y=counts,
                        marker_color='#7c3aed',
                        text=counts,
                        textposition='auto'
                    )])
                    
                    fig.update_layout(
                        title=f"Hourly Distribution - {data_type.title()}",
                        xaxis_title="Hour",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#fafafa')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Peak:** {pattern['peak_hour']}:00 ({pattern['peak_count']} anomalies)")
    
    @staticmethod
    def _create_day_of_week_patterns(insights: Dict):
        """Day of week patterns"""
        st.subheader("📅 Day of Week Patterns")
        
        for data_type, pattern in insights.get('day_of_week_patterns', {}).items():
            if pattern.get('distribution'):
                with st.expander(f"📊 {data_type.replace('_', ' ').title()} Day of Week", expanded=False):
                    days = list(pattern['distribution'].keys())
                    counts = list(pattern['distribution'].values())
                    
                    fig = go.Figure(data=[go.Bar(
                        x=days, y=counts,
                        marker_color='#6d28d9',
                        text=counts,
                        textposition='auto'
                    )])
                    
                    fig.update_layout(
                        title=f"Day of Week Distribution - {data_type.title()}",
                        xaxis_title="Day",
                        yaxis_title="Count",
                        height=400,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#fafafa')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Peak:** {pattern['peak_day']} ({pattern['peak_count']} anomalies)")
    
    @staticmethod
    def _create_3d_scatter(anomaly_results: Dict):
        """3D scatter plot"""
        st.subheader("🎲 3D Anomaly Visualization")
        
        for data_type, result in anomaly_results.items():
            df = result['data_with_anomalies']
            if 'timestamp' not in df.columns:
                continue
            
            anomaly_mask = df['is_anomaly']
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"🎯 {data_type.replace('_', ' ').title()} 3D", expanded=False):
                df_plot = df[anomaly_mask].head(500).copy()
                
                if len(df_plot) == 0:
                    st.warning("No anomaly data for 3D")
                    continue
                
                df_plot['hour'] = pd.to_datetime(df_plot['timestamp']).dt.hour
                df_plot['day'] = pd.to_datetime(df_plot['timestamp']).dt.day
                
                metric_col = df_plot.columns[1]
                
                fig = go.Figure(data=[go.Scatter3d(
                    x=df_plot['day'],
                    y=df_plot['hour'],
                    z=df_plot[metric_col],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=df_plot[metric_col],
                        colorscale='Purples',
                        showscale=True,
                        colorbar=dict(title=metric_col.title()),
                        line=dict(width=0.5, color='white')
                    ),
                    text=[f"Day: {d}<br>Hour: {h}<br>{metric_col}: {v}" 
                          for d, h, v in zip(df_plot['day'], df_plot['hour'], df_plot[metric_col])],
                    hoverinfo='text'
                )])
                
                fig.update_layout(
                    title=f"3D Anomaly Distribution - {data_type.title()}",
                    scene=dict(
                        xaxis_title='Day',
                        yaxis_title='Hour',
                        zaxis_title=metric_col.title(),
                        bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
                        zaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
                    ),
                    height=700,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.info(f"Showing {len(df_plot)} anomalies")
    
    @staticmethod
    def _create_sunburst(insights: Dict):
        """Sunburst chart"""
        st.subheader("☀️ Anomaly Hierarchy")
        
        labels = ['All Anomalies']
        parents = ['']
        values = [insights['summary']['total_anomalies']]
        colors = ['#7c3aed']
        
        for data_type, info in insights['by_data_type'].items():
            if info['anomaly_count'] > 0:
                labels.append(data_type.replace('_', ' ').title())
                parents.append('All Anomalies')
                values.append(info['anomaly_count'])
                colors.append('#6d28d9')
        
        if len(labels) > 1:
            fig = go.Figure(go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                marker=dict(colors=colors),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Anomaly Type Hierarchy",
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa', size=14)
            )
            
            st.plotly_chart(fig, use_container_width=True)
