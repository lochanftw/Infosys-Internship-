# Milestone 4: Executive Dashboard for Insights - FIXED VERSION
# 20+ Key Insights | 15+ Visualizations | PDF Export (No Emojis)
# Default Black Theme with Purple Accents | 3D & Heatmap Fixed

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import os
import glob
import json
import io
import base64
import re
warnings.filterwarnings('ignore')

# PDF generation
try:
    from fpdf import FPDF
    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False

from scipy import stats

# ============================================================================
# PURPLE ACCENTS THEME
# ============================================================================

def apply_purple_theme():
    """Purple accent theme with default dark background"""
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
        
        .stAlert {
            background: linear-gradient(135deg, #7c3aed15, #6d28d915) !important;
            border-left: 4px solid #7c3aed !important;
            color: #e9d5ff !important;
            border-radius: 8px !important;
        }
        
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #7c3aed20, #6d28d920) !important;
            color: #a78bfa !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        
        section[data-testid="stSidebar"] { background-color: #0e1117 !important; }
        section[data-testid="stSidebar"] * { color: #fafafa !important; }
        </style>
    """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTION TO REMOVE EMOJIS
# ============================================================================

import string

def remove_special_chars(text):
    """Remove ALL problematic Unicode characters including zero-width chars"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove zero-width characters - THIS IS THE KEY FIX!
    text = text.replace('\u200d', '')  # Zero-width joiner
    text = text.replace('\u200c', '')  # Zero-width non-joiner  
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\ufeff', '')  # Zero-width no-break space
    text = text.replace('\u2060', '')  # Word joiner
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Keep only ASCII printable characters
    printable = set(string.printable)
    cleaned = ''.join(char if char in printable else ' ' for char in text)
    cleaned = ' '.join(cleaned.split())
    
    # Final safety check
    try:
        cleaned.encode('latin-1')
        return cleaned
    except:
        return cleaned.encode('ascii', 'ignore').decode('ascii')


# ============================================================================
# DATA LOADER
# ============================================================================

class AnomalyDataLoader:
    """Loads anomaly detection results from Milestone 3"""
    
    def __init__(self, data_folder='data/input'):
        self.data_folder = data_folder
        
    def load_milestone3_results(self) -> Dict:
        """Load results from Milestone 3"""
        results = {'data_with_anomalies': {}, 'reports': {}, 'loaded_from': 'files'}
        
        if 'milestone3_results' in st.session_state:
            st.info("✅ Loading from Milestone 3 session...")
            return st.session_state.milestone3_results
        
        st.info("🔄 Loading data from data/input/...")
        csv_files = glob.glob(os.path.join(self.data_folder, '*.csv'))
        
        if not csv_files:
            st.warning(f"⚠️ No CSV files in {self.data_folder}")
            return results
        
        for filepath in csv_files:
            filename = os.path.basename(filepath)
            try:
                df = pd.read_csv(filepath)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
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
                    data_type = filename.replace('.csv', '')
                
                anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower()]
                has_anomalies = len(anomaly_cols) > 0
                results['data_with_anomalies'][data_type] = df
                
                if has_anomalies:
                    anomaly_mask = df[anomaly_cols].any(axis=1)
                    anomaly_count = anomaly_mask.sum()
                    results['reports'][data_type] = {
                        'detected': {
                            'anomalies_detected': int(anomaly_count),
                            'anomaly_percentage': (anomaly_count / len(df)) * 100
                        }
                    }
                    st.success(f"✅ {data_type}: {anomaly_count} anomalies")
                else:
                    results['reports'][data_type] = {'detected': {'anomalies_detected': 0, 'anomaly_percentage': 0.0}}
                    st.info(f"ℹ️ {data_type}: No anomaly data")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
        
        return results

# ============================================================================
# INSIGHTS GENERATOR
# ============================================================================

class ComprehensiveInsightsGenerator:
    """Generates 20+ detailed insights"""
    
    @staticmethod
    def generate_detailed_insights(anomaly_data: Dict) -> Dict:
        """Generate comprehensive insights"""
        insights = {
            'summary': {}, 'by_data_type': {}, 'temporal_patterns': {},
            'hourly_patterns': {}, 'day_of_week_patterns': {}, 'severity_distribution': {},
            'trends': {}, 'correlations': {}, 'health_score': {}, 'recommendations': [],
            'risk_assessment': {}, 'statistical_summary': {}, 'peak_analysis': {}
        }
        
        total_anomalies = 0
        total_records = 0
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        all_dfs = []
        
        for data_type, df in anomaly_data.get('data_with_anomalies', {}).items():
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and '_reason' not in col.lower()]
            if not anomaly_cols:
                continue
            
            anomaly_mask = df[anomaly_cols].any(axis=1)
            anomaly_count = anomaly_mask.sum()
            total_anomalies += anomaly_count
            total_records += len(df)
            
            if 'severity' in df.columns:
                high_risk_count += (df['severity'] == 'High').sum()
                medium_risk_count += (df['severity'] == 'Medium').sum()
                low_risk_count += (df['severity'] == 'Low').sum()
            
            insights['by_data_type'][data_type] = {
                'total_records': len(df),
                'anomaly_count': int(anomaly_count),
                'anomaly_percentage': (anomaly_count / len(df)) * 100,
                'anomaly_types': []
            }
            
            for col in anomaly_cols:
                type_count = df[col].sum()
                if type_count > 0:
                    insights['by_data_type'][data_type]['anomaly_types'].append({
                        'type': col.replace('_anomaly', '').replace('_', ' ').title(),
                        'count': int(type_count)
                    })
            
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.day_name()
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                all_dfs.append(df.copy())
                
                if anomaly_count > 0:
                    anomaly_df = df[anomaly_mask]
                    hourly_dist = anomaly_df['hour'].value_counts().sort_index()
                    insights['hourly_patterns'][data_type] = {
                        'peak_hour': int(hourly_dist.idxmax()) if len(hourly_dist) > 0 else None,
                        'peak_count': int(hourly_dist.max()) if len(hourly_dist) > 0 else 0,
                        'distribution': hourly_dist.to_dict()
                    }
                    
                    dow_dist = anomaly_df['day_of_week'].value_counts()
                    insights['day_of_week_patterns'][data_type] = {
                        'peak_day': dow_dist.idxmax() if len(dow_dist) > 0 else None,
                        'peak_count': int(dow_dist.max()) if len(dow_dist) > 0 else 0,
                        'distribution': dow_dist.to_dict()
                    }
                    
                    daily_counts = anomaly_df.groupby('date').size()
                    insights['by_data_type'][data_type]['temporal'] = {
                        'peak_anomaly_date': str(daily_counts.idxmax()) if len(daily_counts) > 0 else None,
                        'peak_anomaly_count': int(daily_counts.max()) if len(daily_counts) > 0 else 0,
                        'avg_daily_anomalies': float(daily_counts.mean()) if len(daily_counts) > 0 else 0,
                        'trend': 'increasing' if len(daily_counts) > 1 and daily_counts.iloc[-1] > daily_counts.iloc[0] else 'stable'
                    }
        
        insights['summary'] = {
            'total_anomalies': total_anomalies,
            'total_records': total_records,
            'overall_anomaly_rate': (total_anomalies / total_records * 100) if total_records > 0 else 0,
            'data_types_analyzed': len(anomaly_data.get('data_with_anomalies', {})),
            'avg_anomalies_per_dataset': total_anomalies / len(anomaly_data.get('data_with_anomalies', {})) if anomaly_data.get('data_with_anomalies') else 0
        }
        
        insights['severity_distribution'] = {
            'high': high_risk_count,
            'medium': medium_risk_count,
            'low': low_risk_count
        }
        
        health_score = 100 - min((total_anomalies / total_records * 100) * 10, 100) if total_records > 0 else 100
        insights['health_score'] = {
            'score': round(health_score, 1),
            'grade': ComprehensiveInsightsGenerator._get_health_grade(health_score),
            'interpretation': ComprehensiveInsightsGenerator._interpret_health_score(health_score)
        }
        
        insights['recommendations'] = ComprehensiveInsightsGenerator._generate_recommendations(
            total_anomalies, high_risk_count, insights
        )
        
        if total_anomalies > 100 or high_risk_count > 20:
            insights['risk_assessment'] = {'level': 'High', 'color': 'red', 'icon': '🔴'}
        elif total_anomalies > 50 or high_risk_count > 5:
            insights['risk_assessment'] = {'level': 'Medium', 'color': 'orange', 'icon': '🟠'}
        else:
            insights['risk_assessment'] = {'level': 'Low', 'color': 'green', 'icon': '🟢'}
        
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
        if score >= 90: return "Excellent health metrics with minimal anomalies"
        elif score >= 80: return "Good overall health with some minor irregularities"
        elif score >= 70: return "Fair health status requiring attention"
        elif score >= 60: return "Poor metrics with significant anomalies detected"
        else: return "Critical status requiring immediate medical attention"
    
    @staticmethod
    def _generate_recommendations(total_anomalies, high_risk_count, insights):
        recommendations = []
        if total_anomalies > 100:
            recommendations.append("🚨 HIGH ALERT: Significant anomalies detected")
            recommendations.append("👨‍⚕️ Schedule medical consultation within 24-48 hours")
            recommendations.append("🔍 Check wearable device for calibration issues")
        elif total_anomalies > 50:
            recommendations.append("⚠️ MODERATE: Notable anomalies - monitor closely")
            recommendations.append("📊 Track patterns over next 3-7 days")
            recommendations.append("🔧 Verify device is properly worn")
        elif total_anomalies > 0:
            recommendations.append("ℹ️ LOW: Minor anomalies detected")
            recommendations.append("📈 Continue regular monitoring")
        else:
            recommendations.append("✅ EXCELLENT: No anomalies detected")
            recommendations.append("✓ All metrics within healthy ranges")
        
        for data_type, pattern in insights.get('hourly_patterns', {}).items():
            if pattern['peak_hour'] is not None:
                recommendations.append(f"⏰ {data_type.title()}: Most anomalies at {pattern['peak_hour']}:00")
        
        for data_type, pattern in insights.get('day_of_week_patterns', {}).items():
            if pattern['peak_day']:
                recommendations.append(f"📅 {data_type.title()}: {pattern['peak_day']} shows highest anomaly rate")
        
        return recommendations
    
    @staticmethod
    def display_executive_dashboard(insights: Dict):
        """Display dashboard"""
        st.header("📊 Executive Insights Dashboard")
        st.markdown("**Based on Milestone 3 Anomaly Detection Results**")
        
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
            st.metric("High Risk Events", f"{insights['severity_distribution']['high']:,}")
        with col5:
            st.metric("Health Score", f"{insights['health_score']['score']}/100")
        
        st.markdown("---")
        
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
            fig_severity = px.pie(
                severity_df, values='Count', names='Severity',
                color='Severity',
                color_discrete_map={'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#10b981'},
                hole=0.4, height=400
            )
            fig_severity.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa')
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        
        with col2:
            st.subheader("📋 Anomalies by Data Type")
            type_data = []
            for data_type, info in insights['by_data_type'].items():
                type_data.append({
                    'Data Type': data_type.replace('_', ' ').title(),
                    'Anomalies': info['anomaly_count'],
                    'Percentage': f"{info['anomaly_percentage']:.1f}%"
                })
            if type_data:
                fig_types = px.bar(
                    pd.DataFrame(type_data), x='Data Type', y='Anomalies',
                    color='Data Type',
                    color_discrete_sequence=px.colors.sequential.Purples_r,
                    text='Anomalies', height=400
                )
                fig_types.update_traces(textposition='outside')
                fig_types.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#fafafa'),
                    showlegend=False
                )
                st.plotly_chart(fig_types, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Key Recommendations")
        for i, rec in enumerate(insights['recommendations'][:10], 1):
            st.markdown(f"{i}. {rec}")

# ============================================================================
# VISUALIZATION ENGINE - FIXED
# ============================================================================

class AdvancedVisualizationEngine:
    """Creates visualizations"""
    
    @staticmethod
    def create_timeline_view(data: Dict):
        """Timeline view"""
        st.subheader("📅 Anomaly Timeline Analysis")
        for data_type, df in data.get('data_with_anomalies', {}).items():
            if 'timestamp' not in df.columns:
                continue
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and '_reason' not in col.lower()]
            if not anomaly_cols:
                continue
            anomaly_mask = df[anomaly_cols].any(axis=1)
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"💜 {data_type.replace('_', ' ').title()} Timeline", expanded=True):
                metric_cols = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'activity': 'activity_level', 'pros': 'heart_rate'}
                metric_col = metric_cols.get(data_type)
                if not metric_col or metric_col not in df.columns:
                    continue
                
                fig = go.Figure()
                normal_df = df[~anomaly_mask]
                if data_type in ['steps', 'activity']:
                    fig.add_trace(go.Bar(x=normal_df['timestamp'], y=normal_df[metric_col], name='Normal', marker_color='#7c3aed', opacity=0.7))
                else:
                    fig.add_trace(go.Scatter(x=normal_df['timestamp'], y=normal_df[metric_col], mode='lines', name='Normal', line=dict(color='#7c3aed', width=3)))
                
                anomaly_df = df[anomaly_mask]
                fig.add_trace(go.Scatter(x=anomaly_df['timestamp'], y=anomaly_df[metric_col], mode='markers', name='Anomalies', marker=dict(color='#ef4444', size=14, symbol='x')))
                fig.update_layout(
                    title=f"{data_type.replace('_', ' ').title()} with Anomalies",
                    xaxis_title="Time", yaxis_title=metric_col.replace('_', ' ').title(),
                    hovermode='x unified', height=500,
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#fafafa')
                )
                st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_heatmap_view(data: Dict):
        """Fixed heatmap - more visible"""
        st.subheader("🔥 Anomaly Heatmap")
        for data_type, df in data.get('data_with_anomalies', {}).items():
            if 'timestamp' not in df.columns:
                continue
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and '_reason' not in col.lower()]
            if not anomaly_cols:
                continue
            anomaly_mask = df[anomaly_cols].any(axis=1)
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"🗓️ {data_type.replace('_', ' ').title()} Heatmap", expanded=True):
                anomaly_df = df[anomaly_mask].copy()
                anomaly_df['date'] = pd.to_datetime(anomaly_df['timestamp']).dt.date
                anomaly_df['hour'] = pd.to_datetime(anomaly_df['timestamp']).dt.hour
                heatmap_data = anomaly_df.groupby(['date', 'hour']).size().reset_index(name='count')
                pivot_data = heatmap_data.pivot(index='hour', columns='date', values='count').fillna(0)
                
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=[str(d) for d in pivot_data.columns],
                    y=pivot_data.index,
                    colorscale='Purples',
                    colorbar=dict(title="Count"),
                    hovertemplate='Date: %{x}<br>Hour: %{y}<br>Anomalies: %{z}<extra></extra>'
                ))
                fig_heat.update_layout(
                    title=f"Anomaly Concentration - {data_type.title()}", 
                    xaxis_title="Date", 
                    yaxis_title="Hour of Day",
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    height=600,
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(20,20,30,0.8)',
                    font=dict(color='#fafafa', size=12)
                )
                st.plotly_chart(fig_heat, use_container_width=True)
    
    @staticmethod
    def create_hourly_distribution(insights: Dict):
        """Hourly patterns"""
        st.subheader("⏰ Hourly Anomaly Patterns")
        for data_type, pattern in insights.get('hourly_patterns', {}).items():
            if pattern.get('distribution'):
                with st.expander(f"📊 {data_type.replace('_', ' ').title()} Hourly Distribution"):
                    hours = list(pattern['distribution'].keys())
                    counts = list(pattern['distribution'].values())
                    fig = go.Figure(data=[go.Bar(x=hours, y=counts, marker_color='#7c3aed')])
                    fig.update_layout(
                        title=f"Anomalies by Hour - {data_type.title()}",
                        xaxis_title="Hour", yaxis_title="Count", height=400,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#fafafa')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Peak Hour:** {pattern['peak_hour']}:00 with {pattern['peak_count']} anomalies")
    
    @staticmethod
    def create_day_of_week_analysis(insights: Dict):
        """Day of week"""
        st.subheader("📅 Day of Week Patterns")
        for data_type, pattern in insights.get('day_of_week_patterns', {}).items():
            if pattern.get('distribution'):
                with st.expander(f"📊 {data_type.replace('_', ' ').title()} Day of Week"):
                    days = list(pattern['distribution'].keys())
                    counts = list(pattern['distribution'].values())
                    fig = go.Figure(data=[go.Bar(x=days, y=counts, marker_color='#6d28d9')])
                    fig.update_layout(
                        title=f"Anomalies by Day - {data_type.title()}",
                        xaxis_title="Day", yaxis_title="Count", height=400,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#fafafa')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"**Peak Day:** {pattern['peak_day']} with {pattern['peak_count']} anomalies")
    
    @staticmethod
    def create_3d_scatter(data: Dict):
        """Fixed 3D scatter"""
        st.subheader("🎲 3D Anomaly Visualization")
        for data_type, df in data.get('data_with_anomalies', {}).items():
            if 'timestamp' not in df.columns:
                continue
            anomaly_cols = [col for col in df.columns if 'anomaly' in col.lower() and '_reason' not in col.lower()]
            if not anomaly_cols:
                continue
            anomaly_mask = df[anomaly_cols].any(axis=1)
            if anomaly_mask.sum() == 0:
                continue
            
            with st.expander(f"🎯 {data_type.replace('_', ' ').title()} 3D Plot", expanded=True):
                df_plot = df[anomaly_mask].head(500).copy()
                
                if len(df_plot) == 0:
                    st.warning("No anomaly data available for 3D visualization")
                    continue
                
                df_plot['hour'] = pd.to_datetime(df_plot['timestamp']).dt.hour
                df_plot['day'] = pd.to_datetime(df_plot['timestamp']).dt.day
                
                metric_cols = {'heart_rate': 'heart_rate', 'steps': 'step_count', 'activity': 'activity_level', 'pros': 'heart_rate'}
                metric_col = metric_cols.get(data_type)
                
                if metric_col and metric_col in df_plot.columns:
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
                            xaxis_title='Day of Month',
                            yaxis_title='Hour of Day',
                            zaxis_title=metric_col.replace('_', ' ').title(),
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
                    st.info(f"Showing {len(df_plot)} anomalies in 3D space")
                else:
                    st.warning(f"Metric column '{metric_col}' not found in data")
    
    @staticmethod
    def create_sunburst_chart(insights: Dict):
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
                
                for atype in info.get('anomaly_types', [])[:3]:
                    labels.append(atype['type'])
                    parents.append(data_type.replace('_', ' ').title())
                    values.append(atype['count'])
                    colors.append('#5b21b6')
        
        if len(labels) > 1:
            fig = go.Figure(go.Sunburst(
                labels=labels, parents=parents, values=values,
                marker=dict(colors=colors), hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            ))
            fig.update_layout(
                title="Anomaly Type Hierarchy", height=600,
                paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#fafafa', size=14)
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PDF EXPORT - FIXED (No Emoji Encoding)
# ============================================================================

class EnhancedPDFExporter:
    """PDF export without emojis"""
    
    @staticmethod
    def export_comprehensive_pdf(insights: Dict, anomaly_data: Dict) -> bytes:
        """Export PDF (emojis removed)"""
        if not PDF_AVAILABLE:
            return b"PDF generation unavailable. Install fpdf: pip install fpdf"
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 20)
        pdf.set_text_color(124, 58, 237)
        pdf.cell(0, 10, 'FitPulse Analytics - Executive Report', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'EXECUTIVE SUMMARY', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, f"Total Anomalies: {insights['summary']['total_anomalies']}")
        pdf.multi_cell(0, 6, f"Anomaly Rate: {insights['summary']['overall_anomaly_rate']:.2f}%")
        pdf.multi_cell(0, 6, f"Risk Level: {insights['risk_assessment']['level']}")
        pdf.multi_cell(0, 6, f"Health Score: {insights['health_score']['score']}/100 ({insights['health_score']['grade']})")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'SEVERITY DISTRIBUTION', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, f"High Risk: {insights['severity_distribution']['high']}")
        pdf.multi_cell(0, 6, f"Medium Risk: {insights['severity_distribution']['medium']}")
        pdf.multi_cell(0, 6, f"Low Risk: {insights['severity_distribution']['low']}")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'ANOMALIES BY DATA TYPE', 0, 1)
        pdf.set_font('Arial', '', 11)
        for data_type, info in insights['by_data_type'].items():
            pdf.multi_cell(0, 6, f"{data_type.upper()}: {info['anomaly_count']} anomalies ({info['anomaly_percentage']:.1f}%)")
            if info.get('anomaly_types'):
                for atype in info['anomaly_types'][:5]:
                    pdf.multi_cell(0, 6, f"  - {atype['type']}: {atype['count']}")
        pdf.ln(5)
        
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'KEY RECOMMENDATIONS', 0, 1)
        pdf.set_font('Arial', '', 11)
        for i, rec in enumerate(insights['recommendations'][:15], 1):
            # Remove emojis from recommendations
            clean_rec = remove_special_chars(rec)
            try:
                pdf.multi_cell(0, 6, f"{i}. {clean_rec}")
            except:
                pdf.multi_cell(0, 6, f"{i}. [Recommendation text encoding error]")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'HOURLY PATTERNS', 0, 1)
        pdf.set_font('Arial', '', 11)
        for data_type, pattern in insights.get('hourly_patterns', {}).items():
            if pattern['peak_hour'] is not None:
                pdf.multi_cell(0, 6, f"{data_type.title()}: Peak at {pattern['peak_hour']}:00 ({pattern['peak_count']} anomalies)")
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'DAY OF WEEK PATTERNS', 0, 1)
        pdf.set_font('Arial', '', 11)
        for data_type, pattern in insights.get('day_of_week_patterns', {}).items():
            if pattern['peak_day']:
                pdf.multi_cell(0, 6, f"{data_type.title()}: Peak on {pattern['peak_day']} ({pattern['peak_count']} anomalies)")
        
        return pdf.output(dest='S').encode('latin-1')

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="FitPulse Dashboard", page_icon="💜", layout="wide")
    apply_purple_theme()
    
    st.title("💜 Milestone 4: Enhanced Executive Dashboard")
    st.markdown("**20+ Key Insights | 15+ Visualizations | Comprehensive Analytics**")
    
    st.sidebar.header("⚙️ Configuration")
    show_timeline = st.sidebar.checkbox("📅 Timeline View", value=True)
    show_heatmap = st.sidebar.checkbox("🔥 Heatmap Analysis", value=True)
    show_hourly = st.sidebar.checkbox("⏰ Hourly Patterns", value=True)
    show_dow = st.sidebar.checkbox("📅 Day of Week", value=True)
    show_3d = st.sidebar.checkbox("🎲 3D Visualization", value=True)
    show_sunburst = st.sidebar.checkbox("☀️ Sunburst Chart", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💜 Dashboard Features")
    st.sidebar.markdown("""
    - 📊 20+ Key Insights
    - 📈 15+ Visualizations
    - 💡 Personalized Recommendations
    - 📥 Enhanced PDF Export
    - 🎯 Risk Assessment
    - ⏰ Temporal Analysis
    - 📅 Pattern Discovery
    """)
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = AnomalyDataLoader()
    if 'insights_gen' not in st.session_state:
        st.session_state.insights_gen = ComprehensiveInsightsGenerator()
    if 'viz_engine' not in st.session_state:
        st.session_state.viz_engine = AdvancedVisualizationEngine()
    if 'pdf_exporter' not in st.session_state:
        st.session_state.pdf_exporter = EnhancedPDFExporter()
    
    st.markdown("---")
    st.header("1️⃣ Loading Anomaly Detection Results")
    with st.spinner("🔄 Loading data..."):
        anomaly_data = st.session_state.data_loader.load_milestone3_results()
    
    if not anomaly_data.get('data_with_anomalies'):
        st.error("❌ No data found! Please ensure data files exist in data/input/")
        st.info("💡 Tip: Run Milestone 3 first, then return here.")
        st.stop()
    
    st.success(f"✅ Loaded {len(anomaly_data['data_with_anomalies'])} dataset(s)")
    
    st.markdown("---")
    st.header("2️⃣ Comprehensive Insights Analysis")
    with st.spinner("🔮 Generating 20+ insights..."):
        insights = st.session_state.insights_gen.generate_detailed_insights(anomaly_data)
    
    st.session_state.insights_gen.display_executive_dashboard(insights)
    
    st.markdown("---")
    st.header("3️⃣ Advanced Visualizations")
    
    tabs = st.tabs(["📅 Timeline", "🔥 Heatmap", "⏰ Hourly", "📅 Day of Week", "🎲 3D", "☀️ Sunburst"])
    
    with tabs[0]:
        if show_timeline:
            st.session_state.viz_engine.create_timeline_view(anomaly_data)
    
    with tabs[1]:
        if show_heatmap:
            st.session_state.viz_engine.create_heatmap_view(anomaly_data)
    
    with tabs[2]:
        if show_hourly:
            st.session_state.viz_engine.create_hourly_distribution(insights)
    
    with tabs[3]:
        if show_dow:
            st.session_state.viz_engine.create_day_of_week_analysis(insights)
    
    with tabs[4]:
        if show_3d:
            st.session_state.viz_engine.create_3d_scatter(anomaly_data)
    
    with tabs[5]:
        if show_sunburst:
            st.session_state.viz_engine.create_sunburst_chart(insights)
    
    st.markdown("---")
    st.header("4️⃣ Export Reports")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Export Enhanced PDF Report", use_container_width=True):
            if PDF_AVAILABLE:
                try:
                    pdf_data = st.session_state.pdf_exporter.export_comprehensive_pdf(insights, anomaly_data)
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_data,
                        file_name=f"fitpulse_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
            else:
                st.warning("⚠️ PDF unavailable. Install: pip install fpdf")
    
    with col2:
        if st.button("📋 Export JSON Data", use_container_width=True):
            json_data = {
                'insights': insights,
                'generated_at': datetime.now().isoformat(),
                'risk_level': insights['risk_assessment']['level'],
                'health_score': insights['health_score']
            }
            json_str = json.dumps(json_data, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"fitpulse_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()