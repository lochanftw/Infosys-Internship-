"""
FitPulse Data Format Mastery Dashboard - Simple Version
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append('.')

try:
    from src.data_loading.universal_loader import load_fitness_data
    LOADER_AVAILABLE = True
except ImportError:
    LOADER_AVAILABLE = False

def main():
    st.set_page_config(
        page_title="FitPulse Data Format Mastery",
        page_icon="💓"
    )
    
    st.title("🏃‍♂️ FitPulse Data Format Mastery")
    st.markdown("**Task 2: Universal Data Loader & Format Analysis**")
    
    # Sidebar
    st.sidebar.header("📊 Project Information")
    st.sidebar.markdown("""
    **Assignment:** Infosys Task 2
    **Features:** Universal Data Loader, Multi-format Support
    **Author:** Your Name
    """)
    
    if not LOADER_AVAILABLE:
        st.error("❌ Universal loader not found!")
        return
    
    st.success("✅ Universal loader is working!")
    
    # File upload section
    st.header("📁 Universal Data Loader")
    uploaded_file = st.file_uploader(
        "Upload fitness data",
        type=['csv', 'json']
    )
    
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = load_fitness_data(temp_path)
        
        if not df.empty:
            st.success(f"✅ Successfully loaded {len(df)} records")
            st.dataframe(df.head(10))
        else:
            st.error("❌ Failed to load data")
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Sample data section
    st.header("📊 Sample Data Preview")
    
    csv_file = "data/raw/sample_heart_rate.csv"
    if os.path.exists(csv_file):
        df = load_fitness_data(csv_file)
        if not df.empty:
            st.subheader("Heart Rate CSV")
            st.dataframe(df.head())
    
    json_file = "data/raw/sample_fitness_data.json"
    if os.path.exists(json_file):
        df = load_fitness_data(json_file)
        if not df.empty:
            st.subheader("Fitness JSON")
            st.dataframe(df.head())

if __name__ == "__main__":
    main()
