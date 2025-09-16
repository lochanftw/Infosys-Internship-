# =========================
# STREAMLIT APP: ASSIGNMENT_INFOSYS
# =========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import json
import os

# =========================
# Load CSS
# =========================
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ CSS file not found at: {file_name}")

# ✅ Load style.css from assets folder
local_css("assets/style.css")

# =========================
# Load Logo
# =========================
try:
    logo = Image.open("assets/logo.png")
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.write("📌 Logo not found. Add `assets/logo.png`")

# =========================
# Sidebar Info
# =========================
st.sidebar.title("Infosys Dashboard")
st.sidebar.write("Welcome! Upload your fitness CSV/JSON data to see previews, summary, and preprocessing.")
st.sidebar.markdown("---")

# =========================
# Main Title
# =========================
st.title("Data Dashboard")

# =========================
# File Upload
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV or JSON file", type=["csv", "json"])

if uploaded_file:
    df = None

    try:
        # --- Handle CSV ---
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        # --- Handle JSON ---
        elif uploaded_file.name.endswith(".json"):
            uploaded_file.seek(0)
            raw_json = json.load(uploaded_file)
            if 'heart_rate_data' in raw_json:
                df = pd.json_normalize(raw_json['heart_rate_data'])
            else:
                df = pd.json_normalize(raw_json)

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()

    # =========================
    # If Data Loaded
    # =========================
    if df is not None and not df.empty:

        # Data Preview
        st.subheader("👀 Data Preview")
        num_rows = st.slider("Select number of rows to view", 5, 20, 10)
        st.dataframe(df.head(num_rows), use_container_width=True)

        # Dataset Summary
        if st.checkbox("📊 Show summary"):
            st.subheader("Summary Statistics")
            st.write(df.describe(include="all"))
            st.write("Missing values per column:")
            st.write(df.isnull().sum())

        # =========================
        # Preprocessing Options
        # =========================
        st.subheader("⚙️ Preprocessing Options")
        resample_rule = st.selectbox("Resample frequency (for time series)", ["1min", "5min", "15min"])
        fill_method = st.selectbox("Fill missing values", ["ffill", "bfill", "none"])

        if st.button("🚀 Apply Preprocessing"):

            if 'timestamp' in df.columns:
                # Convert to datetime safely
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', infer_datetime_format=True)
                df = df.dropna(subset=['timestamp'])

                # Convert timestamps to UTC
                if df['timestamp'].dt.tz is None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')  # naive timestamps
                else:
                    df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')    # already tz-aware

                # Set timestamp index
                df = df.set_index('timestamp')

                # Resample numeric columns
                numeric_cols = df.select_dtypes(include='number').columns
                df[numeric_cols] = df[numeric_cols].resample(resample_rule).mean()

                # Fill missing values
                if fill_method in ['ffill', 'bfill']:
                    df[numeric_cols] = df[numeric_cols].fillna(method=fill_method)

                df = df.reset_index()
                st.success("✅ Preprocessing applied! (timestamps converted to UTC)")

                # Show processed data
                st.subheader("🔍 Processed Data")
                st.dataframe(df.head(num_rows), use_container_width=True)

                # Download processed CSV
                csv_download = df.copy()
                csv_download['timestamp'] = csv_download['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
                csv = csv_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="💾 Download Processed CSV",
                    data=csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                )

                # =========================
                # Heart Rate Chart
                # =========================
                hr_col = None
                for col in df.columns:
                    if 'bpm' in col.lower() or 'heart_rate' in col.lower():
                        hr_col = col
                        break

                if hr_col:
                    st.subheader("❤️ Heart Rate Chart")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    # Smooth using rolling average
                    df[hr_col + "_smooth"] = df[hr_col].rolling(window=3, min_periods=1).mean()
                    ax.plot(df['timestamp'], df[hr_col + "_smooth"], color="#6366f1", linewidth=2)
                    ax.set_xlabel("Timestamp")
                    ax.set_ylabel("Heart Rate (BPM)")
                    plt.xticks(rotation=30)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ No heart rate column found to plot chart.")

            else:
                st.warning("⚠️ No 'timestamp' column found. Preprocessing skipped.")

    else:
        st.warning("⚠️ Could not parse the file. Please check if it's a valid CSV/JSON.")
