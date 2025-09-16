import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# --- Data Loading Functions ---

def load_csv_data(file_path, data_type='heart_rate'):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        print("First 5 rows:")
        print(df.head())

        if 'timestamp' not in df.columns:
            raise ValueError("CSV must contain 'timestamp' column")

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None


def load_json_data(file_path, extract_type='heart_rate'):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        print(f"JSON file loaded. Available data types: {data.get('data_types', [])}")

        if extract_type == 'heart_rate':
            records = data.get('heart_rate_data', [])
        elif extract_type == 'steps':
            records = data.get('step_data', [])
        elif extract_type == 'sleep':
            records = data.get('sleep_data', [])
        else:
            raise ValueError(f"Unsupported extract_type: {extract_type}")

        df = pd.DataFrame(records)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        print(f"Extracted {len(df)} {extract_type} records")
        return df

    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
        return None
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return None


# --- Pipeline Class ---

class FitnessDataIngester:
    def __init__(self):
        self.supported_formats = ['.csv', '.json']
        self.supported_data_types = ['heart_rate', 'steps', 'sleep']

    def detect_file_format(self, file_path):
        return file_path.lower().split('.')[-1]

    def validate_data_type(self, data_type):
        if data_type not in self.supported_data_types:
            raise ValueError(f"Unsupported data type. Use: {self.supported_data_types}")

    def load_data(self, file_path, data_type='heart_rate'):
        self.validate_data_type(data_type)
        ext = '.' + self.detect_file_format(file_path)

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")

        if ext == '.csv':
            df = load_csv_data(file_path, data_type)
        else:
            df = load_json_data(file_path, data_type)

        if df is not None and not df.empty:
            df = self._standardize_dataframe(df, data_type)
            df = self._align_intervals(df, data_type)

        return df

    def _standardize_dataframe(self, df, data_type):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        if data_type == 'heart_rate':
            hr_cols = ['bpm', 'heart_rate_bpm', 'heartrate', 'hr']
            for col in hr_cols:
                if col in df.columns:
                    df['heart_rate'] = df[col]
                    break
        elif data_type == 'steps':
            step_cols = ['steps', 'step_count', 'steps_per_minute']
            for col in step_cols:
                if col in df.columns:
                    df['steps'] = df[col]
                    break

        print(f"Standardized {data_type} DataFrame shape: {df.shape}")
        return df

    def _align_intervals(self, df, data_type):
        df = df.set_index('timestamp')
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].resample('1min').mean()
        df = df.ffill()
        return df


# --- Data Quality Check Function ---

def perform_data_quality_check(df, data_type):
    report = {
        'total_records': len(df),
        'date_range': None,
        'missing_values': {},
        'data_issues': [],
        'quality_score': 0
    }

    if df.empty:
        report['data_issues'].append("No data found")
        return report

    if 'timestamp' in df.columns:
        report['date_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        }

        time_diffs = df['timestamp'].diff().dropna()
        large_gaps = time_diffs[time_diffs > timedelta(minutes=10)]
        if len(large_gaps) > 0:
            report['data_issues'].append(f"{len(large_gaps)} time gaps > 10 minutes")

    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            report['missing_values'][col] = {'count': missing, 'percentage': missing / len(df) * 100}

    if data_type == 'heart_rate' and 'heart_rate' in df.columns:
        hr_values = df['heart_rate'].dropna()
        report['data_issues'] += [f"{(hr_values < 30).sum()} heart rate values < 30 BPM"] if (hr_values < 30).sum() > 0 else []
        report['data_issues'] += [f"{(hr_values > 220).sum()} heart rate values > 220 BPM"] if (hr_values > 220).sum() > 0 else []

    score = 100 - len(report['data_issues']) * 10 - sum([v['percentage'] for v in report['missing_values'].values()])/2
    report['quality_score'] = max(0, score)
    return report


# --- Testing Pipeline ---

ingester = FitnessDataIngester()

# CSV
df_csv = ingester.load_data('sample_data/sample_heart_rate.csv', 'heart_rate')
print("\nCSV Data Sample:")
print(df_csv.head())
quality_csv = perform_data_quality_check(df_csv, 'heart_rate')
print(f"CSV Quality Score: {quality_csv['quality_score']}")

# JSON Heart Rate
df_json_hr = ingester.load_data('sample_data/sample_fitness_data.json', 'heart_rate')
print("\nJSON HR Sample:")
print(df_json_hr.head())
quality_json_hr = perform_data_quality_check(df_json_hr, 'heart_rate')
print(f"JSON HR Quality Score: {quality_json_hr['quality_score']}")

# JSON Steps
df_json_steps = ingester.load_data('sample_data/sample_fitness_data.json', 'steps')
print("\nJSON Steps Sample:")
print(df_json_steps.head())
quality_json_steps = perform_data_quality_check(df_json_steps, 'steps')
print(f"JSON Steps Quality Score: {quality_json_steps['quality_score']}")

# JSON Sleep
df_json_sleep = ingester.load_data('sample_data/sample_fitness_data.json', 'sleep')
print("\nJSON Sleep Sample:")
print(df_json_sleep.head())
quality_json_sleep = perform_data_quality_check(df_json_sleep, 'sleep')
print(f"JSON Sleep Quality Score: {quality_json_sleep['quality_score']}")
