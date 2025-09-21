import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate complete dataset
np.random.seed(42)
start_date = datetime(2024, 1, 15, 6, 0, 0)  
timestamps = pd.date_range(start=start_date, periods=10081, freq='1min')

heart_rate_data = []
for i, timestamp in enumerate(timestamps):
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Realistic patterns
    if 6 <= hour <= 8:
        base_hr = 70 + np.random.normal(15, 5)
    elif 9 <= hour <= 11:
        base_hr = 75 + np.random.normal(20, 8)
    elif 12 <= hour <= 14:
        base_hr = 72 + np.random.normal(18, 6)
    elif 15 <= hour <= 18:
        base_hr = 78 + np.random.normal(25, 10)
    elif 19 <= hour <= 21:
        base_hr = 74 + np.random.normal(22, 8)
    elif 22 <= hour <= 23:
        base_hr = 68 + np.random.normal(10, 4)
    else:
        base_hr = 60 + np.random.normal(8, 3)
    
    # Add workouts (Tue, Thu, Sat 5-6 PM)
    if day_of_week in [1, 3, 5] and 17 <= hour <= 18:
        if i % 60 < 45:  # 45-min workout
            base_hr += np.random.normal(40, 15)
    
    # Weekend activities
    if day_of_week in [5, 6] and 10 <= hour <= 12:
        base_hr += np.random.normal(20, 8)
    
    base_hr += np.random.normal(0, 3)
    heart_rate = max(50, min(180, round(base_hr, 1)))
    
    heart_rate_data.append({
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'heart_rate': heart_rate,
        'confidence': round(np.random.uniform(0.85, 1.0), 6)
    })

# Save to CSV
df = pd.DataFrame(heart_rate_data)
df.to_csv('complete_heart_rate_data.csv', index=False)
print(f"✅ Created {len(df)} records!")
