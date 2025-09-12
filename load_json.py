# load_json.py
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

def create_sample_json_data():
    """Create sample fitness data in JSON format"""
    
    # Create comprehensive fitness data
    fitness_data = {
        "user_id": "user_123",
        "device": "Fitness Tracker Pro",
        "export_date": datetime.now().strftime('%Y-%m-%d'),
        "data_types": ["heart_rate", "steps", "sleep"],
        "heart_rate_data": [],
        "step_data": [],
        "sleep_data": []
    }
    
    # Generate heart rate data
    start_time = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    for i in range(60):  # 1 hour of data
        timestamp = start_time + timedelta(minutes=i)
        hr = 70 + np.random.normal(0, 8)
        fitness_data["heart_rate_data"].append({
            "timestamp": timestamp.isoformat(),
            "bpm": max(50, min(150, int(hr))),
            "confidence": np.random.uniform(0.8, 1.0)
        })
    
    # Generate step data  
    for i in range(60):
        timestamp = start_time + timedelta(minutes=i)
        steps = np.random.poisson(12)
        fitness_data["step_data"].append({
            "timestamp": timestamp.isoformat(),
            "steps": steps,
            "cadence": steps * 60 if steps > 0 else 0
        })
    
    # Generate sleep data (simulate 1-night sleep)
    sleep_start = datetime.now().replace(hour=23, minute=0, second=0, microsecond=0)
    for i in range(8*60):  # 8 hours, 1-minute intervals
        timestamp = sleep_start + timedelta(minutes=i)
        stage = np.random.choice(['awake', 'light', 'deep', 'rem'], p=[0.05, 0.6, 0.25, 0.1])
        fitness_data["sleep_data"].append({
            "timestamp": timestamp.isoformat(),
            "stage": stage
        })
    
    # Save to sample_data folder
    with open('sample_data/sample_fitness_data.json', 'w') as f:
        json.dump(fitness_data, f, indent=2)
    
    print("Created sample_fitness_data.json in sample_data/")
    return fitness_data

# Generate JSON
json_data = create_sample_json_data()
print(f"JSON contains {len(json_data['heart_rate_data'])} heart rate readings, "
      f"{len(json_data['step_data'])} steps records, "
      f"{len(json_data['sleep_data'])} sleep records")
