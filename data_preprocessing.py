import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # One-hot encode categorical variables
    categorical_cols = ['device_type', 'location', 'weather_condition']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['power_watts', 'energy_kwh', 'room_temp', 'outdoor_temp', 'humidity', 'light_level', 'wifi_signal', 'electricity_price']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Prepare features and targets
    features = df.drop(['timestamp', 'user_id', 'energy_kwh', 'anomaly_score'], axis=1)
    energy_target = df['energy_kwh']
    user_target = pd.get_dummies(df['user_id'])
    anomaly_target = df['anomaly_score']
    
    return train_test_split(features, energy_target, user_target, anomaly_target, test_size=0.2, random_state=42)

def create_sequences(features, energy_target, user_target, anomaly_target, seq_length):
    X, y_energy, y_user, y_anomaly = [], [], [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length].values)
        y_energy.append(energy_target[i+seq_length])
        y_user.append(user_target[i+seq_length].values)
        y_anomaly.append(anomaly_target[i+seq_length])
    return np.array(X), np.array(y_energy), np.array(y_user), np.array(y_anomaly)
