import numpy as np

def decision_engine(energy_pred, user_pred, anomaly_pred, thresholds):
    actions = []
    
    # Energy consumption decision
    if energy_pred > thresholds['energy']:
        actions.append("High energy consumption detected. Consider turning off unused devices.")
    
    # User identification
    user_id = np.argmax(user_pred)
    actions.append(f"User {user_id} identified as the current active user.")
    
    # Anomaly detection
    if anomaly_pred > thresholds['anomaly']:
        actions.append("Unusual energy usage pattern detected. Investigating...")
    
    return actions
