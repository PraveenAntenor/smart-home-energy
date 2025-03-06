import torch
import asyncio
from model.transformer import SmartHomeTransformer
from utils.data_processing import load_and_preprocess_data, create_sequences
from utils.data_preprocessing import load_and_preprocess_data, create_sequences
from utils.decision_engine import decision_engine

# Load the trained model
@st.cache_resource
def load_model():
    input_dim = 20  # Update this based on your actual input dimension
    input_dim = 100  # Update this based on your actual input dimension after preprocessing
    num_users = 5  # Update this based on your actual number of users
    model = SmartHomeTransformer(input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_users=num_users)
    model.load_state_dict(torch.load("model/smart_home_model.pth"))
@@ -33,33 +33,45 @@ def load_model():
    X_seq, _, _, _ = create_sequences(X, pd.Series([0]*len(X)), pd.DataFrame([[0]*5]*len(X)), pd.Series([0]*len(X)), seq_length=60)

    # Make predictions
    async def make_predictions():
    async def make_predictions(input_data):
        with torch.no_grad():
            energy_pred, user_pred, anomaly_pred = model(torch.FloatTensor(X_seq))
            energy_pred, user_pred, anomaly_pred = model(torch.FloatTensor(input_data))
        return energy_pred[:, -1, :], user_pred[:, -1, :], anomaly_pred[:, -1, :]

    # Decision engine
    async def get_decisions(energy_pred, user_pred, anomaly_pred):
        thresholds = {'energy': 0.8, 'anomaly': 0.7}
        return decision_engine(energy_pred.item(), user_pred.numpy(), anomaly_pred.item(), thresholds)
    async def get_decisions(energy_pred, user_pred, anomaly_pred, device_status, device_type):
        thresholds = {
            'energy': 0.8,
            'anomaly': 0.7,
            'user_presence': 0.5,
            'standby': 0.1
        }
        return decision_engine(energy_pred.item(), user_pred.numpy(), anomaly_pred.item(), thresholds, device_status, device_type)

    # Run predictions and decision engine concurrently
    async def process_data():
        energy_pred, user_pred, anomaly_pred = await make_predictions()
        decisions = await get_decisions(energy_pred[0], user_pred[0], anomaly_pred[0])
    async def process_data(input_data, device_status, device_type):
        energy_pred, user_pred, anomaly_pred = await make_predictions(input_data)
        decisions = await get_decisions(energy_pred[0], user_pred[0], anomaly_pred[0], device_status, device_type)
        return energy_pred[0].item(), user_pred[0].numpy(), anomaly_pred[0].item(), decisions

    # Run the async functions
    energy_pred, user_pred, anomaly_pred, decisions = asyncio.run(process_data())
    # Run the async functions for each device
    results = []
    for i in range(len(X_seq)):
        device_status = data.iloc[i]['operational_status']
        device_type = data.iloc[i]['device_type']
        results.append(asyncio.run(process_data(X_seq[i:i+1], device_status, device_type)))

    # Display results
    st.write("Energy Consumption Prediction:", energy_pred)
    st.write("User Identification:", np.argmax(user_pred))
    st.write("Anomaly Score:", anomaly_pred)
    
    st.write("Decisions:")
    for decision in decisions:
        st.write(f"- {decision}")
    for i, (energy_pred, user_pred, anomaly_pred, decisions) in enumerate(results):
        st.write(f"Device {i+1}:")
        st.write(f"Energy Consumption Prediction: {energy_pred:.2f}")
        st.write(f"User Identification: {np.argmax(user_pred)}")
        st.write(f"Anomaly Score: {anomaly_pred:.2f}")
        
        st.write("Decisions:")
        for decision in decisions:
            st.write(f"- {decision}")
        st.write("---")

else:
    st.write("Please upload a CSV file to start the analysis.")
