import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Model Definition ---
class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# --- Load Model + Scaler ---
scaler = joblib.load("scaler.pkl")
model = MLPModel(5, 5)
model.load_state_dict(torch.load("bike_model.pth", map_location=torch.device("cpu")))
model.eval()

# --- Streamlit UI ---
st.title("🚲 Bike Part Failure Predictor")

st.subheader("📊 Why these features?")
st.markdown("""
We selected these **key sensor features** because they directly affect e-Bike health:
- **Vibration (g-force):** Excessive vibration signals mechanical issues like loose parts or imbalanced wheels.
- **Motor Temperature (°C):** High temperatures accelerate wear and can trigger sudden motor failures.
- **Ride Duration (mins):** Longer rides increase cumulative stress, raising the chance of overheating and component fatigue.
""")



temperature = st.slider("Temperature (°C)", 0, 120, 70)
vibration = st.slider("Vibration (mm/s)", 0, 20, 5)
pressure = st.slider("Tire Pressure (psi)", 50, 200, 100)
anomaly_score = st.slider("Anomaly Score", 0.0, 1.0, 0.5)
ride_duration = st.slider("Ride Duration (mins)", 0, 300, 60)

if st.button("🔮 Predict Failure Risks"):
    # prepare input
    X = np.array([[temperature, vibration, pressure, anomaly_score, ride_duration]])
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    
    
    

    # get predictions
    with torch.no_grad():
        probs = model(X_tensor).numpy()[0]

    parts = ["Brakes", "Tires", "Chain", "Gears", "Electronics"]

    st.subheader("Prediction Results")
    for p, prob in zip(parts, probs):
        st.write(f"**{p} Failure Risk:** {prob*100:.1f}%")

    # --- Show Bar Graph ---
    st.subheader("📊 Risk Visualization")
    fig, ax = plt.subplots()
    ax.bar(parts, probs * 100, color="orange")
    ax.set_ylabel("Failure Probability (%)")
    ax.set_ylim(0, 100)
    st.pyplot(fig)

