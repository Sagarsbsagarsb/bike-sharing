import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# ---------- Data Generation ----------
def generate_data(n=10000, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        "temperature": np.random.normal(70, 15, n),   # deg C
        "vibration": np.random.normal(5, 2, n),       # mm/s
        "pressure": np.random.normal(100, 20, n),     # psi
        "anomaly_score": np.random.uniform(0, 1, n),
        "ride_duration": np.random.exponential(60, n) # mins
    })

    brakes_prob = (data["temperature"]/100 + data["ride_duration"]/200) * 0.6
    tires_prob = (1 - data["pressure"]/200 + data["temperature"]/120) * 0.5
    chain_prob = (data["vibration"]/10 + data["ride_duration"]/300) * 0.7
    gears_prob = (data["vibration"]/8 + data["ride_duration"]/250) * 0.5
    electronics_prob = (data["anomaly_score"] + data["temperature"]/150) * 0.8

    data["brakes_replaced"] = np.random.binomial(1, np.clip(brakes_prob, 0, 1))
    data["tires_replaced"] = np.random.binomial(1, np.clip(tires_prob, 0, 1))
    data["chain_replaced"] = np.random.binomial(1, np.clip(chain_prob, 0, 1))
    data["gears_replaced"] = np.random.binomial(1, np.clip(gears_prob, 0, 1))
    data["electronics_replaced"] = np.random.binomial(1, np.clip(electronics_prob, 0, 1))

    return data

# ---------- PyTorch Model ----------
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

# ---------- Training ----------
def train_model(data, epochs=50, lr=0.001):
    features = ["temperature","vibration","pressure","anomaly_score","ride_duration"]
    labels = ["brakes_replaced","tires_replaced","chain_replaced","gears_replaced","electronics_replaced"]

    X = data[features].values
    y = data[labels].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    model = MLPModel(X_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val), y_val).item()
            print(f"Epoch {epoch}: Train Loss {loss.item():.4f}, Val Loss {val_loss:.4f}")

    return model, scaler

# ---------- Run Training ----------
if __name__ == "__main__":
    data = generate_data(10000)
    model, scaler = train_model(data, epochs=50)

    # save model + scaler
    torch.save(model.state_dict(), "bike_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model and scaler saved.")
