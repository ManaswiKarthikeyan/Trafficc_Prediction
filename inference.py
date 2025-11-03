# inference.py
import torch
import torch.nn as nn
import numpy as np
import os

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "processed_data"
HORIZON = 12
HIDDEN = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(DEVICE)

n_sensors = X_test.shape[2]

# -------------------------
# MODEL DEFINITION (must match train.py)
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, n_sensors, hidden, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_sensors * horizon)
        self.n_sensors = n_sensors
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]).view(-1, self.horizon, self.n_sensors)
        return out

# -------------------------
# LOAD TRAINED MODEL
# -------------------------
model = LSTMModel(n_sensors, HIDDEN, HORIZON).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_lstm.pt"), map_location=DEVICE))
model.eval()
print("✅ Loaded trained model.")

# -------------------------
# PREDICTION
# -------------------------
with torch.no_grad():
    predictions = model(X_test)  # shape: (num_samples, HORIZON, n_sensors)

# Convert to numpy and save
predictions = predictions.cpu().numpy()
np.save(os.path.join(DATA_DIR, "predictions.npy"), predictions)
print(f"✅ Predictions saved to {DATA_DIR}/predictions.npy")
