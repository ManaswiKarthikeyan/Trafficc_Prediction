# train.py (with batching)
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "processed_data"
LOOKBACK = 12
HORIZON = 12
HIDDEN = 64
EPOCHS = 10       # increase for better accuracy
BATCH_SIZE = 64   # smaller if you still get memory issues
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD DATA
# -------------------------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
Y_train = np.load(os.path.join(DATA_DIR, "Y_train.npy"))
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
Y_val   = np.load(os.path.join(DATA_DIR, "Y_val.npy"))

print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
print(f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val, dtype=torch.float32)
Y_val   = torch.tensor(Y_val, dtype=torch.float32)

n_sensors = X_train.shape[2]

# -------------------------
# DATALOADERS
# -------------------------
train_dataset = TensorDataset(X_train, Y_train)
val_dataset   = TensorDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# MODEL DEFINITION
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

model = LSTMModel(n_sensors, HIDDEN, HORIZON).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.L1Loss()  # MAE

# -------------------------
# TRAINING LOOP
# -------------------------
print(f"Starting training on {len(train_dataset)} samples, {n_sensors} sensors...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = loss_fn(outputs, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            outputs = model(xb)
            loss = loss_fn(outputs, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# -------------------------
# SAVE MODEL
# -------------------------
os.makedirs(DATA_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_lstm.pt"))
print("✅ Training completed. Model saved as 'processed_data/best_lstm.pt'")
