# baseline_and_plot.py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "processed_data"
HORIZON = 12  # next 12 timesteps (1 hour if 5-min intervals)
TIMESTEP_MIN = 5

# -------------------------
# LOAD DATA
# -------------------------
X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))
predictions = np.load(os.path.join(DATA_DIR, "predictions.npy"))

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def mape(y_true, y_pred):
    epsilon = 1e-5
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# -------------------------
# HISTORICAL AVERAGE BASELINE
# -------------------------
# Predict next timesteps using mean of input sequence per sensor
baseline_preds = np.mean(X_test, axis=1, keepdims=True)  # shape: (num_samples, 1, n_sensors)
baseline_preds = np.repeat(baseline_preds, HORIZON, axis=1)  # repeat for HORIZON timesteps

# -------------------------
# COMPUTE ERRORS PER HORIZON
# -------------------------
rmse_lstm, mae_lstm, mape_lstm = [], [], []
rmse_baseline, mae_baseline, mape_baseline = [], [], []

for h in range(HORIZON):
    y_true = Y_test[:, h, :]
    y_lstm = predictions[:, h, :]
    y_base = baseline_preds[:, h, :]

    rmse_lstm.append(np.sqrt(mean_squared_error(y_true, y_lstm)))
    mae_lstm.append(mean_absolute_error(y_true, y_lstm))
    mape_lstm.append(mape(y_true, y_lstm))

    rmse_baseline.append(np.sqrt(mean_squared_error(y_true, y_base)))
    mae_baseline.append(mean_absolute_error(y_true, y_base))
    mape_baseline.append(mape(y_true, y_base))

# -------------------------
# PRINT COMPARISON
# -------------------------
print("Horizon | RMSE LSTM | RMSE Baseline | MAE LSTM | MAE Baseline | MAPE LSTM | MAPE Baseline")
for h in range(HORIZON):
    print(f"{(h+1)*TIMESTEP_MIN:>6} | {rmse_lstm[h]:.3f}      | {rmse_baseline[h]:.3f}         | "
          f"{mae_lstm[h]:.3f}     | {mae_baseline[h]:.3f}        | {mape_lstm[h]:.2f}%     | {mape_baseline[h]:.2f}%")

# -------------------------
# PLOT ERROR VS HORIZON
# -------------------------
horizons = np.arange(1, HORIZON+1) * TIMESTEP_MIN

plt.figure(figsize=(10,6))
plt.plot(horizons, rmse_lstm, marker='o', label='LSTM RMSE')
plt.plot(horizons, rmse_baseline, marker='x', label='Baseline RMSE')
plt.plot(horizons, mae_lstm, marker='o', linestyle='--', label='LSTM MAE')
plt.plot(horizons, mae_baseline, marker='x', linestyle='--', label='Baseline MAE')
plt.xlabel("Forecast Horizon (min)")
plt.ylabel("Error")
plt.title("Traffic Prediction Error vs Forecast Horizon")
plt.grid(True)
plt.legend()
plt.show()
