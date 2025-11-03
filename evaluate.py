# evaluate.py
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# CONFIG
# -------------------------
DATA_DIR = "processed_data"
HORIZON = 12  # number of predicted timesteps (matches train/inference)
TIMESTEP_MIN = 5  # each timestep in minutes (for labeling: 15, 30, 60 min)

# -------------------------
# LOAD DATA
# -------------------------
Y_test = np.load(os.path.join(DATA_DIR, "Y_test.npy"))          # true values
predictions = np.load(os.path.join(DATA_DIR, "predictions.npy"))  # predicted values

print(f"Y_test shape: {Y_test.shape}, predictions shape: {predictions.shape}")

# -------------------------
# METRIC FUNCTIONS
# -------------------------
def mape(y_true, y_pred):
    epsilon = 1e-5  # avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# -------------------------
# EVALUATE PER HORIZON
# -------------------------
rmse_list = []
mae_list = []
mape_list = []

for h in range(HORIZON):
    y_true_h = Y_test[:, h, :]
    y_pred_h = predictions[:, h, :]
    
    rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
    mae = mean_absolute_error(y_true_h, y_pred_h)
    mape_val = mape(y_true_h, y_pred_h)
    
    rmse_list.append(rmse)
    mae_list.append(mae)
    mape_list.append(mape_val)
    
    print(f"Horizon {(h+1)*TIMESTEP_MIN} min | RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape_val:.2f}%")

# -------------------------
# AVERAGE ACROSS HORIZON
# -------------------------
print("\nAverage metrics across all horizons:")
print(f"RMSE: {np.mean(rmse_list):.4f} | MAE: {np.mean(mae_list):.4f} | MAPE: {np.mean(mape_list):.2f}%")

# -------------------------
# SAVE REPORT
# -------------------------
report_path = os.path.join(DATA_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    for h in range(HORIZON):
        f.write(f"Horizon {(h+1)*TIMESTEP_MIN} min | RMSE: {rmse_list[h]:.4f} | MAE: {mae_list[h]:.4f} | MAPE: {mape_list[h]:.2f}%\n")
    f.write(f"\nAverage metrics across all horizons:\nRMSE: {np.mean(rmse_list):.4f} | MAE: {np.mean(mae_list):.4f} | MAPE: {np.mean(mape_list):.2f}%\n")

print(f"✅ Evaluation report saved to {report_path}")
