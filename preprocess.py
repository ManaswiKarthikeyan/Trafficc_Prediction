# preprocess.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------
# CONFIG
# -------------------------
RAW_CSV = "data/metr_la_speed.csv"
OUTPUT_DIR = "processed_data"
LOOKBACK = 12   # past 1 hour if 5-min intervals
HORIZON = 12    # next 1 hour
SAMPLING_MIN = 5
USE_EXOG = True
VAL_RATIO = 0.1
TEST_RATIO = 0.1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# LOAD CSV
# -------------------------
df_raw = pd.read_csv(RAW_CSV)
print("Columns in CSV:", df_raw.columns)

# Generate timestamp if missing
if 'timestamp' not in df_raw.columns:
    n_rows = df_raw.shape[0]
    timestamps = pd.date_range(start='2012-01-01', periods=n_rows, freq=f'{SAMPLING_MIN}min')
    df_raw['timestamp'] = timestamps
df_raw = df_raw.set_index('timestamp')

# -------------------------
# Select numeric columns only
# -------------------------
df_numeric = df_raw.select_dtypes(include='number')

# -------------------------
# Resample and fill missing values
# -------------------------
df_numeric = df_numeric.resample(f'{SAMPLING_MIN}min').mean().fillna(method='ffill').fillna(method='bfill')

# -------------------------
# Exogenous features (optional)
# -------------------------
if USE_EXOG:
    df_exog = pd.DataFrame(index=df_numeric.index)
    df_exog['tod_sin'] = np.sin(2*np.pi*(df_numeric.index.hour*60+df_numeric.index.minute)/(24*60))
    df_exog['tod_cos'] = np.cos(2*np.pi*(df_numeric.index.hour*60+df_numeric.index.minute)/(24*60))
    dow_ohe = pd.get_dummies(df_numeric.index.dayofweek).astype(float)
    df_exog = pd.concat([df_exog,dow_ohe],axis=1)
    exog_vals = df_exog.values
else:
    exog_vals = None

# -------------------------
# Scale sensor values
# -------------------------
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_numeric.values)
np.savez(os.path.join(OUTPUT_DIR,"scaler.npz"), mean=scaler.mean_, var=scaler.var_)

# -------------------------
# Create sequences
# -------------------------
def make_sequences(data_scaled, exog_vals, lookback, horizon):
    X, X_exog, Y = [], [], []
    for i in range(len(data_scaled)-lookback-horizon+1):
        X.append(data_scaled[i:i+lookback])
        X_exog.append(exog_vals[i:i+lookback] if exog_vals is not None else np.zeros((lookback,0)))
        Y.append(data_scaled[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(X_exog), np.array(Y)

X, X_exog, Y = make_sequences(data_scaled, exog_vals, LOOKBACK, HORIZON)

# -------------------------
# Train / Val / Test split
# -------------------------
n_samples = X.shape[0]
n_test = int(n_samples*TEST_RATIO)
n_val  = int(n_samples*VAL_RATIO)
n_train = n_samples - n_val - n_test

np.save(os.path.join(OUTPUT_DIR,"X_train.npy"), X[:n_train])
np.save(os.path.join(OUTPUT_DIR,"Y_train.npy"), Y[:n_train])
np.save(os.path.join(OUTPUT_DIR,"X_val.npy"), X[n_train:n_train+n_val])
np.save(os.path.join(OUTPUT_DIR,"Y_val.npy"), Y[n_train:n_train+n_val])
np.save(os.path.join(OUTPUT_DIR,"X_test.npy"), X[n_train+n_val:])
np.save(os.path.join(OUTPUT_DIR,"Y_test.npy"), Y[n_train+n_val:])

np.save(os.path.join(OUTPUT_DIR,"Xexog_train.npy"), X_exog[:n_train])
np.save(os.path.join(OUTPUT_DIR,"Xexog_val.npy"), X_exog[n_train:n_train+n_val])
np.save(os.path.join(OUTPUT_DIR,"Xexog_test.npy"), X_exog[n_train+n_val:])

print("✅ Preprocessing completed. Processed data saved in 'processed_data/'")
