# Traffic_Prediction
# Traffic Prediction ML Project

Predict short-term traffic flow (speeds/counts) using historical METR-LA data. Implements an LSTM model with multi-step forecasting, baseline comparison, and evaluation (RMSE, MAE, MAPE). Includes preprocessing, training, inference, and visualization of prediction errors.

---

## Folder Structure

TrafficPrediction/
│
├── data/ # Raw dataset (optional)
│ └── metr_la_speed.csv
│
├── processed_data/ # Processed data and outputs
│ ├── X_train.npy
│ ├── Y_train.npy
│ ├── X_val.npy
│ ├── Y_val.npy
│ ├── X_test.npy
│ ├── Y_test.npy
│ ├── scaler.npz
│ ├── best_lstm.pt # Trained model checkpoint
│ └── predictions.npy # LSTM predictions
│
├── preprocess.py # Data preprocessing script
├── train.py # Model training script
├── inference.py # Generate predictions
├── evaluate.py # Compute metrics per horizon
├── baseline_and_plot.py # Baseline comparison + error plot
└── evaluation_report.txt # Summary of evaluation metrics

## How to Run

1. **Preprocess the data:**
python preprocess.py

2.Train the LSTM model:
python train.py

3.Generate predictions on test set:
python inference.py

4.Evaluate predictions:
python evaluate.py

5.Compare with baseline and plot errors:
python baseline_and_plot.py

---
##Dependencies
Python >= 3.8
PyTorch
NumPy
Pandas
Scikit-learn
Matplotlib (for plots)
---
##Features
Multi-step traffic forecasting using LSTM
Baseline comparison using historical average
Evaluation with RMSE, MAE, and MAPE per forecast horizon
Error visualization per horizon
Fully reproducible pipeline from raw data to predictions
---
##Notes
Trained model checkpoint is saved in processed_data/best_lstm.pt
Predictions and evaluation report are included for review
Optional exogenous features (time-of-day, day-of-week) can be added for improved performance

yaml
Copy code
