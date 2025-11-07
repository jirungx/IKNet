from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

def compute_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def print_metrics(metrics, label=""):
    print(f"[{label}] RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, SMAPE: {metrics['SMAPE']:.4f}, R2: {metrics['R2']:.4f}")