import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def split_by_rolling_window(df, train_years=3, test_years=1, start_year=2015, end_year=2024):
    """
    Split dataset into rolling train-test windows by year.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    windows = []

    for train_start in range(start_year, end_year - train_years - test_years + 3):
        train_end = train_start + train_years - 1
        test_start = train_end + 1
        test_end = test_start + test_years - 1

        train_mask = (df['date'].dt.year >= train_start) & (df['date'].dt.year <= train_end)
        test_mask = (df['date'].dt.year >= test_start) & (df['date'].dt.year <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            windows.append((train_start, test_start, train_df, test_df))
    return windows


def make_sliding_sequences(df, feature_cols, time_steps, horizon, target_col="close"):
    """
    Create sliding window sequences for time-series forecasting.
    """
    X, y = [], []
    for i in range(len(df) - time_steps - horizon + 1):
        X_seq = df[feature_cols].iloc[i : i + time_steps].values
        y_val = df[target_col].iloc[i + time_steps + horizon - 1]
        X.append(X_seq)
        y.append(y_val)
    return np.array(X), np.array(y)


def normalize_and_sequence(train_df, test_df, feature_cols, time_steps, horizon):
    """
    Normalize features and targets using MinMaxScaler, 
    then create sliding sequences for train/test sets.
    """
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_df_original = train_df.copy()
    test_df_original = test_df.copy()

    # --- Target scaling (y) ---
    scaler_y.fit(train_df_original[['close']])
    y_train_scaled = scaler_y.transform(train_df_original[['close']])
    y_test_scaled = scaler_y.transform(test_df_original[['close']])

    # --- Feature scaling (X) ---
    feature_cols_x = feature_cols.copy()
    scaler_x.fit(train_df_original[feature_cols_x])
    X_train_scaled_features = scaler_x.transform(train_df_original[feature_cols_x])
    X_test_scaled_features = scaler_x.transform(test_df_original[feature_cols_x])

    # --- Rebuild scaled DataFrames ---
    train_df_scaled = pd.DataFrame(X_train_scaled_features, columns=feature_cols_x, index=train_df_original.index)
    test_df_scaled = pd.DataFrame(X_test_scaled_features, columns=feature_cols_x, index=test_df_original.index)

    train_df_scaled['date'] = train_df_original['date']
    test_df_scaled['date'] = test_df_original['date']
    train_df_scaled['target_close'] = y_train_scaled
    test_df_scaled['target_close'] = y_test_scaled

    # --- Create time-series sequences ---
    X_train, y_train = make_sliding_sequences(train_df_scaled, feature_cols_x, time_steps, horizon, target_col="target_close")
    X_test, y_test = make_sliding_sequences(test_df_scaled, feature_cols_x, time_steps, horizon, target_col="target_close")

    return X_train, y_train, X_test, y_test, scaler_x, scaler_y
