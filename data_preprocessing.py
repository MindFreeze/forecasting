import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def load_and_preprocess_data(file_path: str, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    # Load the CSV file
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.set_index('time', inplace=True)

    # Select features for input
    features = ['A', 'V', 'consumption', 'solar', 'outside_temp']

    # Handle missing values
    df[features] = df[features].ffill().bfill()

    # Drop any remaining rows with NaN values
    df.dropna(subset=features, inplace=True)

    data = df[features].values

    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Create sequences
    X, y = [], []
    for i in range(len(normalized_data) - sequence_length):
        X.append(normalized_data[i:i+sequence_length])
        y.append(normalized_data[i+sequence_length, 1])  # 1 is the index of 'V' in features

    return np.array(X), np.array(y), scaler

def inverse_transform_voltage(scaler: MinMaxScaler, normalized_voltage: np.ndarray) -> np.ndarray:
    # Create a dummy array with the same shape as the original data
    dummy = np.zeros((normalized_voltage.shape[0], 5))
    dummy[:, 1] = normalized_voltage  # 1 is the index of 'V' in features

    # Inverse transform
    return scaler.inverse_transform(dummy)[:, 1]

def preprocess_input_data(input_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    # Fill NaN values with the mean of each column
    column_means = np.nanmean(input_data, axis=0)
    nan_indices = np.where(np.isnan(input_data))
    input_data[nan_indices] = np.take(column_means, nan_indices[1])

    # Normalize the input data
    normalized_input = scaler.transform(input_data)

    return normalized_input
