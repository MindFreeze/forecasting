import pandas as pd
import numpy as np

def get_latest_data(file_path: str, sequence_length: int = 60) -> np.ndarray:
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.set_index('time', inplace=True)

    features = ['A', 'V', 'consumption', 'solar', 'outside_temp']
    latest_data = df[features].iloc[-sequence_length:].values

    return latest_data
