import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from model import TransformerModel
from data_preprocessing import inverse_transform_voltage, preprocess_input_data

def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, MinMaxScaler]:
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = 5  # number of features
    hidden_dim = 64
    num_layers = 2
    num_heads = 4
    output_dim = 1

    model = TransformerModel(input_dim, hidden_dim, num_layers, num_heads, output_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']

    return model, scaler

def predict_voltage(model: nn.Module, scaler: MinMaxScaler, input_sequence: np.ndarray,
                    n_steps: int, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = []

    with torch.no_grad():
        for _ in range(n_steps):
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(device)
            output = model(input_tensor)

            prediction = output.cpu().numpy()[0, 0]
            predictions.append(prediction)

            # Update input sequence for next prediction
            input_sequence = np.roll(input_sequence, -1, axis=0)
            input_sequence[-1] = scaler.transform(np.array([[0, prediction, 0, 0, 0]]))[0]

    return np.array(predictions)

def run_prediction(model_path: str, input_data: np.ndarray, n_steps: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and scaler
    model, scaler = load_model(model_path, device)

    # Preprocess and normalize input data
    normalized_input = preprocess_input_data(input_data, scaler)

    # Make predictions
    normalized_predictions = predict_voltage(model, scaler, normalized_input, n_steps, device)

    # Inverse transform predictions
    voltage_predictions = inverse_transform_voltage(scaler, normalized_predictions)

    return voltage_predictions
