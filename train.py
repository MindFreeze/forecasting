import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple
from model import TransformerModel
from data_preprocessing import load_and_preprocess_data

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int, learning_rate: float, device: torch.device) -> Tuple[nn.Module, list]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            # Check for nan loss
            if torch.isnan(loss):
                print(f"NaN loss encountered. Skipping batch.")
                continue

            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, (train_losses, val_losses)

def prepare_data_loaders(X: np.ndarray, y: np.ndarray, batch_size: int, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    # Split data into train and validation sets
    train_size = int(len(X) * train_ratio)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def run_training(data_path: str, model_save_path: str):
    # Hyperparameters
    sequence_length = 60
    hidden_dim = 64
    num_layers = 2
    num_heads = 4
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    X, y, scaler = load_and_preprocess_data(data_path, sequence_length)

    # Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(X, y, batch_size)

    # Initialize model
    input_dim = X.shape[2]
    output_dim = 1
    model = TransformerModel(input_dim, hidden_dim, num_layers, num_heads, output_dim).to(device)

    # Train model
    trained_model, _ = train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Save the trained model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'scaler': scaler
    }, model_save_path)

    print(f"Model saved to {model_save_path}")
