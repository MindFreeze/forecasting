import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, output_dim: int):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (batch, seq_len, features)
        x = self.fc(x[:, -1, :])  # Only use the last output for prediction
        return x
