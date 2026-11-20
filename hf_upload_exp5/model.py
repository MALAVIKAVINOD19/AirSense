import torch
import torch.nn as nn

class AirSenseTransformerExp5(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_projection = nn.Linear(9, 64)
        self.position_embedding = nn.Parameter(torch.zeros(1, 12, 64))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.output_layer = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_layer(x)