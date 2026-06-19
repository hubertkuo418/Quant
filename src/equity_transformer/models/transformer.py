from __future__ import annotations

import torch
from torch import nn


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sequence_length: int,
        output_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )
        nn.init.normal_(self.position_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence_length = features.shape[1]
        encoded = self.input_projection(features)
        encoded = encoded + self.position_embedding[:, :sequence_length]
        encoded = self.encoder(encoded)
        return self.output_head(encoded[:, -1])
