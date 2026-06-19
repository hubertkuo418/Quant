from __future__ import annotations

import torch
from torch import nn


class RecurrentSequenceModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        model_type: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        recurrent_cls = {
            "rnn": nn.RNN,
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }.get(model_type.lower())
        if recurrent_cls is None:
            raise ValueError(f"Unsupported recurrent model_type: {model_type}")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = recurrent_cls(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        sequence_output, _ = self.encoder(features)
        return self.output_head(sequence_output[:, -1])
