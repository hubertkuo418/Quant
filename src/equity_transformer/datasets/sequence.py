from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SampleMetadata:
    ticker: str
    date: pd.Timestamp
    split: str


class EquitySequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        target_columns: tuple[str, ...],
        sequence_length: int,
        split: str,
    ) -> None:
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.sequence_length = sequence_length
        self._sequences: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._metadata: list[SampleMetadata] = []
        self._build(frame, split)

    def _build(self, frame: pd.DataFrame, split: str) -> None:
        for ticker, ticker_frame in frame.groupby("ticker", sort=False):
            ordered = ticker_frame.sort_values("date").reset_index(drop=True)
            features = ordered.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
            targets = ordered.loc[:, self.target_columns].to_numpy(dtype=np.float32)
            for endpoint in range(self.sequence_length - 1, len(ordered)):
                row = ordered.iloc[endpoint]
                if row["split"] != split or np.isnan(targets[endpoint]).any():
                    continue
                sequence = features[endpoint - self.sequence_length + 1 : endpoint + 1]
                if not np.isfinite(sequence).all():
                    continue
                self._sequences.append(sequence)
                self._targets.append(targets[endpoint])
                self._metadata.append(
                    SampleMetadata(ticker=ticker, date=row["date"], split=split)
                )

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._sequences[index]),
            torch.from_numpy(self._targets[index]),
        )

    def metadata(self, index: int) -> SampleMetadata:
        return self._metadata[index]

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._targets:
            return (
                np.empty(
                    (0, self.sequence_length, len(self.feature_columns)),
                    dtype=np.float32,
                ),
                np.empty((0, len(self.target_columns)), dtype=np.float32),
            )
        return np.stack(self._sequences), np.stack(self._targets)

    @property
    def sample_metadata(self) -> tuple[SampleMetadata, ...]:
        return tuple(self._metadata)


def make_dataloader(
    dataset: EquitySequenceDataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
