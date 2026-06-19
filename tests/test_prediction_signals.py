from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.strategies.prediction_signals import (
    PredictionSignalConfig,
    convert_prediction_config,
    convert_prediction_file,
    load_prediction_signal_config,
    predictions_to_signal_panel,
)


def make_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-01",
                "2024-01-01",
                "2024-01-02",
            ],
            "ticker": ["AAA", "BBB", "AAA", "AAA"],
            "model": ["ridge", "ridge", "gru", "ridge"],
            "horizon": [5, 5, 20, 5],
            "prediction": [0.02, -0.01, 0.03, 0.04],
            "target": [0.01, 0.0, 0.02, 0.03],
        }
    )


def test_predictions_to_signal_panel_filters_model_and_horizon() -> None:
    signal = predictions_to_signal_panel(
        make_predictions(),
        horizon=5,
        model="ridge",
        score_column="predicted_return",
    )

    assert signal.columns.tolist() == ["date", "ticker", "predicted_return"]
    assert signal["ticker"].tolist() == ["AAA", "BBB", "AAA"]
    assert signal["predicted_return"].tolist() == [0.02, -0.01, 0.04]


def test_predictions_to_signal_panel_requires_model_column_for_filter() -> None:
    predictions = make_predictions().drop(columns=["model"])

    with pytest.raises(ValueError, match="Model filter"):
        predictions_to_signal_panel(predictions, horizon=5, model="ridge")


def test_convert_prediction_file_writes_signal_parquet(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "signals.parquet"
    make_predictions().to_parquet(input_path, index=False)

    signal = convert_prediction_file(
        input_path,
        output_path,
        horizon=20,
        score_column="model_score",
    )

    reloaded = pd.read_parquet(output_path)
    assert output_path.exists()
    assert len(signal) == 1
    assert reloaded["model_score"].iloc[0] == 0.03


def test_predictions_require_model_when_filter_would_leave_duplicate_keys() -> None:
    predictions = make_predictions()
    duplicate = predictions.iloc[[0]].assign(model="gru")

    with pytest.raises(ValueError, match="select one model"):
        predictions_to_signal_panel(
            pd.concat([predictions, duplicate], ignore_index=True), horizon=5
        )


def test_predictions_reject_empty_horizon() -> None:
    with pytest.raises(ValueError, match="No predictions matched"):
        predictions_to_signal_panel(make_predictions(), horizon=60)


def test_config_conversion_writes_lineage_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "predictions.parquet"
    output_path = tmp_path / "signals.parquet"
    metadata_path = tmp_path / "signals.json"
    make_predictions().to_parquet(input_path, index=False)
    config = PredictionSignalConfig(
        input_path=input_path,
        output_path=output_path,
        metadata_path=metadata_path,
        horizon=5,
        model="ridge",
    )

    signal = convert_prediction_config(config)
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(signal) == 3
    assert manifest["rows"] == 3
    assert len(manifest["input_sha256"]) == 64


def test_load_prediction_signal_config(tmp_path: Path) -> None:
    path = tmp_path / "prediction.yaml"
    path.write_text(
        "input_path: in.parquet\n"
        "output_path: out.parquet\n"
        "metadata_path: manifest.json\n"
        "horizon: 20\n"
        "model: gru\n",
        encoding="utf-8",
    )

    config = load_prediction_signal_config(path)

    assert config.horizon == 20
    assert config.model == "gru"
    assert config.score_column == "model_score"
