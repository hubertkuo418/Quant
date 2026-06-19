from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class PredictionSignalConfig:
    input_path: Path
    output_path: Path
    horizon: int
    model: str | None = None
    score_column: str = "model_score"
    metadata_path: Path | None = None


def load_prediction_signal_config(path: str | Path) -> PredictionSignalConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    horizon = int(payload["horizon"])
    if horizon <= 0:
        raise ValueError("Prediction signal horizon must be positive.")
    score_column = str(payload.get("score_column", "model_score")).strip()
    if not score_column:
        raise ValueError("Prediction signal score_column cannot be empty.")
    if score_column in {"date", "ticker"}:
        raise ValueError("Prediction signal score_column is a reserved key column.")
    return PredictionSignalConfig(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        horizon=horizon,
        model=str(payload["model"]) if payload.get("model") else None,
        score_column=score_column,
        metadata_path=(
            Path(payload["metadata_path"]) if payload.get("metadata_path") else None
        ),
    )


def predictions_to_signal_panel(
    predictions: pd.DataFrame,
    horizon: int,
    model: str | None = None,
    score_column: str = "model_score",
) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("Prediction signal horizon must be positive.")
    if not score_column or score_column in {"date", "ticker"}:
        raise ValueError("Prediction score column must be a non-key column.")
    required = {"date", "ticker", "horizon", "prediction"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype("string").str.strip().str.upper()
    frame = frame[frame["horizon"] == horizon]
    if model is not None:
        if "model" not in frame.columns:
            raise ValueError("Model filter requires a 'model' column.")
        frame = frame[frame["model"] == model]

    signal = frame[["date", "ticker", "prediction"]].rename(
        columns={"prediction": score_column}
    )
    if signal.empty:
        raise ValueError(
            f"No predictions matched horizon={horizon} and model={model!r}."
        )
    if signal[["date", "ticker"]].isna().any().any() or signal["ticker"].eq("").any():
        raise ValueError("Prediction date and ticker keys must be valid and non-empty.")
    if signal[["date", "ticker"]].duplicated().any():
        raise ValueError(
            "Prediction filter produced duplicate (date, ticker) rows; "
            "select one model explicitly."
        )
    scores = pd.to_numeric(signal[score_column], errors="coerce")
    if not np.isfinite(scores.to_numpy(dtype="float64")).all():
        raise ValueError("Prediction scores must be finite numeric values.")
    signal[score_column] = scores
    return signal.sort_values(["date", "ticker"]).reset_index(drop=True)


def convert_prediction_file(
    input_path: str | Path,
    output_path: str | Path,
    horizon: int,
    model: str | None = None,
    score_column: str = "model_score",
    metadata_path: str | Path | None = None,
) -> pd.DataFrame:
    source = Path(input_path)
    signal = predictions_to_signal_panel(
        pd.read_parquet(source),
        horizon=horizon,
        model=model,
        score_column=score_column,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signal.to_parquet(path, index=False)
    if metadata_path is not None:
        metadata = Path(metadata_path)
        metadata.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "run_utc": datetime.now(UTC).isoformat(),
            "config": {
                "input_path": str(source),
                "output_path": str(path),
                "horizon": horizon,
                "model": model,
                "score_column": score_column,
            },
            "input_sha256": _sha256(source),
            "rows": len(signal),
            "dates": signal["date"].nunique(),
            "tickers": signal["ticker"].nunique(),
            "min_date": signal["date"].min().isoformat(),
            "max_date": signal["date"].max().isoformat(),
        }
        metadata.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return signal


def convert_prediction_config(config: PredictionSignalConfig) -> pd.DataFrame:
    return convert_prediction_file(**asdict(config))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
