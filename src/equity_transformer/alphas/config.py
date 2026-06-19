from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AlphaConfig:
    market_path: Path
    output_path: Path
    metadata_path: Path
    alphas: tuple[str, ...]


def load_alpha_config(path: str | Path) -> AlphaConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return AlphaConfig(
        market_path=Path(payload["market_path"]),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
        alphas=tuple(str(alpha) for alpha in payload["alphas"]),
    )
