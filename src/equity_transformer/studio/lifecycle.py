from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from equity_transformer.studio.specs import (
    StrategySpec,
    load_strategy_spec,
    save_strategy_spec,
)


@dataclass(frozen=True)
class StrategyVersionRecord:
    path: Path
    status: str
    name: str
    version: str
    spec_hash: str

    @property
    def label(self) -> str:
        return f"{self.status} | {self.version} | {self.spec_hash[:8]}"


class StrategyLifecycleManager:
    def __init__(self, root: str | Path = "strategies") -> None:
        self.root = Path(root)
        self.history_root = self.root / "history"
        self.archive_root = self.root / "archive"
        self.trash_root = self.root / "trash"

    def save(
        self,
        spec: StrategySpec,
        path: str | Path | None = None,
    ) -> Path:
        spec.validate()
        target = self._active_path(path or self.root / f"{spec.slug}.yaml")
        if target.exists():
            existing = load_strategy_spec(target)
            if existing.spec_hash != spec.spec_hash:
                self._snapshot(existing)
        save_strategy_spec(spec, target)
        return target

    def duplicate(
        self,
        source: str | Path,
        name: str,
        version: str = "1.0.0",
    ) -> Path:
        original = load_strategy_spec(self._active_path(source))
        duplicate = replace(
            original,
            name=name.strip(),
            version=version.strip(),
            parent_run_id=None,
        )
        target = self.root / f"{duplicate.slug}.yaml"
        if target.exists():
            raise FileExistsError(f"Strategy already exists: {target}")
        return self.save(duplicate, target)

    def archive(self, source: str | Path) -> Path:
        return self._move_out(source, self.archive_root)

    def soft_delete(self, source: str | Path, confirmation_name: str) -> Path:
        path = self._active_path(source)
        spec = load_strategy_spec(path)
        if confirmation_name.strip() != spec.name:
            raise ValueError("Confirmation name does not match the strategy name.")
        return self._move_out(path, self.trash_root)

    def restore(self, source: str | Path) -> Path:
        source_path = self._managed_path(source)
        if source_path.parent not in {self.archive_root, self.trash_root}:
            raise ValueError("Only archived or deleted strategies can be restored.")
        spec = load_strategy_spec(source_path)
        target = self.root / f"{spec.slug}.yaml"
        if target.exists():
            raise FileExistsError(f"Active strategy already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target)
        return target

    def versions(self, source: str | Path) -> list[StrategyVersionRecord]:
        current_path = self._active_path(source)
        current = load_strategy_spec(current_path)
        records = [self._record(current_path, "目前版本")]
        history_dir = self.history_root / current.slug
        if history_dir.exists():
            records.extend(
                self._record(path, "歷史版本")
                for path in sorted(history_dir.glob("*.yaml"), reverse=True)
            )
        return records

    def archived(self) -> list[StrategyVersionRecord]:
        return self._records_in(self.archive_root, "已封存")

    def deleted(self) -> list[StrategyVersionRecord]:
        return self._records_in(self.trash_root, "已刪除")

    def _snapshot(self, spec: StrategySpec) -> Path:
        output = (
            self.history_root
            / spec.slug
            / f"{spec.version}__{spec.spec_hash[:12]}.yaml"
        )
        if not output.exists():
            save_strategy_spec(spec, output)
        return output

    def _move_out(self, source: str | Path, destination: Path) -> Path:
        path = self._active_path(source)
        spec = load_strategy_spec(path)
        self._snapshot(spec)
        destination.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = destination / f"{spec.slug}__{timestamp}.yaml"
        path.replace(target)
        return target

    def _records_in(self, directory: Path, status: str) -> list[StrategyVersionRecord]:
        if not directory.exists():
            return []
        return [
            self._record(path, status)
            for path in sorted(directory.glob("*.yaml"), reverse=True)
        ]

    @staticmethod
    def _record(path: Path, status: str) -> StrategyVersionRecord:
        spec = load_strategy_spec(path)
        return StrategyVersionRecord(
            path=path,
            status=status,
            name=spec.name,
            version=spec.version,
            spec_hash=spec.spec_hash,
        )

    def _active_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = candidate
        resolved = candidate.resolve()
        if resolved.parent != self.root.resolve():
            raise ValueError("Strategy path must be directly under the strategy root.")
        return candidate

    def _managed_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        resolved = candidate.resolve()
        allowed = {self.archive_root.resolve(), self.trash_root.resolve()}
        if resolved.parent not in allowed:
            raise ValueError("Strategy path is outside managed lifecycle directories.")
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate


def compare_strategy_versions(
    left: StrategySpec,
    right: StrategySpec,
    include_unchanged: bool = False,
) -> pd.DataFrame:
    left_values = _flatten(left.to_dict())
    right_values = _flatten(right.to_dict())
    rows = []
    for field in sorted(set(left_values) | set(right_values)):
        left_value = left_values.get(field)
        right_value = right_values.get(field)
        changed = left_value != right_value
        if changed or include_unchanged:
            rows.append(
                {
                    "field": field,
                    "left": left_value,
                    "right": right_value,
                    "changed": changed,
                }
            )
    return pd.DataFrame(rows, columns=["field", "left", "right", "changed"])


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(item, field))
        return flattened
    if isinstance(value, list):
        return {prefix: json.dumps(value, ensure_ascii=False, sort_keys=True)}
    return {prefix: value}
