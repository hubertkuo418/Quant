from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from equity_transformer.studio.lifecycle import (
    StrategyLifecycleManager,
    compare_strategy_versions,
)
from equity_transformer.studio.specs import (
    PortfolioSpec,
    SignalSpec,
    StrategySpec,
    load_strategy_spec,
    save_strategy_spec,
)


def make_spec(root: Path, name: str = "factor-top10") -> StrategySpec:
    return StrategySpec(
        name=name,
        version="1.0.0",
        description="base",
        market_path=root / "market.parquet",
        signal=SignalSpec(root / "signals.parquet", "factor_score"),
        portfolio=PortfolioSpec(top_k=10),
    )


def test_save_snapshots_changed_active_strategy(tmp_path: Path) -> None:
    root = tmp_path / "strategies"
    root.mkdir()
    path = root / "factor.yaml"
    original = make_spec(tmp_path)
    save_strategy_spec(original, path)
    changed = replace(
        original,
        version="1.1.0",
        portfolio=replace(original.portfolio, top_k=5),
    )
    manager = StrategyLifecycleManager(root)

    manager.save(changed, path)
    versions = manager.versions(path)

    assert load_strategy_spec(path) == changed
    assert [record.status for record in versions] == ["目前版本", "歷史版本"]
    assert versions[1].spec_hash == original.spec_hash


def test_duplicate_requires_a_unique_strategy_name(tmp_path: Path) -> None:
    root = tmp_path / "strategies"
    root.mkdir()
    source = root / "factor.yaml"
    save_strategy_spec(make_spec(tmp_path), source)
    manager = StrategyLifecycleManager(root)

    duplicate = manager.duplicate(source, "Balanced Strategy", "2.0.0")

    assert duplicate == root / "balanced-strategy.yaml"
    assert load_strategy_spec(duplicate).version == "2.0.0"
    with pytest.raises(FileExistsError):
        manager.duplicate(source, "Balanced Strategy")


def test_archive_delete_and_restore_are_reversible(tmp_path: Path) -> None:
    root = tmp_path / "strategies"
    root.mkdir()
    source = root / "factor.yaml"
    spec = make_spec(tmp_path)
    save_strategy_spec(spec, source)
    manager = StrategyLifecycleManager(root)

    archived = manager.archive(source)
    restored = manager.restore(archived)
    with pytest.raises(ValueError, match="Confirmation"):
        manager.soft_delete(restored, "wrong")
    deleted = manager.soft_delete(restored, spec.name)

    assert not restored.exists()
    assert deleted.exists()
    assert manager.deleted()[0].name == spec.name
    assert manager.restore(deleted).exists()


def test_compare_strategy_versions_reports_nested_changes(tmp_path: Path) -> None:
    left = make_spec(tmp_path)
    right = replace(
        left,
        version="1.1.0",
        portfolio=replace(left.portfolio, top_k=7),
    )

    comparison = compare_strategy_versions(left, right)

    assert set(comparison["field"]) == {"portfolio.top_k", "version"}
    assert comparison["changed"].all()


def test_lifecycle_rejects_paths_outside_strategy_root(tmp_path: Path) -> None:
    root = tmp_path / "strategies"
    root.mkdir()
    outside = tmp_path / "outside.yaml"
    save_strategy_spec(make_spec(tmp_path), outside)

    with pytest.raises(ValueError, match="strategy root"):
        StrategyLifecycleManager(root).archive(outside)
