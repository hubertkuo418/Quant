from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

from equity_transformer.gui.artifacts import load_dashboard_artifacts
from equity_transformer.gui.configs import (
    list_config_entries,
    read_config_text,
    write_config_text,
)
from equity_transformer.gui.metrics import (
    FINANCING_METRICS,
    OVERVIEW_METRICS,
    RELATIVE_METRICS,
    TAIL_RISK_METRICS,
    available_metrics,
    metric_table,
)
from equity_transformer.gui.strategy_builder import apply_selected_factor_to_strategy
from equity_transformer.gui.workflow import (
    latest_workflow_runs,
    list_workflow_steps,
    run_workflow_step,
)
from equity_transformer.studio.comparison import compare_strategy_runs
from equity_transformer.studio.optimizer import (
    StrategyOptimizer,
    load_optimization_config,
)
from equity_transformer.studio.recommendation import (
    load_recommendation_profile,
    recommend_strategies,
    save_recommendations,
)
from equity_transformer.studio.registry import StrategyRunRegistry
from equity_transformer.studio.runner import StrategyStudioRunner
from equity_transformer.studio.specs import (
    save_strategy_spec,
    strategy_spec_from_dict,
)

st.set_page_config(page_title="QuantLab Strategy Studio", layout="wide")
st.title("QuantLab Strategy Studio")
st.caption("Design, backtest, compare, optimize, and version quant strategies.")

artifacts = load_dashboard_artifacts()

tabs = st.tabs(
    [
        "Dashboard",
        "Factor Explorer",
        "Strategy",
        "Model Lab",
        "Backtest",
        "Workflow",
        "Configs",
        "Strategy Studio",
    ]
)

with tabs[0]:
    quality_summary = artifacts["data_quality_summary"]
    if quality_summary:
        st.subheader("Market Data Quality")
        quality_columns = st.columns(4)
        quality_columns[0].metric("Rows", f"{quality_summary.get('rows', 0):,}")
        quality_columns[1].metric("Tickers", quality_summary.get("tickers", 0))
        quality_columns[2].metric(
            "Calendar Dates", quality_summary.get("calendar_dates", 0)
        )
        quality_columns[3].metric(
            "Issues", quality_summary.get("issue_count", 0)
        )
        quality_issues = artifacts["data_quality_issues"]
        if isinstance(quality_issues, pd.DataFrame) and not quality_issues.empty:
            st.dataframe(quality_issues, hide_index=True)

    st.subheader("Performance Overview")
    metrics = artifacts["backtest_metrics"]
    if metrics:
        overview = available_metrics(metrics, OVERVIEW_METRICS)
        if overview:
            columns = st.columns(len(overview))
            for column, (key, value) in zip(columns, overview, strict=True):
                column.metric(key.replace("_", " ").title(), f"{value:.4f}")

        risk_table = metric_table(metrics, TAIL_RISK_METRICS)
        if not risk_table.empty:
            st.subheader("Tail Risk")
            st.dataframe(risk_table, hide_index=True)

        relative_table = metric_table(metrics, RELATIVE_METRICS)
        if not relative_table.empty:
            st.subheader("Benchmark Relative")
            st.dataframe(relative_table, hide_index=True)

        financing_table = metric_table(metrics, FINANCING_METRICS)
        if not financing_table.empty:
            st.subheader("Costs And Financing")
            st.dataframe(financing_table, hide_index=True)
    else:
        st.info("Run a backtest to populate performance metrics.")

    equity = artifacts["backtest_equity"]
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.plotly_chart(px.line(equity, x="date", y="nav", title="Equity Curve"))

    benchmarks = artifacts["benchmark_comparison"]
    if isinstance(benchmarks, pd.DataFrame) and not benchmarks.empty:
        st.subheader("Benchmarks")
        st.dataframe(benchmarks)

    portfolio_comparison = artifacts["portfolio_comparison"]
    if (
        isinstance(portfolio_comparison, pd.DataFrame)
        and not portfolio_comparison.empty
    ):
        st.subheader("Portfolio Comparison")
        st.dataframe(portfolio_comparison)

    common_period = artifacts["common_period_comparison"]
    if isinstance(common_period, pd.DataFrame) and not common_period.empty:
        st.subheader("Common-Period Portfolio Comparison")
        st.dataframe(common_period)
        if {"portfolio", "sharpe_ratio"}.issubset(common_period.columns):
            st.plotly_chart(
                px.bar(
                    common_period,
                    x="portfolio",
                    y="sharpe_ratio",
                    title="Common-Period Sharpe Ratio",
                )
            )

    regime_performance = artifacts["regime_performance"]
    if isinstance(regime_performance, pd.DataFrame) and not regime_performance.empty:
        st.subheader("Market Regimes")
        st.dataframe(regime_performance)
        if {"regime", "sharpe_ratio"}.issubset(regime_performance.columns):
            st.plotly_chart(
                px.bar(
                    regime_performance,
                    x="regime",
                    y="sharpe_ratio",
                    title="Sharpe Ratio By Market Regime",
                )
            )

    sensitivity = artifacts["sensitivity_comparison"]
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        st.subheader("Sensitivity Analysis")
        st.dataframe(sensitivity)
        if {"scenario", "sharpe_ratio"}.issubset(sensitivity.columns):
            st.plotly_chart(
                px.bar(
                    sensitivity,
                    x="scenario",
                    y="sharpe_ratio",
                    title="Sharpe Ratio Under Stress Scenarios",
                )
            )

    trading_diagnostics = artifacts["trading_diagnostics"]
    if isinstance(trading_diagnostics, pd.DataFrame) and not trading_diagnostics.empty:
        st.subheader("Trading Diagnostics")
        st.dataframe(trading_diagnostics)

    model_comparison = artifacts["model_comparison"]
    if isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(model_comparison)

with tabs[1]:
    selected_factors = artifacts["selected_factors"]
    if isinstance(selected_factors, pd.DataFrame) and not selected_factors.empty:
        st.subheader("Selected Factors")
        st.dataframe(selected_factors)
        if "selection_score" in selected_factors.columns:
            st.plotly_chart(
                px.bar(
                    selected_factors.head(20),
                    x="factor",
                    y="selection_score",
                    color="family" if "family" in selected_factors.columns else None,
                    title="Selected Factor Scores",
                )
            )

    st.subheader("Factor IC Summary")
    factor_ic = artifacts["factor_ic"]
    if isinstance(factor_ic, pd.DataFrame) and not factor_ic.empty:
        st.dataframe(factor_ic)
        st.plotly_chart(
            px.bar(
                factor_ic.head(20),
                x="factor",
                y="mean_rank_ic",
                color="family",
                title="Top Factor Rank IC",
            )
        )
    else:
        st.info("Run factor validation to populate the Factor Explorer.")

    quantiles = artifacts["factor_quantiles"]
    if isinstance(quantiles, pd.DataFrame) and not quantiles.empty:
        st.subheader("Quantile Returns")
        st.dataframe(quantiles)

    factor_signals = artifacts["factor_signals"]
    if isinstance(factor_signals, pd.DataFrame) and not factor_signals.empty:
        st.subheader("IC-Weighted Factor Signals")
        st.dataframe(factor_signals.head(200))
        if "factor_score" in factor_signals.columns:
            st.plotly_chart(
                px.histogram(
                    factor_signals,
                    x="factor_score",
                    title="Factor Score Distribution",
                )
            )
    factor_signals_manifest = artifacts["factor_signals_manifest"]
    if factor_signals_manifest:
        st.subheader("Factor Signal Manifest")
        st.json(factor_signals_manifest)

with tabs[2]:
    st.subheader("Strategy Target Weights")
    strategy_manifest = artifacts["strategy_manifest"]
    if strategy_manifest:
        st.subheader("Strategy Settings")
        st.json(strategy_manifest)
    selected_manifest = artifacts["selected_factors_manifest"]
    selected_names = selected_manifest.get("selected_factors", [])
    if selected_names:
        chosen_factor = st.selectbox("Use Selected Factor", options=selected_names)
        if st.button("Apply Factor To Strategy Config"):
            try:
                apply_selected_factor_to_strategy(
                    factor_index=selected_names.index(chosen_factor)
                )
            except Exception as exc:
                st.error(f"Could not update strategy config: {exc}")
            else:
                st.success(f"Strategy score_column set to {chosen_factor}.")
    weights = artifacts["strategy_weights"]
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        st.dataframe(weights)
    else:
        st.info("Run strategy construction to populate target weights.")

with tabs[3]:
    st.subheader("Model Lab")
    model_comparison = artifacts["model_comparison"]
    if isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(model_comparison)
        if {"model", "horizon", "rmse"}.issubset(model_comparison.columns):
            st.plotly_chart(
                px.bar(
                    model_comparison,
                    x="model",
                    y="rmse",
                    color="source" if "source" in model_comparison.columns else None,
                    facet_col="horizon",
                    title="RMSE by Model and Horizon",
                )
            )
    else:
        st.info("Run model training and report aggregation to populate Model Lab.")

    transformer_metrics = artifacts["transformer_metrics"]
    if transformer_metrics:
        st.subheader("Transformer Summary")
        st.json(transformer_metrics)

    transformer_predictions = artifacts["transformer_predictions"]
    if (
        isinstance(transformer_predictions, pd.DataFrame)
        and not transformer_predictions.empty
    ):
        st.subheader("Transformer Test Predictions")
        st.dataframe(transformer_predictions.head(200))
        if {"horizon", "prediction"}.issubset(transformer_predictions.columns):
            st.plotly_chart(
                px.histogram(
                    transformer_predictions,
                    x="prediction",
                    color="horizon",
                    title="Prediction Distribution by Horizon",
                )
            )

    model_signals = artifacts["model_signals"]
    if isinstance(model_signals, pd.DataFrame) and not model_signals.empty:
        st.subheader("Strategy-Ready Model Signals")
        st.dataframe(model_signals.head(200))

with tabs[4]:
    st.subheader("Backtest Details")
    equity = artifacts["backtest_equity"]
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.dataframe(equity)
        exposure = artifacts["backtest_exposure"]
        if isinstance(exposure, pd.DataFrame) and not exposure.empty:
            st.subheader("Exposure")
            st.dataframe(exposure)
            st.plotly_chart(
                px.line(
                    exposure,
                    x="date",
                    y=["gross_exposure", "net_exposure"],
                    title="Portfolio Exposure",
                )
            )
        sector_exposure = artifacts["backtest_sector_exposure"]
        if isinstance(sector_exposure, pd.DataFrame) and not sector_exposure.empty:
            st.subheader("Sector Exposure")
            st.dataframe(sector_exposure)
            st.plotly_chart(
                px.line(
                    sector_exposure,
                    x="date",
                    y="gross_exposure",
                    color="sector",
                    title="Sector Gross Exposure",
                )
            )
        trades = artifacts["backtest_trades"]
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            st.subheader("Trade Log")
            st.dataframe(trades)
        attribution_metrics = artifacts["attribution_metrics"]
        if attribution_metrics:
            st.subheader("Contribution Concentration")
            st.json(attribution_metrics)
        attribution_summary = artifacts["attribution_summary"]
        if (
            isinstance(attribution_summary, pd.DataFrame)
            and not attribution_summary.empty
        ):
            st.subheader("Ticker Return Attribution")
            st.dataframe(attribution_summary)
            if {"ticker", "total_contribution"}.issubset(
                attribution_summary.columns
            ):
                st.plotly_chart(
                    px.bar(
                        attribution_summary.head(20),
                        x="ticker",
                        y="total_contribution",
                        title="Top Gross Return Contributors",
                    )
                )
    else:
        st.info("Run a backtest to populate this page.")

with tabs[5]:
    st.subheader("Workflow Runner")
    st.caption("Runs only known QuantLab scripts through the tested backend modules.")
    for step in list_workflow_steps():
        with st.expander(step.label):
            st.write(step.description)
            st.code("python " + " ".join(step.command), language="powershell")
            if st.button(f"Run {step.label}", key=f"run_{step.name}"):
                with st.spinner(f"Running {step.label}..."):
                    result = run_workflow_step(step.name)
                if result["success"]:
                    st.success(f"{step.label} completed.")
                else:
                    st.error(f"{step.label} failed with code {result['returncode']}.")
                if result["stdout"]:
                    st.text_area("stdout", result["stdout"], height=160)
                if result["stderr"]:
                    st.text_area("stderr", result["stderr"], height=160)

    st.subheader("Recent Runs")
    runs = latest_workflow_runs()
    if runs:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "step": run["step"],
                        "success": run["success"],
                        "returncode": run["returncode"],
                        "started_utc": run["started_utc"],
                        "duration_seconds": run["duration_seconds"],
                    }
                    for run in runs
                ]
            )
        )
    else:
        st.info("No workflow runs recorded yet.")

with tabs[6]:
    st.subheader("Configuration Editor")
    st.caption("Only known QuantLab YAML configs can be edited here.")
    entries = {entry.name: entry for entry in list_config_entries()}
    selected = st.selectbox("Config", options=list(entries))
    st.write(entries[selected].description)
    try:
        current_text = read_config_text(selected)
    except FileNotFoundError as exc:
        st.error(str(exc))
        current_text = ""
    edited = st.text_area("YAML", current_text, height=360)
    if st.button("Validate and Save Config"):
        try:
            parsed = write_config_text(selected, edited)
        except Exception as exc:
            st.error(f"Config was not saved: {exc}")
        else:
            st.success(f"Saved {selected} config with {len(parsed)} top-level keys.")

with tabs[7]:
    st.subheader("Strategy Studio")
    st.caption(
        "Design versioned strategies, run backtests, compare runs, and search "
        "for constraint-aware candidates."
    )
    strategy_paths = sorted(Path("strategies").glob("*.yaml"))
    if strategy_paths:
        selected_strategy = st.selectbox(
            "Strategy specification",
            options=strategy_paths,
            format_func=lambda path: path.name,
        )
        strategy_text = st.text_area(
            "StrategySpec YAML",
            selected_strategy.read_text(encoding="utf-8"),
            height=500,
        )
        save_column, run_column = st.columns(2)
        if save_column.button("Validate and Save Strategy"):
            try:
                payload = yaml.safe_load(strategy_text) or {}
                spec = strategy_spec_from_dict(payload)
                save_strategy_spec(spec, selected_strategy)
            except Exception as exc:
                st.error(f"Strategy was not saved: {exc}")
            else:
                st.success(f"Saved {spec.name} {spec.version} ({spec.spec_hash[:8]}).")
        if run_column.button("Run Strategy"):
            try:
                payload = yaml.safe_load(strategy_text) or {}
                spec = strategy_spec_from_dict(payload)
                result = StrategyStudioRunner().run(spec)
            except Exception as exc:
                st.error(f"Strategy run failed: {exc}")
            else:
                st.success(f"Completed {result.run_id}")
                st.json(result.metrics)
    else:
        st.info("Add a StrategySpec YAML file under strategies/.")

    registry = StrategyRunRegistry()
    run_summary = registry.summary()
    st.subheader("Saved Runs")
    if run_summary.empty:
        st.info("No Strategy Studio runs have been saved yet.")
    else:
        st.dataframe(run_summary, hide_index=True)
        selected_runs = st.multiselect(
            "Runs to compare",
            options=run_summary["run_id"].tolist(),
        )
        if st.button("Compare Selected Runs"):
            try:
                comparison = compare_strategy_runs(
                    selected_runs,
                    registry,
                    "artifacts/studio/comparisons/gui_latest",
                )
            except Exception as exc:
                st.error(f"Run comparison failed: {exc}")
            else:
                st.dataframe(comparison, hide_index=True)
                st.plotly_chart(
                    px.scatter(
                        comparison,
                        x="max_drawdown",
                        y="annual_return",
                        color="sharpe_ratio",
                        hover_name="portfolio",
                        title="Common-Period Risk and Return",
                    )
                )

    st.subheader("Strategy Optimizer")
    st.caption(
        "Searches parameter combinations, applies constraints, and marks "
        "Pareto-efficient candidates on a common period."
    )
    if st.button("Run Configured Optimization"):
        try:
            optimization = StrategyOptimizer(
                load_optimization_config("configs/studio_optimizer.yaml")
            ).run()
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
        else:
            st.dataframe(optimization, hide_index=True)
    optimization_path = Path("artifacts/studio/optimizations/factor_search/results.csv")
    if optimization_path.exists():
        latest_optimization = pd.read_csv(optimization_path)
        st.dataframe(latest_optimization, hide_index=True)
        if {"max_drawdown", "annual_return", "pareto_efficient"}.issubset(
            latest_optimization.columns
        ):
            st.plotly_chart(
                px.scatter(
                    latest_optimization,
                    x="max_drawdown",
                    y="annual_return",
                    color="pareto_efficient",
                    hover_name="run_id",
                    title="Strategy Candidate Pareto Frontier",
                )
            )

    st.subheader("Personalized Candidates")
    st.caption(
        "Applies the configured risk tolerance, drawdown, turnover, and return "
        "constraints. Rankings are historical candidates, not return guarantees."
    )
    if st.button("Build Profile Recommendations"):
        try:
            profile = load_recommendation_profile("configs/studio_profile.yaml")
            candidates = pd.read_csv(optimization_path)
            recommendations = recommend_strategies(candidates, profile)
            save_recommendations(
                recommendations,
                profile,
                "artifacts/studio/recommendations",
            )
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")
        else:
            st.dataframe(recommendations, hide_index=True)
    recommendation_path = Path(
        "artifacts/studio/recommendations/recommendations.csv"
    )
    if recommendation_path.exists():
        st.dataframe(pd.read_csv(recommendation_path), hide_index=True)
