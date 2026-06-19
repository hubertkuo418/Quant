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
from equity_transformer.gui.studio_wizard import (
    StrategyWizardValues,
    build_strategy_from_wizard,
    components_from_frame,
    components_to_frame,
    strategy_output_path,
    strategy_to_wizard_values,
)
from equity_transformer.gui.workflow import (
    latest_workflow_runs,
    list_workflow_steps,
    run_workflow_step,
)
from equity_transformer.studio.comparison import compare_strategy_runs
from equity_transformer.studio.lifecycle import (
    StrategyLifecycleManager,
    compare_strategy_versions,
)
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
    load_strategy_spec,
    strategy_spec_from_dict,
)
from equity_transformer.studio.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
)

DISPLAY_LABELS = {
    "absolute_contribution": "絕對貢獻",
    "active_positions": "持倉數量",
    "adj_close": "調整收盤價",
    "annual_alpha": "年化 Alpha",
    "annual_borrow_rate": "年化借券費率",
    "annual_return": "年化報酬率",
    "annual_volatility": "年化波動率",
    "average_abs_trade_value": "平均絕對交易金額",
    "average_active_positions": "平均持倉數量",
    "average_gross_exposure": "平均總曝險",
    "average_net_exposure": "平均淨曝險",
    "average_turnover": "平均換手率",
    "benchmark": "基準",
    "benchmark_return": "基準報酬",
    "beta": "Beta",
    "borrow_cost": "借券成本",
    "cagr": "年複合成長率",
    "calmar_ratio": "Calmar 比率",
    "cash_interest": "現金利息",
    "changed": "是否變更",
    "commission_bps": "手續費（基點）",
    "common_end": "共同結束日",
    "common_start": "共同起始日",
    "conditional_value_at_risk_95": "條件風險值（95%）",
    "contribution_rank": "貢獻排名",
    "cost": "成本",
    "coverage": "覆蓋率",
    "created_utc": "建立時間（UTC）",
    "date": "日期",
    "days_held": "持有天數",
    "delta_weight": "權重變動",
    "duration_seconds": "執行秒數",
    "ending_nav": "期末淨值",
    "end_date": "結束日期",
    "evaluation_basis": "評估基礎",
    "execution_lag_days": "執行延遲天數",
    "execution_robustness": "執行穩健性",
    "field": "設定欄位",
    "fold": "Fold",
    "family": "因子類別",
    "factor": "因子",
    "factor_score": "因子分數",
    "feasible": "符合限制",
    "financing_return": "融資報酬",
    "from_weight": "原權重",
    "gross_exposure": "總曝險",
    "gross_return": "成本前報酬",
    "horizon": "預測週期",
    "ic_std": "IC 標準差",
    "icir": "ICIR",
    "information_ratio": "資訊比率",
    "lag_sharpe_std": "延遲夏普標準差",
    "left": "版本 A",
    "long_exposure": "多頭曝險",
    "max_drawdown": "最大回撤",
    "max_gross_exposure": "最大總曝險",
    "mean_count": "平均樣本數",
    "mean_forward_return": "平均未來報酬",
    "mean_ic": "平均 IC",
    "mean_observations": "平均觀測數",
    "mean_rank_ic": "平均 Rank IC",
    "metric": "指標",
    "model": "模型",
    "nav": "淨值",
    "net_exposure": "淨曝險",
    "net_return": "成本後報酬",
    "observations": "觀測數",
    "parameters": "參數",
    "pareto_efficient": "Pareto 有效率",
    "periods": "期數",
    "portfolio": "投資組合",
    "positive_ic_rate": "正 IC 比率",
    "prediction": "預測值",
    "profit_factor": "獲利因子",
    "profile": "需求設定",
    "path": "檔案路徑",
    "quantile": "分位數",
    "rank_ic_std": "Rank IC 標準差",
    "rank_icir": "Rank ICIR",
    "rationale": "推薦理由",
    "recommendation_rank": "推薦排名",
    "recommendation_score": "推薦分數",
    "returncode": "返回碼",
    "right": "版本 B",
    "run_id": "執行編號",
    "scenario": "情境",
    "score": "分數",
    "selection_score": "選擇分數",
    "share_of_absolute_contribution": "絕對貢獻占比",
    "sharpe_ratio": "夏普比率",
    "short_exposure": "空頭曝險",
    "side": "方向",
    "slippage_bps": "滑價（基點）",
    "sortino_ratio": "Sortino 比率",
    "source": "來源",
    "spec_hash": "規格雜湊",
    "split": "資料切分",
    "start_date": "起始日期",
    "started_utc": "開始時間（UTC）",
    "step": "步驟",
    "strategy": "策略",
    "strategy_family": "策略類別",
    "status": "狀態",
    "success": "成功",
    "target": "實際值",
    "ticker": "股票代號",
    "test_end": "測試結束日",
    "test_start": "測試起始日",
    "to_weight": "目標權重",
    "total_abs_trade_value": "絕對交易總額",
    "total_borrow_cost": "借券總成本",
    "total_cash_interest": "現金利息總額",
    "total_contribution": "總貢獻",
    "total_cost": "總成本",
    "total_return": "總報酬率",
    "total_trade_cost": "交易總成本",
    "tracking_error": "追蹤誤差",
    "train_end": "訓練結束日",
    "train_start": "訓練起始日",
    "trade_count": "交易筆數",
    "trade_value": "交易金額",
    "trial": "試驗編號",
    "turnover": "換手率",
    "value": "數值",
    "value_at_risk_95": "風險值（95%）",
    "version": "版本",
    "volatility_20d": "20 日波動率",
    "weight": "權重",
    "win_rate": "勝率",
    "worst_case_sharpe": "最差情境夏普比率",
}


def display_label(value: str) -> str:
    if value.startswith("raw_"):
        return f"原始{display_label(value.removeprefix('raw_'))}"
    return DISPLAY_LABELS.get(value, value.replace("_", " ").title())


def display_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {column: display_label(str(column)) for column in frame}
    return frame.rename(columns=columns)


st.set_page_config(page_title="QuantLab 量化策略工作室", layout="wide")
st.title("QuantLab 量化策略工作室")
st.caption("自由設計、回測、比較、優化並版本化管理量化策略。")

artifacts = load_dashboard_artifacts()

tabs = st.tabs(
    [
        "總覽儀表板",
        "因子探索",
        "策略組合",
        "模型實驗室",
        "回測分析",
        "工作流程",
        "設定管理",
        "策略工作室",
    ]
)

with tabs[0]:
    quality_summary = artifacts["data_quality_summary"]
    if quality_summary:
        st.subheader("市場資料品質")
        quality_columns = st.columns(4)
        quality_columns[0].metric("資料列數", f"{quality_summary.get('rows', 0):,}")
        quality_columns[1].metric("股票數量", quality_summary.get("tickers", 0))
        quality_columns[2].metric(
            "交易日期數", quality_summary.get("calendar_dates", 0)
        )
        quality_columns[3].metric(
            "問題數量", quality_summary.get("issue_count", 0)
        )
        quality_issues = artifacts["data_quality_issues"]
        if isinstance(quality_issues, pd.DataFrame) and not quality_issues.empty:
            st.dataframe(display_table(quality_issues), hide_index=True)

    st.subheader("績效總覽")
    metrics = artifacts["backtest_metrics"]
    if metrics:
        overview = available_metrics(metrics, OVERVIEW_METRICS)
        if overview:
            columns = st.columns(len(overview))
            for column, (key, value) in zip(columns, overview, strict=True):
                column.metric(display_label(key), f"{value:.4f}")

        risk_table = metric_table(metrics, TAIL_RISK_METRICS)
        if not risk_table.empty:
            st.subheader("尾端風險")
            st.dataframe(display_table(risk_table), hide_index=True)

        relative_table = metric_table(metrics, RELATIVE_METRICS)
        if not relative_table.empty:
            st.subheader("相對基準績效")
            st.dataframe(display_table(relative_table), hide_index=True)

        financing_table = metric_table(metrics, FINANCING_METRICS)
        if not financing_table.empty:
            st.subheader("成本與資金")
            st.dataframe(display_table(financing_table), hide_index=True)
    else:
        st.info("請先執行回測以產生績效指標。")

    equity = artifacts["backtest_equity"]
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.plotly_chart(
            px.line(
                equity,
                x="date",
                y="nav",
                title="資產淨值曲線",
                labels={"date": "日期", "nav": "淨值"},
            )
        )

    benchmarks = artifacts["benchmark_comparison"]
    if isinstance(benchmarks, pd.DataFrame) and not benchmarks.empty:
        st.subheader("基準策略")
        st.dataframe(display_table(benchmarks))

    portfolio_comparison = artifacts["portfolio_comparison"]
    if (
        isinstance(portfolio_comparison, pd.DataFrame)
        and not portfolio_comparison.empty
    ):
        st.subheader("投資組合比較")
        st.dataframe(display_table(portfolio_comparison))

    common_period = artifacts["common_period_comparison"]
    if isinstance(common_period, pd.DataFrame) and not common_period.empty:
        st.subheader("共同期間投資組合比較")
        st.dataframe(display_table(common_period))
        if {"portfolio", "sharpe_ratio"}.issubset(common_period.columns):
            st.plotly_chart(
                px.bar(
                    common_period,
                    x="portfolio",
                    y="sharpe_ratio",
                    title="共同期間夏普比率",
                    labels={"portfolio": "投資組合", "sharpe_ratio": "夏普比率"},
                )
            )

    regime_performance = artifacts["regime_performance"]
    if isinstance(regime_performance, pd.DataFrame) and not regime_performance.empty:
        st.subheader("市場狀態分析")
        st.dataframe(display_table(regime_performance))
        if {"regime", "sharpe_ratio"}.issubset(regime_performance.columns):
            st.plotly_chart(
                px.bar(
                    regime_performance,
                    x="regime",
                    y="sharpe_ratio",
                    title="各市場狀態夏普比率",
                    labels={"regime": "市場狀態", "sharpe_ratio": "夏普比率"},
                )
            )

    sensitivity = artifacts["sensitivity_comparison"]
    if isinstance(sensitivity, pd.DataFrame) and not sensitivity.empty:
        st.subheader("敏感度分析")
        st.dataframe(display_table(sensitivity))
        if {"scenario", "sharpe_ratio"}.issubset(sensitivity.columns):
            st.plotly_chart(
                px.bar(
                    sensitivity,
                    x="scenario",
                    y="sharpe_ratio",
                    title="壓力情境下的夏普比率",
                    labels={"scenario": "情境", "sharpe_ratio": "夏普比率"},
                )
            )

    trading_diagnostics = artifacts["trading_diagnostics"]
    if isinstance(trading_diagnostics, pd.DataFrame) and not trading_diagnostics.empty:
        st.subheader("交易診斷")
        st.dataframe(display_table(trading_diagnostics))

    model_comparison = artifacts["model_comparison"]
    if isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
        st.subheader("模型比較")
        st.dataframe(display_table(model_comparison))

with tabs[1]:
    selected_factors = artifacts["selected_factors"]
    if isinstance(selected_factors, pd.DataFrame) and not selected_factors.empty:
        st.subheader("已選因子")
        st.dataframe(display_table(selected_factors))
        if "selection_score" in selected_factors.columns:
            st.plotly_chart(
                px.bar(
                    selected_factors.head(20),
                    x="factor",
                    y="selection_score",
                    color="family" if "family" in selected_factors.columns else None,
                    title="已選因子分數",
                    labels={
                        "factor": "因子",
                        "selection_score": "選擇分數",
                        "family": "因子類別",
                    },
                )
            )

    st.subheader("因子 IC 摘要")
    factor_ic = artifacts["factor_ic"]
    if isinstance(factor_ic, pd.DataFrame) and not factor_ic.empty:
        st.dataframe(display_table(factor_ic))
        st.plotly_chart(
            px.bar(
                factor_ic.head(20),
                x="factor",
                y="mean_rank_ic",
                color="family",
                title="最佳因子 Rank IC",
                labels={
                    "factor": "因子",
                    "mean_rank_ic": "平均 Rank IC",
                    "family": "因子類別",
                },
            )
        )
    else:
        st.info("請先執行因子驗證以產生因子探索結果。")

    quantiles = artifacts["factor_quantiles"]
    if isinstance(quantiles, pd.DataFrame) and not quantiles.empty:
        st.subheader("分位數報酬")
        st.dataframe(display_table(quantiles))

    factor_signals = artifacts["factor_signals"]
    if isinstance(factor_signals, pd.DataFrame) and not factor_signals.empty:
        st.subheader("IC 加權因子訊號")
        st.dataframe(display_table(factor_signals.head(200)))
        if "factor_score" in factor_signals.columns:
            st.plotly_chart(
                px.histogram(
                    factor_signals,
                    x="factor_score",
                    title="因子分數分布",
                    labels={"factor_score": "因子分數"},
                )
            )
    factor_signals_manifest = artifacts["factor_signals_manifest"]
    if factor_signals_manifest:
        st.subheader("因子訊號執行資訊")
        st.json(factor_signals_manifest)

with tabs[2]:
    st.subheader("策略目標權重")
    strategy_manifest = artifacts["strategy_manifest"]
    if strategy_manifest:
        st.subheader("策略設定")
        st.json(strategy_manifest)
    selected_manifest = artifacts["selected_factors_manifest"]
    selected_names = selected_manifest.get("selected_factors", [])
    if selected_names:
        chosen_factor = st.selectbox("選用因子", options=selected_names)
        if st.button("套用至策略設定"):
            try:
                apply_selected_factor_to_strategy(
                    factor_index=selected_names.index(chosen_factor)
                )
            except Exception as exc:
                st.error(f"無法更新策略設定：{exc}")
            else:
                st.success(f"策略分數欄位已設為 {chosen_factor}。")
    weights = artifacts["strategy_weights"]
    if isinstance(weights, pd.DataFrame) and not weights.empty:
        st.dataframe(display_table(weights))
    else:
        st.info("請先建立策略以產生目標權重。")

with tabs[3]:
    st.subheader("模型實驗室")
    model_comparison = artifacts["model_comparison"]
    if isinstance(model_comparison, pd.DataFrame) and not model_comparison.empty:
        st.subheader("模型比較")
        st.dataframe(display_table(model_comparison))
        if {"model", "horizon", "rmse"}.issubset(model_comparison.columns):
            st.plotly_chart(
                px.bar(
                    model_comparison,
                    x="model",
                    y="rmse",
                    color="source" if "source" in model_comparison.columns else None,
                    facet_col="horizon",
                    title="各模型與預測週期 RMSE",
                    labels={
                        "model": "模型",
                        "rmse": "RMSE",
                        "source": "來源",
                        "horizon": "預測週期",
                    },
                )
            )
    else:
        st.info("請先執行模型訓練與報告彙整。")

    transformer_metrics = artifacts["transformer_metrics"]
    if transformer_metrics:
        st.subheader("Transformer 摘要")
        st.json(transformer_metrics)

    transformer_predictions = artifacts["transformer_predictions"]
    if (
        isinstance(transformer_predictions, pd.DataFrame)
        and not transformer_predictions.empty
    ):
        st.subheader("Transformer 測試集預測")
        st.dataframe(display_table(transformer_predictions.head(200)))
        if {"horizon", "prediction"}.issubset(transformer_predictions.columns):
            st.plotly_chart(
                px.histogram(
                    transformer_predictions,
                    x="prediction",
                    color="horizon",
                    title="各預測週期的預測值分布",
                    labels={"prediction": "預測值", "horizon": "預測週期"},
                )
            )

    model_signals = artifacts["model_signals"]
    if isinstance(model_signals, pd.DataFrame) and not model_signals.empty:
        st.subheader("可供策略使用的模型訊號")
        st.dataframe(display_table(model_signals.head(200)))

with tabs[4]:
    st.subheader("回測明細")
    equity = artifacts["backtest_equity"]
    if isinstance(equity, pd.DataFrame) and not equity.empty:
        st.dataframe(display_table(equity))
        exposure = artifacts["backtest_exposure"]
        if isinstance(exposure, pd.DataFrame) and not exposure.empty:
            st.subheader("投資組合曝險")
            st.dataframe(display_table(exposure))
            st.plotly_chart(
                px.line(
                    exposure,
                    x="date",
                    y=["gross_exposure", "net_exposure"],
                    title="投資組合曝險",
                    labels={
                        "date": "日期",
                        "value": "曝險比例",
                        "variable": "曝險類型",
                    },
                )
            )
        sector_exposure = artifacts["backtest_sector_exposure"]
        if isinstance(sector_exposure, pd.DataFrame) and not sector_exposure.empty:
            st.subheader("產業曝險")
            st.dataframe(display_table(sector_exposure))
            st.plotly_chart(
                px.line(
                    sector_exposure,
                    x="date",
                    y="gross_exposure",
                    color="sector",
                    title="產業總曝險",
                    labels={
                        "date": "日期",
                        "gross_exposure": "總曝險",
                        "sector": "產業",
                    },
                )
            )
        trades = artifacts["backtest_trades"]
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            st.subheader("交易紀錄")
            st.dataframe(display_table(trades))
        attribution_metrics = artifacts["attribution_metrics"]
        if attribution_metrics:
            st.subheader("報酬貢獻集中度")
            st.json(attribution_metrics)
        attribution_summary = artifacts["attribution_summary"]
        if (
            isinstance(attribution_summary, pd.DataFrame)
            and not attribution_summary.empty
        ):
            st.subheader("個股報酬歸因")
            st.dataframe(display_table(attribution_summary))
            if {"ticker", "total_contribution"}.issubset(
                attribution_summary.columns
            ):
                st.plotly_chart(
                    px.bar(
                        attribution_summary.head(20),
                        x="ticker",
                        y="total_contribution",
                        title="主要報酬貢獻股票",
                        labels={
                            "ticker": "股票代號",
                            "total_contribution": "總貢獻",
                        },
                    )
                )
    else:
        st.info("請先執行回測以產生本頁內容。")

with tabs[5]:
    st.subheader("工作流程執行器")
    st.caption("僅能透過已測試的後端模組執行已登錄的 QuantLab 指令。")
    for step in list_workflow_steps():
        with st.expander(step.label):
            st.write(step.description)
            st.code("python " + " ".join(step.command), language="powershell")
            if st.button(f"執行：{step.label}", key=f"run_{step.name}"):
                with st.spinner(f"正在執行 {step.label}..."):
                    result = run_workflow_step(step.name)
                if result["success"]:
                    st.success(f"{step.label} 已完成。")
                else:
                    st.error(f"{step.label} 執行失敗，返回碼 {result['returncode']}。")
                if result["stdout"]:
                    st.text_area("標準輸出", result["stdout"], height=160)
                if result["stderr"]:
                    st.text_area("錯誤輸出", result["stderr"], height=160)

    st.subheader("近期執行紀錄")
    runs = latest_workflow_runs()
    if runs:
        st.dataframe(
            display_table(pd.DataFrame(
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
            ))
        )
    else:
        st.info("目前尚無工作流程執行紀錄。")

with tabs[6]:
    st.subheader("設定檔編輯器")
    st.caption("此處僅允許編輯已登錄的 QuantLab YAML 設定檔。")
    entries = {entry.name: entry for entry in list_config_entries()}
    selected = st.selectbox("設定檔", options=list(entries))
    st.write(entries[selected].description)
    try:
        current_text = read_config_text(selected)
    except FileNotFoundError as exc:
        st.error(str(exc))
        current_text = ""
    edited = st.text_area("YAML", current_text, height=360)
    if st.button("驗證並儲存設定"):
        try:
            parsed = write_config_text(selected, edited)
        except Exception as exc:
            st.error(f"設定檔未儲存：{exc}")
        else:
            st.success(f"已儲存 {selected}，共 {len(parsed)} 個頂層欄位。")

with tabs[7]:
    st.subheader("策略工作室")
    st.caption(
        "建立版本化策略、執行回測、比較結果，並搜尋符合限制條件的候選方案。"
    )
    lifecycle = StrategyLifecycleManager()
    strategy_paths = sorted(Path("strategies").glob("*.yaml"))
    if strategy_paths:
        selected_strategy = st.selectbox(
            "策略規格",
            options=strategy_paths,
            format_func=lambda path: path.name,
        )
        editing_mode = st.radio(
            "編輯模式",
            options=["視覺化策略精靈", "進階 YAML"],
            horizontal=True,
        )
        if editing_mode == "視覺化策略精靈":
            base_spec = load_strategy_spec(selected_strategy)
            defaults = strategy_to_wizard_values(base_spec)
            with st.form("strategy_wizard"):
                st.markdown("#### 1. 基本資訊")
                basic_name, basic_version = st.columns([2, 1])
                name = basic_name.text_input(
                    "策略名稱",
                    value=defaults.name,
                    help="修改名稱時會自動建立新的策略 YAML 檔案。",
                )
                version = basic_version.text_input(
                    "版本",
                    value=defaults.version,
                    help="建議使用 1.0.0 這類語意化版本。",
                )
                description = st.text_area(
                    "策略說明",
                    value=defaults.description,
                    height=80,
                )

                st.markdown("#### 2. 資料與訊號")
                market_path = st.text_input(
                    "市場資料路徑",
                    value=defaults.market_path,
                )
                signal_path = st.text_input(
                    "訊號資料路徑",
                    value=defaults.signal_path,
                )
                signal_left, signal_middle, signal_right = st.columns(3)
                score_column = signal_left.text_input(
                    "策略分數欄位",
                    value=defaults.score_column,
                )
                volatility_column = signal_middle.text_input(
                    "波動率欄位（選填）",
                    value=defaults.volatility_column,
                )
                sector_column = signal_right.text_input(
                    "產業欄位（選填）",
                    value=defaults.sector_column,
                )
                st.caption(
                    "多訊號元件會依權重合成策略分數；留空代表直接使用分數欄位。"
                )
                component_frame = st.data_editor(
                    components_to_frame(base_spec.signal.components),
                    num_rows="dynamic",
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "column": st.column_config.TextColumn("訊號欄位"),
                        "weight": st.column_config.NumberColumn(
                            "權重",
                            format="%.3f",
                        ),
                        "transform": st.column_config.SelectboxColumn(
                            "轉換方式",
                            options=[
                                "raw",
                                "cross_sectional_rank",
                                "cross_sectional_zscore",
                            ],
                        ),
                    },
                )

                st.markdown("#### 3. 股票池與投資組合")
                universe_left, universe_middle, universe_right = st.columns(3)
                excluded_tickers = universe_left.text_input(
                    "排除股票（逗號分隔）",
                    value=defaults.excluded_tickers,
                )
                start_date = universe_middle.text_input(
                    "回測起始日（選填）",
                    value=defaults.start_date,
                    placeholder="YYYY-MM-DD",
                )
                end_date = universe_right.text_input(
                    "回測結束日（選填）",
                    value=defaults.end_date,
                    placeholder="YYYY-MM-DD",
                )
                strategy_types = ["long_only_top_k", "long_short_quantile"]
                weighting_options = [
                    "equal",
                    "score",
                    "inverse_volatility",
                    "risk_parity",
                ]
                rebalance_options = ["D", "W-FRI", "ME"]
                portfolio_left, portfolio_middle, portfolio_right = st.columns(3)
                strategy_type = portfolio_left.selectbox(
                    "組合類型",
                    options=strategy_types,
                    index=strategy_types.index(defaults.strategy_type),
                    format_func=lambda value: {
                        "long_only_top_k": "僅做多 Top-K",
                        "long_short_quantile": "多空分位數",
                    }[value],
                )
                weighting = portfolio_middle.selectbox(
                    "權重方式",
                    options=weighting_options,
                    index=weighting_options.index(defaults.weighting),
                    format_func=lambda value: {
                        "equal": "等權重",
                        "score": "依分數加權",
                        "inverse_volatility": "反波動率",
                        "risk_parity": "風險平價",
                    }[value],
                )
                rebalance_frequency = portfolio_right.selectbox(
                    "再平衡頻率",
                    options=rebalance_options,
                    index=(
                        rebalance_options.index(defaults.rebalance_frequency)
                        if defaults.rebalance_frequency in rebalance_options
                        else 1
                    ),
                    format_func=lambda value: {
                        "D": "每日",
                        "W-FRI": "每週五",
                        "ME": "每月底",
                    }[value],
                )
                portfolio_size, long_side, short_side = st.columns(3)
                top_k = portfolio_size.number_input(
                    "Top-K 持股數",
                    min_value=1,
                    value=defaults.top_k,
                    step=1,
                )
                long_quantile = long_side.number_input(
                    "多頭分位比例",
                    min_value=0.01,
                    max_value=0.5,
                    value=defaults.long_quantile,
                    step=0.05,
                )
                short_quantile = short_side.number_input(
                    "空頭分位比例",
                    min_value=0.01,
                    max_value=0.5,
                    value=defaults.short_quantile,
                    step=0.05,
                )

                st.markdown("#### 4. 風險限制")
                risk_toggle_1, risk_toggle_2, risk_toggle_3 = st.columns(3)
                use_position_limit = risk_toggle_1.checkbox(
                    "啟用單一持股上限",
                    value=defaults.use_position_limit,
                )
                use_sector_limit = risk_toggle_2.checkbox(
                    "啟用產業權重上限",
                    value=defaults.use_sector_limit,
                )
                use_liquidity_limit = risk_toggle_3.checkbox(
                    "啟用最低流動性",
                    value=defaults.use_liquidity_limit,
                )
                risk_value_1, risk_value_2, risk_value_3 = st.columns(3)
                max_position_weight = risk_value_1.number_input(
                    "單一持股權重上限",
                    min_value=0.01,
                    max_value=1.0,
                    value=defaults.max_position_weight,
                    step=0.01,
                    disabled=not use_position_limit,
                )
                max_sector_weight = risk_value_2.number_input(
                    "產業權重上限",
                    min_value=0.01,
                    max_value=1.0,
                    value=defaults.max_sector_weight,
                    step=0.05,
                    disabled=not use_sector_limit,
                )
                min_dollar_volume = risk_value_3.number_input(
                    "最低日均成交額",
                    min_value=0.0,
                    value=defaults.min_dollar_volume,
                    step=1_000_000.0,
                    disabled=not use_liquidity_limit,
                )
                liquidity_window = st.number_input(
                    "流動性計算天數",
                    min_value=1,
                    value=defaults.liquidity_window,
                    step=1,
                )

                st.markdown("#### 5. 交易與成本")
                execution_1, execution_2, execution_3, execution_4 = st.columns(4)
                initial_capital = execution_1.number_input(
                    "初始資金",
                    min_value=1.0,
                    value=defaults.initial_capital,
                    step=100_000.0,
                )
                commission_bps = execution_2.number_input(
                    "手續費（基點）",
                    min_value=0.0,
                    value=defaults.commission_bps,
                    step=1.0,
                )
                slippage_bps = execution_3.number_input(
                    "滑價（基點）",
                    min_value=0.0,
                    value=defaults.slippage_bps,
                    step=1.0,
                )
                execution_lag_days = execution_4.number_input(
                    "執行延遲天數",
                    min_value=0,
                    value=defaults.execution_lag_days,
                    step=1,
                )
                rate_1, rate_2, rate_3 = st.columns(3)
                annual_cash_rate = rate_1.number_input(
                    "年化現金利率",
                    value=defaults.annual_cash_rate,
                    step=0.005,
                    format="%.4f",
                )
                annual_borrow_rate = rate_2.number_input(
                    "年化借券費率",
                    min_value=0.0,
                    value=defaults.annual_borrow_rate,
                    step=0.01,
                    format="%.4f",
                )
                risk_free_rate = rate_3.number_input(
                    "無風險利率",
                    value=defaults.risk_free_rate,
                    step=0.005,
                    format="%.4f",
                )
                benchmark_ticker = st.text_input(
                    "比較基準股票代號",
                    value=defaults.benchmark_ticker,
                )
                save_wizard, run_wizard = st.columns(2)
                save_requested = save_wizard.form_submit_button(
                    "驗證並儲存策略",
                    use_container_width=True,
                )
                run_requested = run_wizard.form_submit_button(
                    "儲存並立即回測",
                    use_container_width=True,
                    type="primary",
                )

            if save_requested or run_requested:
                try:
                    wizard_values = StrategyWizardValues(
                        name=name,
                        version=version,
                        description=description,
                        market_path=market_path,
                        signal_path=signal_path,
                        score_column=score_column,
                        volatility_column=volatility_column,
                        sector_column=sector_column,
                        excluded_tickers=excluded_tickers,
                        start_date=start_date,
                        end_date=end_date,
                        strategy_type=strategy_type,
                        top_k=int(top_k),
                        long_quantile=long_quantile,
                        short_quantile=short_quantile,
                        weighting=weighting,
                        rebalance_frequency=rebalance_frequency,
                        use_position_limit=use_position_limit,
                        max_position_weight=max_position_weight,
                        use_sector_limit=use_sector_limit,
                        max_sector_weight=max_sector_weight,
                        use_liquidity_limit=use_liquidity_limit,
                        min_dollar_volume=min_dollar_volume,
                        liquidity_window=int(liquidity_window),
                        initial_capital=initial_capital,
                        commission_bps=commission_bps,
                        slippage_bps=slippage_bps,
                        execution_lag_days=int(execution_lag_days),
                        annual_cash_rate=annual_cash_rate,
                        annual_borrow_rate=annual_borrow_rate,
                        benchmark_ticker=benchmark_ticker,
                        risk_free_rate=risk_free_rate,
                    )
                    spec = build_strategy_from_wizard(
                        base_spec,
                        wizard_values,
                        components_from_frame(component_frame),
                    )
                    output_path = (
                        selected_strategy
                        if spec.slug == base_spec.slug
                        else strategy_output_path(spec)
                    )
                    lifecycle.save(spec, output_path)
                    if run_requested:
                        result = StrategyStudioRunner().run(spec)
                except Exception as exc:
                    st.error(f"策略處理失敗：{exc}")
                else:
                    st.success(
                        f"已儲存至 {output_path}；規格雜湊 {spec.spec_hash[:8]}。"
                    )
                    if run_requested:
                        st.success(f"回測已完成：{result.run_id}")
                        st.json(result.metrics)
        else:
            strategy_text = st.text_area(
                "策略規格 YAML",
                selected_strategy.read_text(encoding="utf-8"),
                height=500,
            )
            save_column, run_column = st.columns(2)
            if save_column.button("驗證並儲存 YAML"):
                try:
                    payload = yaml.safe_load(strategy_text) or {}
                    spec = strategy_spec_from_dict(payload)
                    lifecycle.save(spec, selected_strategy)
                except Exception as exc:
                    st.error(f"策略未儲存：{exc}")
                else:
                    st.success(
                        f"已儲存 {spec.name} {spec.version} "
                        f"({spec.spec_hash[:8]})。"
                    )
            if run_column.button("執行 YAML 策略"):
                try:
                    payload = yaml.safe_load(strategy_text) or {}
                    spec = strategy_spec_from_dict(payload)
                    result = StrategyStudioRunner().run(spec)
                except Exception as exc:
                    st.error(f"策略執行失敗：{exc}")
                else:
                    st.success(f"已完成 {result.run_id}")
                    st.json(result.metrics)
    else:
        st.info("請在 strategies/ 目錄新增策略規格 YAML 檔案。")

    st.divider()
    st.subheader("策略版本與生命週期")
    st.caption("所有覆寫都會保留舊版；刪除採可還原的軟刪除方式。")
    if strategy_paths:
        current_spec = load_strategy_spec(selected_strategy)
        versions = lifecycle.versions(selected_strategy)
        version_table = pd.DataFrame(
            [
                {
                    "status": record.status,
                    "version": record.version,
                    "spec_hash": record.spec_hash[:12],
                    "path": str(record.path),
                }
                for record in versions
            ]
        )
        st.dataframe(display_table(version_table), hide_index=True)

        if len(versions) >= 2:
            version_left, version_right = st.columns(2)
            left_version = version_left.selectbox(
                "版本 A",
                options=versions,
                format_func=lambda record: record.label,
                key="lifecycle_left_version",
            )
            right_version = version_right.selectbox(
                "版本 B",
                options=versions,
                index=1,
                format_func=lambda record: record.label,
                key="lifecycle_right_version",
            )
            if st.button("比較策略版本差異"):
                version_diff = compare_strategy_versions(
                    load_strategy_spec(left_version.path),
                    load_strategy_spec(right_version.path),
                )
                if version_diff.empty:
                    st.info("兩個版本內容完全相同。")
                else:
                    st.dataframe(display_table(version_diff), hide_index=True)
        else:
            st.info("目前只有一個版本；下次修改並儲存後會自動保留舊版。")

        with st.expander("複製為新策略"):
            duplicate_name = st.text_input(
                "新策略名稱",
                value=f"{current_spec.name}-copy",
                key="lifecycle_duplicate_name",
            )
            duplicate_version = st.text_input(
                "新策略版本",
                value="1.0.0",
                key="lifecycle_duplicate_version",
            )
            if st.button("建立策略副本"):
                try:
                    duplicate_path = lifecycle.duplicate(
                        selected_strategy,
                        duplicate_name,
                        duplicate_version,
                    )
                except Exception as exc:
                    st.error(f"策略複製失敗：{exc}")
                else:
                    st.success(f"已建立策略副本：{duplicate_path}")

        archive_column, delete_column = st.columns(2)
        with archive_column:
            st.markdown("#### 封存")
            st.caption("封存後不再出現在啟用策略清單，但可隨時還原。")
            if st.button("封存目前策略", use_container_width=True):
                try:
                    archive_path = lifecycle.archive(selected_strategy)
                except Exception as exc:
                    st.error(f"策略封存失敗：{exc}")
                else:
                    st.success(f"策略已封存：{archive_path}")
        with delete_column:
            st.markdown("#### 軟刪除")
            confirmation_name = st.text_input(
                f"輸入策略名稱「{current_spec.name}」以啟用刪除",
                key="lifecycle_delete_confirmation",
            )
            if st.button(
                "將目前策略移至垃圾區",
                disabled=confirmation_name != current_spec.name,
                use_container_width=True,
            ):
                try:
                    deleted_path = lifecycle.soft_delete(
                        selected_strategy,
                        confirmation_name,
                    )
                except Exception as exc:
                    st.error(f"策略刪除失敗：{exc}")
                else:
                    st.success(f"策略已移至垃圾區：{deleted_path}")

    recoverable = [*lifecycle.archived(), *lifecycle.deleted()]
    if recoverable:
        with st.expander("還原封存或已刪除策略"):
            restore_record = st.selectbox(
                "選擇要還原的策略",
                options=recoverable,
                format_func=lambda record: (
                    f"{record.status} | {record.name} | {record.version}"
                ),
                key="lifecycle_restore_record",
            )
            if st.button("還原策略"):
                try:
                    restored_path = lifecycle.restore(restore_record.path)
                except Exception as exc:
                    st.error(f"策略還原失敗：{exc}")
                else:
                    st.success(f"策略已還原：{restored_path}")

    st.divider()
    st.subheader("Walk-forward 樣本外評估")
    st.caption(
        "使用凍結策略、purge gap 與互不重疊測試窗，拼接真正的滾動 OOS 績效。"
    )
    if strategy_paths and selected_strategy.exists():
        with st.form("walk_forward_form"):
            wf_1, wf_2, wf_3, wf_4 = st.columns(4)
            wf_train_days = wf_1.number_input(
                "訓練窗交易日",
                min_value=20,
                value=120,
                step=20,
            )
            wf_test_days = wf_2.number_input(
                "測試窗交易日",
                min_value=5,
                value=20,
                step=5,
            )
            wf_step_days = wf_3.number_input(
                "每次前進交易日",
                min_value=5,
                value=20,
                step=5,
            )
            wf_purge_days = wf_4.number_input(
                "Purge gap 交易日",
                min_value=0,
                value=5,
                step=1,
            )
            wf_anchored = st.checkbox(
                "使用 Anchored 訓練窗（起點固定）",
                value=False,
            )
            run_walk_forward = st.form_submit_button(
                "執行 Walk-forward 評估",
                type="primary",
                use_container_width=True,
            )
        wf_output = Path("artifacts/studio/walk_forward") / current_spec.slug
        if run_walk_forward:
            try:
                wf_result = WalkForwardEvaluator(
                    WalkForwardConfig(
                        strategy_spec_path=selected_strategy,
                        output_dir=wf_output,
                        train_days=int(wf_train_days),
                        test_days=int(wf_test_days),
                        step_days=int(wf_step_days),
                        purge_days=int(wf_purge_days),
                        anchored_train=wf_anchored,
                    )
                ).run()
            except Exception as exc:
                st.error(f"Walk-forward 評估失敗：{exc}")
            else:
                st.success(f"已完成 {len(wf_result.folds)} 個 OOS folds。")
                st.dataframe(display_table(wf_result.folds), hide_index=True)
                st.json(wf_result.metrics)
        wf_folds_path = wf_output / "folds.csv"
        wf_equity_path = wf_output / "oos_equity_curve.csv"
        if wf_folds_path.exists():
            st.markdown("#### 最新 Walk-forward 結果")
            latest_folds = pd.read_csv(wf_folds_path)
            st.dataframe(display_table(latest_folds), hide_index=True)
            if wf_equity_path.exists():
                latest_wf_equity = pd.read_csv(wf_equity_path)
                st.plotly_chart(
                    px.line(
                        latest_wf_equity,
                        x="date",
                        y="nav",
                        title="Walk-forward 樣本外淨值曲線",
                        labels={"date": "日期", "nav": "淨值"},
                    )
                )
    else:
        st.info("請先建立或還原一個啟用中的策略。")

    registry = StrategyRunRegistry()
    run_summary = registry.summary()
    st.subheader("已儲存的執行紀錄")
    if run_summary.empty:
        st.info("目前尚無策略工作室執行紀錄。")
    else:
        st.dataframe(display_table(run_summary), hide_index=True)
        selected_runs = st.multiselect(
            "選擇要比較的執行紀錄",
            options=run_summary["run_id"].tolist(),
        )
        if st.button("比較所選紀錄"):
            try:
                comparison = compare_strategy_runs(
                    selected_runs,
                    registry,
                    "artifacts/studio/comparisons/gui_latest",
                )
            except Exception as exc:
                st.error(f"執行結果比較失敗：{exc}")
            else:
                st.dataframe(display_table(comparison), hide_index=True)
                st.plotly_chart(
                    px.scatter(
                        comparison,
                        x="max_drawdown",
                        y="annual_return",
                        color="sharpe_ratio",
                        hover_name="portfolio",
                        title="共同期間風險與報酬",
                        labels={
                            "max_drawdown": "最大回撤",
                            "annual_return": "年化報酬率",
                            "sharpe_ratio": "夏普比率",
                            "portfolio": "投資組合",
                        },
                    )
                )

    st.subheader("策略優化器")
    st.caption(
        "搜尋參數組合、套用限制條件，並在共同期間標示 Pareto 有效率候選方案。"
    )
    if st.button("執行設定好的策略優化"):
        try:
            optimization = StrategyOptimizer(
                load_optimization_config("configs/studio_optimizer.yaml")
            ).run()
        except Exception as exc:
            st.error(f"策略優化失敗：{exc}")
        else:
            st.dataframe(display_table(optimization), hide_index=True)
    optimization_path = Path("artifacts/studio/optimizations/factor_search/results.csv")
    if optimization_path.exists():
        latest_optimization = pd.read_csv(optimization_path)
        st.dataframe(display_table(latest_optimization), hide_index=True)
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
                    title="策略候選方案 Pareto 前緣",
                    labels={
                        "max_drawdown": "最大回撤",
                        "annual_return": "年化報酬率",
                        "pareto_efficient": "Pareto 有效率",
                        "run_id": "執行編號",
                    },
                )
            )

    st.subheader("個人化候選方案")
    st.caption(
        "依設定的風險承受度、回撤、換手率與報酬限制篩選。排名僅代表歷史候選結果，"
        "不構成報酬保證。"
    )
    if st.button("建立個人需求推薦"):
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
            st.error(f"推薦建立失敗：{exc}")
        else:
            st.dataframe(display_table(recommendations), hide_index=True)
    recommendation_path = Path(
        "artifacts/studio/recommendations/recommendations.csv"
    )
    if recommendation_path.exists():
        st.dataframe(display_table(pd.read_csv(recommendation_path)), hide_index=True)
