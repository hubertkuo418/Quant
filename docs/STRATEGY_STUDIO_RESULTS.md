# QuantLab Strategy Studio Results

> Generated from registered strategy artifacts. Historical results are not investment advice or future-return guarantees.

## Platform Snapshot

- Registered strategy runs: 27
- Distinct strategy names: 3
- Optimizer candidates: 12
- Feasible candidates: 12
- Pareto candidates: 4

## Core Strategy Comparison

| portfolio | common_start | common_end | annual_return | sharpe_ratio | max_drawdown | average_turnover |
| --- | --- | --- | --- | --- | --- | --- |
| 20260618T211452778590Z_momentum-top10_2733624e | 2026-02-27 | 2026-06-17 | 0.4322 | 2.1441 | -0.0646 | 0.1481 |
| 20260618T210518293703Z_factor-top10_72f858a6 | 2026-02-27 | 2026-06-17 | 0.3391 | 1.7380 | -0.0861 | 0.1481 |
| 20260618T211454259961Z_momentum-low-vol_9213e830 | 2026-02-27 | 2026-06-17 | 0.0950 | 0.7437 | -0.0801 | 0.1711 |

## Profile Recommendations

| recommendation_rank | run_id | recommendation_score | annual_return | sharpe_ratio | max_drawdown | average_turnover | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | 20260618T210627596459Z_factor-top10_9bed54c7 | 0.8625 | 0.8078 | 2.6032 | -0.0678 | 0.1632 | balanced profile; Sharpe 2.60, drawdown -6.8%, turnover 16.3%, worst-lag Sharpe 1.76 |
| 2.0000 | 20260618T210629717182Z_factor-top10_8860a7c0 | 0.8125 | 0.4062 | 2.0157 | -0.0748 | 0.1500 | balanced profile; Sharpe 2.02, drawdown -7.5%, turnover 15.0%, worst-lag Sharpe 1.83 |
| 3.0000 | 20260618T210629470628Z_factor-top10_19e73424 | 0.6875 | 0.3640 | 1.8336 | -0.0822 | 0.1368 | balanced profile; Sharpe 1.83, drawdown -8.2%, turnover 13.7%, worst-lag Sharpe 1.83 |
| 4.0000 | 20260618T210628523091Z_factor-top10_b6a56e95 | 0.5750 | 0.4015 | 1.7723 | -0.0715 | 0.2073 | balanced profile; Sharpe 1.77, drawdown -7.1%, turnover 20.7%, worst-lag Sharpe 0.74 |
| 5.0000 | 20260618T210627921017Z_factor-top10_68477841 | 0.5500 | 0.4751 | 1.7562 | -0.0853 | 0.1763 | balanced profile; Sharpe 1.76, drawdown -8.5%, turnover 17.6%, worst-lag Sharpe 1.76 |

## Frozen-Strategy Walk-Forward OOS

- Folds: 3
- Total return: 0.1607
- Sharpe ratio: 4.3349
- Maximum drawdown: -0.0436

| fold | train_start | train_end | test_start | test_end | total_return | sharpe_ratio | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10.0000 | 2025-09-10 | 2026-03-03 | 2026-03-11 | 2026-04-08 | 0.0004 | 0.1109 | -0.0436 |
| 11.0000 | 2025-10-08 | 2026-03-31 | 2026-04-09 | 2026-05-06 | 0.1315 | 9.3795 | -0.0100 |
| 12.0000 | 2025-11-05 | 2026-04-29 | 2026-05-07 | 2026-06-04 | 0.0254 | 2.7320 | -0.0149 |

## Unified OOS Robustness

- Scenarios: 6
- Pass rate: 100.00%
- Worst Sharpe: 3.3492
- Worst maximum drawdown: -0.0453

| scenario | oos_observations | sharpe_ratio | max_drawdown | fold_sharpe_min | positive_fold_rate | passes_constraints |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 60.0000 | 4.1824 | -0.0436 | 0.1031 | 1.0000 | True |
| double_costs | 60.0000 | 3.9412 | -0.0453 | -0.1752 | 0.6667 | True |
| lag_plus_one | 60.0000 | 3.9809 | -0.0387 | -0.1839 | 0.6667 | True |
| top_k_minus_two | 60.0000 | 4.3360 | -0.0362 | 1.1980 | 1.0000 | True |
| top_k_plus_two | 60.0000 | 4.1862 | -0.0405 | 0.7252 | 1.0000 | True |
| monthly_rebalance | 60.0000 | 3.3492 | -0.0137 | 0.3925 | 1.0000 | True |

## Interpretation Boundaries

- The live data window is short and uses a present-day static universe.
- Nasdaq close is used as adjusted close; corporate actions may distort returns.
- Optimizer results are recomputed on a common period before ranking.
- Execution lag 1/2 sensitivity is included in recommendation robustness.
- Candidate rankings describe historical fit under configured constraints.
- Walk-forward uses a frozen strategy and does not refit factors inside each fold.
- The current stitched OOS result has only three short folds; one strong fold dominates the aggregate.
