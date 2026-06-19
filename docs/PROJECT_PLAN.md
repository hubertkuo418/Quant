# QuantLab Strategy Studio 完整計畫書

## 1. 產品定位

QuantLab Strategy Studio 是一個可自由組合、回測、比較、優化與版本化量化策略的平台。因子、技術指標、新聞情緒、傳統機器學習、LSTM 與 Transformer 都是可替換的訊號元件，不再由單一模型定義整個專案。

核心流程如下：

```text
市場資料 -> 特徵與訊號 -> StrategySpec -> 投資組合與風控
-> 成本感知回測 -> Run Registry -> 比較 / 優化 / 推薦 -> 報告
```

## 2. 產品目標

1. 使用 YAML 或 GUI 建立、修改及保存策略。
2. 讓不同訊號共用相同的投資組合、成本、風控與回測契約。
3. 保存每次執行的規格、資料雜湊、環境與完整績效產物。
4. 在共同期間比較策略，避免日期差異造成不公平排名。
5. 依報酬、風險、換手率與穩健性限制搜尋候選方案。
6. 依使用者需求產生可解釋的候選排名，而非宣稱保證適合。

## 3. V1 範圍

- 標的：可設定的美國流動性股票與 ETF universe。
- 資料：Nasdaq 日 OHLCV 或經驗證的本地 CSV。
- 介面：Python backend、YAML 規格、CLI 與 Streamlit GUI。
- 訊號：技術、動能、低波動、多因子、模型預測與外部 signal panel。
- 組合：long-only Top-K 與 long-short quantile。
- 回測：交易成本、滑價、執行延遲、現金、借券與風險限制。
- 評估：Sharpe、CAGR、最大回撤、換手率、尾部風險及基準比較。

## 4. 系統模組

### Data Center

- Nasdaq provider、CSV importer、raw/processed Parquet 與 DuckDB catalog。
- 統一 `(date, ticker, OHLCV)` schema。
- 日期覆蓋、價格停滯、零成交量與異常報酬檢查。
- 資料來源、下載時間、SHA-256 與 adjusted-close policy。

### Feature Library

- 報酬、動能、波動、RSI、均線、MACD、ATR、Bollinger 與流動性。
- Point-in-time fundamental 與 news sentiment join。
- Formulaic alpha registry、IC、Rank IC、分組報酬與因子選擇。

### StrategySpec

- 名稱、版本、說明、universe、訊號來源與組合方式。
- Top-K、quantile、權重、再平衡頻率與持有規則。
- 流動性、單一部位、產業與曝險限制。
- Commission、slippage、execution lag、cash rate 與 borrow rate。
- Benchmark、初始資金與穩定 SHA-256 規格雜湊。

### Backtest Engine

- Holdings、cash、NAV、turnover、trade log 與 exposure。
- 成本、滑價、延遲、流動性與借券成本。
- Sharpe、Sortino、Calmar、drawdown、VaR、CVaR 與 win rate。

### Run Registry

每次執行建立不可變的 `run_id`，保存：

- StrategySpec、spec hash、market hash 與 signal hash。
- Target weights、equity curve、holdings、trades 與 metrics。
- Python、套件版本、Git commit 與父策略版本。

### Comparison Center

- 多個 run 的共同期間績效與 rebased equity curve。
- 基準、回撤、regime、敏感度與貢獻集中度分析。
- 禁止直接比較不同觀測期間產生的原始年化數字。

### Strategy Optimizer

- Grid 與 random search。
- 可設定 Top-K、權重、再平衡、成本與風險參數。
- 先套用限制，再依共同期間結果建立 Pareto candidates。
- 正式候選禁止使用零日 execution lag。

### Recommendation Engine

- Conservative、balanced、aggressive 使用者 profile。
- 最大回撤、換手率、最低報酬與延遲穩健性限制。
- 提供可解釋排名與淘汰原因，不把歷史排名描述為投資保證。

### Streamlit Studio

1. Strategy Builder
2. Backtest Runner
3. Run Registry
4. Strategy Comparison
5. Strategy Optimizer
6. Profile Recommendations
7. Data、Factor、Model 與 Report views

## 5. 實作階段與狀態

| Phase | 內容 | 狀態 |
|---|---|---|
| 1 | Data pipeline、CSV、quality、DuckDB | 完成 |
| 2 | Feature、factor、alpha 與 signal library | 完成 |
| 3 | StrategySpec、YAML round trip 與 hash | 完成 |
| 4 | Strategy runner 與 immutable run artifacts | 完成 |
| 5 | Registry 與 common-period comparison | 完成 |
| 6 | Grid/random optimizer、constraints、Pareto | 完成 |
| 7 | Profile recommendation 與 lag robustness | 完成 |
| 8 | Streamlit Strategy Studio | 完成 |
| 9 | 自動成果報告、文件與履歷素材 | 完成 |
| 10 | 測試、靜態檢查與 GitHub release | 完成 |

## 6. 驗收標準

- 使用者能從 YAML 或 GUI 建立並重跑策略。
- 每個 run 都能追溯規格、資料、程式版本與績效產物。
- 策略比較一律使用共同期間。
- Optimizer 能套用限制並輸出 Pareto candidates。
- 推薦結果包含 profile、排名、穩健性與可解釋理由。
- 真實資料結果與 synthetic smoke 結果清楚分開。
- 完整測試、Ruff、`git diff --check` 與瀏覽器 GUI 驗證通過。

## 7. 已知限制

- 現有 live history 較短，不能視為跨景氣循環的長期證據。
- 目前 universe 是今日靜態清單，存在 survivorship bias。
- Nasdaq endpoint 無 adjusted close，目前以 close 代替，企業行動可能扭曲報酬。
- Point-in-time fundamentals 與 news join 已完成，但正式資料仍需授權來源。
- 推薦引擎提供歷史條件下的決策支援，不是個人化投資顧問。

## 8. V2 Roadmap

- Point-in-time historical universe 與 corporate-action adjusted data。
- Licensed fundamental/news connectors 與完整 Alpha101。
- Adaptive per-fold model refitting、Optuna/Bayesian optimizer。
- Covariance-aware portfolio optimization 與進階產業限制。
- Plugin-style Python strategy SDK、paper trading 與 broker adapter。
- 多使用者策略權限、排程、監控與部署。
