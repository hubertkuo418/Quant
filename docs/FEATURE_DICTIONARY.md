# Feature Dictionary

All features are calculated per ticker and may use only observations available
on or before the row's `date`.

| Feature | Definition | Earliest valid row |
|---|---|---|
| `return_1d` | Adjusted-close trailing 1-day return | 2nd observation |
| `return_5d` | Adjusted-close trailing 5-day return | 6th observation |
| `return_20d` | Adjusted-close trailing 20-day return | 21st observation |
| `volatility_20d` | Standard deviation of trailing daily returns | 21st observation |
| `rsi_14d` | Simple-average 14-day Relative Strength Index | 15th observation |
| `price_ma_20d_ratio` | Adjusted close / trailing 20-day mean - 1 | 20th observation |
| `price_ma_60d_ratio` | Adjusted close / trailing 60-day mean - 1 | 60th observation |
| `log_volume_zscore_20d` | Log-volume z-score over trailing 20 observations | 20th observation |
| `avg_dollar_volume_20d` | Trailing mean adjusted close times volume | 20th observation |
| `momentum_63d` | Adjusted-close trailing 3-month return | 64th observation |
| `momentum_126d` | Adjusted-close trailing 6-month return | 127th observation |
| `momentum_252d` | Adjusted-close trailing 12-month return | 253rd observation |
| `volatility_60d` | Standard deviation of trailing daily returns | 61st observation |
| `max_drawdown_60d` | Worst drawdown inside trailing 60-day window | 60th observation |
| `macd` | EMA(12) minus EMA(26) | 1st observation |
| `macd_signal` | EMA(9) of MACD | 1st observation |
| `macd_histogram` | MACD minus signal line | 1st observation |
| `atr_14d` | 14-day average true range | 14th observation |
| `atr_14d_ratio` | ATR divided by adjusted close | 14th observation |
| `bollinger_percent_b_20d` | Location inside 20-day Bollinger bands | 20th observation |
| `bollinger_bandwidth_20d` | Bollinger band width divided by moving average | 20th observation |
| `pe_ratio` | Latest published P/E with `available_at <= date` | Source-dependent |
| `pb_ratio` | Latest published P/B with `available_at <= date` | Source-dependent |
| `news_sentiment` | Mean news score usable on that trading date | Same date |
| `news_article_count` | Scored article count usable on that date | Same date |

## Optional Input Contracts

Fundamentals CSV or Parquet:

```text
ticker,available_at,pe_ratio,pb_ratio
AAPL,2024-02-02,28.1,42.3
```

`available_at` is the first trading date on which the value was publicly known
and usable. It is not the fiscal period end.

News CSV or Parquet:

```text
ticker,available_at,headline,sentiment_score
AAPL,2024-02-02,Apple beats estimates,0.8
```

`sentiment_score` is optional when `headline` exists; the offline MVP lexicon
will score the headline. Articles published after the execution cutoff must be
shifted to the next observed market date. Configure `news_market_timezone` and
`news_cutoff_time` in `configs/features.yaml`; the default is New York 09:30.
