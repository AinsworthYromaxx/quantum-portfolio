# V4 Ultimate — Project Explanation

## What it is

A quantitative equity selection system that combines a multi-factor scoring engine, a quantum-inspired wave function portfolio constructor, empirical phase calibration, walk-forward validation, macro regime detection, and a Streamlit dashboard. The output is a ranked portfolio of 50 US equities with per-position weights, IBKR-ready order JSON, and Monte Carlo risk projections.

---

## Directory Layout

```
v4_ultimate/
  dashboard.py            Main Streamlit app (port 8502)
  launch_dashboard.bat    Runs: streamlit run dashboard.py --server.port 8502

  notebooks/
    StockPickerV4_Ultimate.ipynb   Master notebook — runs the full pipeline
    PortfolioReport_V4.ipynb       Post-run report generation

  src/
    stock_analysis.py     Universe fetch, price download, composite scoring, rebalancing signals
    quantum_portfolio.py  Wave function construction, portfolio collapse, Monte Carlo
    regime_detector.py    Macro regime classification (BULL/NEUTRAL/BEAR)
    transaction_model.py  Transaction cost estimation
    data_providers.py     Fundamental and alternative data (FMP, FRED, Finnhub)

  cache/
    v3/                   Finviz ticker list (24h TTL), FRED JSON files (1h TTL)
    expanded/             Fundamentals, macro, alternative, global sub-caches

  output/
    v3/
      ultimate_portfolio_MMDDYYYY.csv   Final weighted portfolio
      calibration_meta.json             All calibration outputs + MC stats
      ibkr_orders.json                  Share-count buy orders
      *.png                             All charts (heatmap, OOS, MC, drawdown, etc.)
      reports/                          Dashboard-display PNGs
```

---

## Pipeline Sequence (notebook cells in order)

### Phase 1 — Regime Detection
`RegimeDetector.detect()` reads live VIX, FRED yield curve (T10Y2Y), FRED HY spread (BAMLH0A0HYM2), and SPY vs its 200-day SMA. Returns BULL / NEUTRAL / BEAR and the corresponding basis weight dict.

### Phase 2 — Universe Construction
`ComprehensiveStockAnalyzer` pulls up to 1,000 tickers from Finviz, excluding Basic Materials and all commodity/energy industries. Tickers are stratified into four market-cap tiers (small/mid/large/mega, 25% each) to prevent mega-cap dominance.

### Phase 3 — Price Download
`yfinance.download()` in batches of 50, 3-year daily Close history. Any ticker with fewer than 252 observations is dropped.

### Phase 4 — Metric Calculation
Two metric paths run in parallel:
- `calculate_consistency_metrics()` — returns, Sortino at 4 horizons, win rate, max drawdown, Calmar, trend score, momentum acceleration. These feed the composite score.
- `_compute_metrics()` — leaner version producing sortino, win_rate, max_dd, momentum (avg 3M+6M), vol_60. These feed the wave function.

### Phase 5 — Composite Scoring
`calculate_composite_score()` ranks stocks by a weighted blend of 5 percentile-ranked components plus high-return and acceleration bonuses. This rank is used to pre-select candidates fed into the quantum layer.

### CAL-A — Phase Calibration
35-combo grid search over phase parameters (a, b). Four walk-forward windows (2-year train, 6-month test). Selects (BEST_A, BEST_B) that maximise mean OOS Sharpe.

### CAL-B — OOS Validation
5-fold walk-forward with phi-dropout noise (sigma=5%, 10 resamples per fold). Rejects the run if < 3 folds show IS-to-OOS Sharpe decay below 10%.

### CAL-C — RL Weight Optimisation
L2-regularised optimisation of basis weights per regime, blended 70/30 with a Q-table RL agent trained on historical regime transitions.

### Phase 6 — Portfolio Construction
`QuantumPortfolioConstructorV4.build()`:
1. Build complex amplitudes using calibrated (a, b) and regime-adaptive weights
2. Filter: sortino > 0, max_dd > -60%
3. Select top 50 by |psi|^2
4. Weights = 40% Born-rule + 30% inverse-vol + 30% risk-parity
5. Hard cap at 5% per position, sector cap at 35%

### Phase 7 — Monte Carlo
`bootstrap_mc()`: 1,000 paths x 252 days using 21-day block resampling. Outputs median return, VaR 95%, CVaR 95%, and drawdown distributions.

### Phase 8 — Cointegration Signals
`QuarterlyRebalancer` tests each holding against its sector ETF (XLK, XLV, XLF, etc.) for cointegration, then computes a rolling Z-score of the spread. Z > +2 = TRIM, Z < -2 = ROTATE_OUT.

### Phase 9 — Cost Estimation
`TransactionCostModel` estimates round-trip cost per trade (0.1 bps commission + Almgren-Chriss impact) and annualises by multiplying by 4 (quarterly rebalancing).

### Phase 10 — Export
Writes `ultimate_portfolio_*.csv`, `calibration_meta.json`, `ibkr_orders.json`, and all chart PNGs.

---

## Feature Flags

Set at the top of the notebook. All default to True except `global`:

| Flag | Effect when True |
|------|-----------------|
| `cointegration` | Run QuarterlyRebalancer Z-score signals |
| `global` | Append EU/JP tickers (disabled by default) |
| `bootstrap` | Run block-bootstrap Monte Carlo |
| `dynamic_weights` | Use regime-adaptive basis weights |
| `risk_parity` | Use 3-way Born/InvVol/RP weight blend |
| `costs` | Run TransactionCostModel and report drag |
| `phase_calibration` | Run CAL-A grid search |
| `oos_validation` | Run CAL-B walk-forward validation |
| `regime_rl` | Run CAL-C L2 + Q-table optimisation |

---

## Dashboard Tabs

| Tab | Content |
|-----|---------|
| Overview | 5 live regime metrics + methodology summary |
| Holdings | Weighted table + sector bar + rebalancing signals |
| Performance | Sharpe/Sortino/Calmar/Return/Vol/WinRate + vs-SPY chart |
| Monte Carlo | Fan chart + return distribution + drawdown distribution |
| Drawdown | Underwater curve + monthly return heatmap |
| Quant Engine | CAL-A heatmap + CAL-B OOS bars + wave amplitudes + IBKR orders |

Launch: `launch_dashboard.bat` or `streamlit run dashboard.py --server.port 8502`

---

## Key Source Modules

**`stock_analysis.py`** — `ComprehensiveStockAnalyzer`, `QuarterlyRebalancer`
**`quantum_portfolio.py`** — `QuantumPortfolioConstructorV4`, `bootstrap_mc`, `risk_parity_weights`
**`regime_detector.py`** — `RegimeDetector` (FRED data, VIX, SPY SMA)
**`transaction_model.py`** — `TransactionCostModel`
**`data_providers.py`** — `FundamentalDataProvider`, `ExpandedDataProvider` (FMP, Finnhub, FRED)

---

## API Keys Required

| Service | Env Variable | Free tier |
|---------|-------------|-----------|
| Financial Modeling Prep | `FMP_API_KEY` | Yes (limited) |
| Finnhub | `FINNHUB_API_KEY` | Yes |
| FRED | `FRED_API_KEY` | Yes (free registration) |

FRED data for regime detection uses the public CSV endpoint (no key required). Keys only needed for fundamentals and sentiment in `data_providers.py`.
