# V4 Ultimate Quantum Portfolio — Complete Methodology

> This document is the single authoritative reference for every analytical decision made in the V4 Ultimate strategy. It covers the full pipeline from raw data ingestion to IBKR order generation, including every formula, threshold, and calibration procedure baked into the code.

---

## Table of Contents

1. [Universe Construction](#1-universe-construction)
2. [Price Data Pipeline](#2-price-data-pipeline)
3. [Metric Calculation](#3-metric-calculation)
4. [Composite Scoring](#4-composite-scoring)
5. [Macro Regime Detection](#5-macro-regime-detection)
6. [Quantum Wave Function Construction](#6-quantum-wave-function-construction)
7. [CAL-A: Empirical Phase Calibration](#cal-a-empirical-phase-calibration)
8. [CAL-B: Walk-Forward OOS Validation](#cal-b-walk-forward-oos-validation)
9. [CAL-C: Regime-Adaptive Basis Weights and RL Blending](#cal-c-regime-adaptive-basis-weights-and-rl-blending)
10. [Portfolio Construction and Position Sizing](#10-portfolio-construction-and-position-sizing)
11. [Cointegration Rebalancing Filter](#11-cointegration-rebalancing-filter)
12. [Bootstrap Monte Carlo Simulation](#12-bootstrap-monte-carlo-simulation)
13. [Transaction Cost Model](#13-transaction-cost-model)
14. [Export and IBKR Integration](#14-export-and-ibkr-integration)
15. [Default Parameters Reference](#15-default-parameters-reference)

---

## 1. Universe Construction

**Source:** Finviz screener via the `finviz` Python library.

**Pre-filters applied at fetch time:**

| Filter | Value |
|--------|-------|
| Average daily volume | > 100,000 shares |
| Share price | > $5.00 |
| Minimum market cap | configurable (default $300M) |

**Sector and industry exclusions** — hardcoded in `ComprehensiveStockAnalyzer`:

```
Excluded sectors : Basic Materials
Excluded industries: Gold, Silver, Copper, Aluminum, Steel, Coking Coal,
  Thermal Coal, Uranium, Oil & Gas E&P, Oil & Gas Midstream,
  Oil & Gas Refining & Marketing, Oil & Gas Drilling,
  Oil & Gas Equipment & Services, Other Precious Metals & Mining,
  Other Industrial Metals & Mining, Agricultural Inputs,
  Lumber & Wood Production
```

Rationale: Commodity-linked equities introduce macro-factor exposure (futures curves, physical supply constraints) that is orthogonal to the pure equity quality signals the strategy targets. Their risk-return profiles violate the stationarity assumptions underlying the cointegration rebalancing filter.

**Tiered market-cap selection (`get_tiered_tickers`):**

The universe is stratified into four capitalisation buckets and sampled proportionally to avoid mega-cap concentration bias:

| Tier | Cap range | Default allocation (balanced) |
|------|-----------|-------------------------------|
| Small | $300M – $2B | 25% of max_tickers |
| Mid | $2B – $10B | 25% |
| Large | $10B – $50B | 25% |
| Mega | > $50B | 25% |

Within each bucket, stocks are ranked by average daily volume and the top N are selected. Alternative allocation schemes (`growth`, `value`) shift the percentages toward small/mid or large/mega respectively.

**Caching:** Finviz metadata is written to `cache/v3/finviz_data.csv` with a 24-hour TTL. Subsequent same-day runs are instant.

---

## 2. Price Data Pipeline

**Source:** Yahoo Finance via `yfinance.download()`, batched in groups of 50 tickers.

**History window:** 3 years (period='3y'), daily Close prices.

**Minimum usable history:** 252 trading days (1 year). Any ticker with fewer valid Close observations is dropped silently.

**Rate limiting:** 0.1-second sleep between batches to stay within yfinance's informal rate limit.

**Multi-level column handling:** When downloading a single ticker, yfinance returns a flat DataFrame; for batches it returns a MultiIndex. Both cases are handled explicitly before extracting the Close series.

**Optional global extension (feature flag `global`):**

When enabled, `get_global_tickers()` appends European (`.DE`, `.PA`, `.L`) and Japanese (`.T`) tickers fetched from a curated list. Prices are FX-converted to USD using daily rates downloaded from yfinance (`EURUSD=X`, `GBPUSD=X`, `JPYUSD=X`). The converted series are merged into `analyzer.price_data` before the calibration phases run, so the full universe (US + international) goes through identical processing.

---

## 3. Metric Calculation

All metrics are computed per-ticker from their Close price series. Two separate metric calculation paths exist in the codebase:

### 3A. ComprehensiveStockAnalyzer metrics (composite scoring)

Calculated in `calculate_consistency_metrics()`:

**Return metrics:**

| Metric | Formula |
|--------|---------|
| Return 1M | `(P[-1] / P[-21] - 1) * 100` |
| Return 3M | `(P[-1] / P[-63] - 1) * 100` |
| Return 6M | `(P[-1] / P[-126] - 1) * 100` |
| Return 1Y | `(P[-1] / P[-252] - 1) * 100` |

**Risk metrics:**

**Sortino Ratio** computed at four time horizons (1Y, 6M, 3M, 1M):

```
sortino(window) = (R_ann - RF) / downside_vol_ann
```

Where:
- `R_ann = mean(daily_returns[window]) * 252`
- `downside_vol_ann = std(daily_returns[window] where return < 0) * sqrt(252)`
- `RF = 0.04` (4% annual risk-free rate, hardcoded)

**Win Rate:**

```
win_rate = count(monthly_returns > 0) / total_months * 100
```

Monthly returns computed from month-end Close prices (`resample('ME').last()`).

**Max Drawdown:**

```
max_dd = min((P - cummax(P)) / cummax(P)) * 100
```

Uses expanding maximum (true peak-to-trough over the full history), not a rolling window.

**Calmar Ratio:**

```
calmar = (annualised_return * 100) / abs(max_dd)
```

Annualised return uses CAGR: `(P[-1]/P[0])^(252/n) - 1`.

**Trend Score** (0–100 composite):

```
base    = 50
+20 if current_price > MA_50
+15 if current_price > MA_200
+15 if MA_50 > MA_200 (golden cross)
```

**Momentum Acceleration flag:**

Binary signal: 1 if `sortino_1M > sortino_3M > sortino_6M`, i.e., Sortino is improving at all horizons simultaneously.

### 3B. Quantum portfolio metrics (wave function inputs)

A second, leaner metric calculation runs in `_compute_metrics()` inside `quantum_portfolio.py`. This produces the inputs fed directly into the wave function:

| Column | Calculation |
|--------|-------------|
| `sortino` | 1-year annualised Sortino (same formula as 3A) |
| `win_rate` | Fraction of positive months (not percentage) |
| `max_dd` | Raw decimal (e.g., -0.25 means -25%) |
| `momentum` | `((m3 + m6) / 2)` where `m3 = P[-1]/P[-63]-1`, `m6 = P[-1]/P[-126]-1` |
| `vol_60` | `std(daily_returns[-60:]) * sqrt(252)` (60-day annualised vol) |

Minimum requirement: 252 daily observations.

---

## 4. Composite Scoring

**Location:** `calculate_composite_score()` in `stock_analysis.py`.

**Pre-filter before scoring:**

```
max_drawdown_pct > -65%   (removes extreme disaster stocks)
win_rate_pct > 35%        (removes persistently negative return profiles)
```

**Individual rank components** (all computed as percentile ranks, 0–100):

| Component | Source column | Weight |
|-----------|--------------|--------|
| Return rank | 60% * return_6m + 40% * return_1y | 35% |
| Momentum Sortino rank | sortino_3m | 25% |
| Trend rank | trend_score | 20% |
| Sortino rank | sortino_ratio (1Y) | 15% |
| Calmar rank | calmar_ratio | 5% |

**Bonuses (added directly to percentile score):**

| Condition | Bonus points |
|-----------|-------------|
| Momentum accelerating (sortino_1M > sortino_3M > sortino_6M) | +15 |
| 6M return > 150% | +30 |
| 6M return 100-150% | +22 |
| 6M return 80-100% | +16 |
| 6M return 50-80% | +10 |
| 6M return 30-50% | +5 |

**Final composite score:**

```
composite = (
    raw_return_rank    * 0.35 +
    momentum_sortino_rank * 0.25 +
    trend_rank         * 0.20 +
    sortino_rank       * 0.15 +
    calmar_rank        * 0.05 +
    acceleration_bonus +
    high_return_bonus
)
```

This score is used to rank the universe and feed the top candidates into the quantum selection layer.

---

## 5. Macro Regime Detection

**Location:** `src/regime_detector.py`, `RegimeDetector` class.

The four indicators and their bear-signal thresholds:

| Indicator | Source | Bear signal threshold |
|-----------|--------|----------------------|
| VIX | yfinance `^VIX`, last close | > 25 |
| 10Y–2Y Yield Curve | FRED series `T10Y2Y` | < 0 (inverted) |
| HY Credit Spread | FRED series `BAMLH0A0HYM2` | > 5.0% |
| SPY vs 200-SMA | yfinance `SPY`, 200-day rolling mean | SPY < SMA-200 |

FRED data is fetched via the public CSV endpoint (`https://fred.stlouisfed.org/graph/fredgraph.csv?id=`) with a 1-hour file cache. The most recent non-missing observation is used.

**Regime scoring function:**

Each indicator contributes a weighted negative score when its bear signal fires:

```
score = -0.25 * vix_bear
      - 0.35 * curve_inverted
      - 0.25 * spread_wide
      - 0.15 * spy_below_sma200
```

The yield curve and credit spread receive higher weights because they are more persistent leading indicators of economic stress. VIX is reactive/volatile and receives a lighter weight despite being the most visible indicator.

**Classification thresholds:**

```
score > -0.25  =>  BULL
score < -0.50  =>  BEAR
otherwise      =>  NEUTRAL
```

**Regime-adaptive basis weights** — the regime classification directly changes how much each quantum basis state contributes to the wave function:

| Regime | momentum | safety | sortino | value |
|--------|----------|--------|---------|-------|
| BULL | 0.40 | 0.25 | 0.20 | 0.15 |
| NEUTRAL | 0.30 | 0.30 | 0.25 | 0.15 |
| BEAR | 0.10 | 0.45 | 0.20 | 0.25 |

In a Bear regime, safety (win rate + drawdown protection) receives the dominant weight and momentum drops to near-zero to avoid buying into downtrends.

---

## 6. Quantum Wave Function Construction

**Location:** `QuantumPortfolioConstructor._build_wave_function()` in `quantum_portfolio.py`.

The wave function is a metaphor for a multi-dimensional scoring function that creates interference effects between basis components. Each stock receives a complex amplitude built from four orthogonal basis states.

### 6.1 Basis States

| Basis | Metric source |
|-------|--------------|
| `momentum` | Normalised average of 3M and 6M price return |
| `safety` | Average of normalised win_rate and normalised (negated) max_dd |
| `sortino` | Normalised 1-year Sortino ratio |
| `value` | Normalised inverse of 60-day annualised volatility |

All normalisation uses min-max scaling to [0, 1] across the universe. When the max and min are degenerate (< 1e-10 apart), 0.5 is used.

### 6.2 Phase Angle

For each basis state k, the phase angle is:

```
theta_k(phi_k) = a * pi * (1 - phi_k)^b
```

Where:
- `phi_k` is the normalised score on basis k (0 = worst, 1 = best)
- `a` and `b` are calibrated parameters (see CAL-A)
- Default before calibration: a = 1.0, b = 1.0

When `phi_k = 1` (best possible score): `theta = 0` → `cos(0) = 1` → **fully constructive**
When `phi_k = 0` (worst possible score): `theta = a*pi` → `cos(a*pi) = -1` when a=1 → **fully destructive**

This is the key mechanism: strong stocks interfere constructively, weak stocks interfere destructively.

### 6.3 Amplitude Accumulation

The total complex amplitude for stock i sums contributions from all basis states:

```
psi_real_i = sum_k [sqrt(w_k) * phi_ki * cos(theta_ki)]
psi_imag_i = sum_k [sqrt(w_k) * phi_ki * sin(theta_ki)]
```

Where `w_k` is the basis weight for state k (regime-adaptive, from Section 5).

### 6.4 Born Rule

The selection probability for stock i:

```
|psi_i|^2 = psi_real_i^2 + psi_imag_i^2
```

Born-rule normalised probability:

```
P_i = |psi_i|^2 / sum_j |psi_j|^2
```

This is a proper probability distribution: all values are non-negative and sum to 1.

---

## CAL-A: Empirical Phase Calibration

**Location:** Notebook cell `CAL-A` in `StockPickerV4_Ultimate.ipynb`.

**Problem:** The original `a=1, b=1` phase function was a heuristic. CAL-A finds the (a, b) pair that maximises out-of-sample Sharpe ratio over historical data.

**Grid definition:**

```
A_GRID = linspace(0.5, 2.0, 7)  = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
B_GRID = linspace(0.8, 1.5, 5)  = [0.80, 0.975, 1.15, 1.325, 1.50]
```

35 total (a, b) combinations.

**Walk-forward windows** (4 windows, non-overlapping test periods):

| Window | Train | Test |
|--------|-------|------|
| 1 | 2021-01 — 2022-12 | 2023-01 — 2023-06 |
| 2 | 2022-01 — 2023-12 | 2024-01 — 2024-06 |
| 3 | 2022-07 — 2024-06 | 2024-07 — 2024-12 |
| 4 | 2023-01 — 2024-12 | 2025-01 — 2025-06 |

**Evaluation procedure for each (a, b) combination:**

1. Compute `|psi|^2` for all stocks using `_psi_sq_ab(metrics, basis_weights, a, b)`
2. Select top 50 by `|psi|^2`
3. Build weights: 50% Born-rule probability + 50% inverse-vol, capped at `max_position_pct`
4. Compute annualised Sharpe on the **test window** using actual daily returns
5. Average OOS Sharpe across all valid windows

**Selection:** The (a, b) pair with the highest mean OOS Sharpe is chosen as `BEST_A`, `BEST_B`. These are stored in `calibration_meta.json` and used for the remainder of the run.

**Baseline comparison:** The grid point closest to (a=1, b=1) serves as the fixed-phase baseline. The dashboard reports the uplift from calibration relative to this baseline.

**Output:** `output/v3/v4_phase_calibration_heatmap.png` — OOS Sharpe heatmap with best cell highlighted in cyan.

---

## CAL-B: Walk-Forward OOS Validation

**Location:** Notebook cell `CAL-B` in `StockPickerV4_Ultimate.ipynb`.

**Purpose:** Validate that the calibrated parameters generalise — i.e., the model is not overfit to the training periods used in CAL-A. Also tests robustness to noise in the input signals via phi-dropout.

**Parameters:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| k | 5 | Number of cross-validation folds |
| DECAY_THRESHOLD | 10% | Maximum tolerated IS→OOS Sharpe degradation |
| NOISE_SIGMA | 5% | Relative perturbation applied to each phi in dropout |
| DROPOUT_N | 10 | Number of noisy resamples per fold |

**Walk-forward windows** (5 folds, earlier and longer than CAL-A windows):

| Fold | Train | Test |
|------|-------|------|
| 1 | 2020-01 — 2021-12 | 2022-01 — 2022-06 |
| 2 | 2020-07 — 2022-06 | 2022-07 — 2022-12 |
| 3 | 2021-01 — 2022-12 | 2023-01 — 2023-06 |
| 4 | 2021-07 — 2023-06 | 2023-07 — 2023-12 |
| 5 | 2022-01 — 2023-12 | 2024-01 — 2024-06 |

**Per-fold calculation:**

1. `IS_Sharpe` = annualised Sharpe built on the training window
2. `OOS_Sharpe` = annualised Sharpe on the test window (no noise)
3. `Drop_OOS` = mean OOS Sharpe over DROPOUT_N resamples with phi-dropout noise
4. `Decay_%` = `(IS_Sharpe - OOS_Sharpe) / |IS_Sharpe| * 100`
5. `Pass` = decay <= 10%

**Phi-dropout noise:** For each of the 10 dropout resamples, every metric column (`momentum`, `sortino`, `win_rate`, `max_dd`, `vol_60`) is perturbed:

```
phi_noisy = phi * (1 + Normal(0, 0.05))
```

This simulates estimation noise in the input signals and tests whether the portfolio selection is stable under slight perturbations.

**Verdict:**
- ACCEPTED if >= 3 of 5 folds pass (majority criterion)
- CAUTION if < 3 folds pass — suggests overfitting or structural change

**Output:** `output/v3/v4_oos_validation.png` — side-by-side bar charts of IS vs OOS vs Dropout Sharpe per fold, plus a decay line chart.

---

## CAL-C: Regime-Adaptive Basis Weights and RL Blending

**Location:** Notebook cell `CAL-C` in `StockPickerV4_Ultimate.ipynb`.

### CAL-C Part 1: L2-Regularised Weight Optimisation

For each regime (BULL, NEUTRAL, BEAR), an L2-penalised (ridge regression) optimisation finds the basis weight vector that maximises historical Sharpe on the relevant market periods.

The prior weights (from `REGIME_WEIGHTS` in `regime_detector.py`) serve as the regularisation target. With strong L2 penalties, the optimised weights shrink back toward these informed priors, preventing overfitting to short historical windows.

### CAL-C Part 2: Q-Table Reinforcement Learning Blend

A simple tabular Q-learning agent is trained on historical regime transitions:

**State space:** The four macro indicators discretised into binary signals (vix_high, curve_inverted, spread_wide, spy_below_sma) gives 16 possible states.

**Action space:** 3 regime choices (BULL, NEUTRAL, BEAR) mapped to their weight vectors.

**Reward:** Annualised Sharpe of the resulting portfolio over the following 21-day period.

**Q-update:**

```
Q(s, a) <- Q(s, a) + lr * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

With `lr=0.1`, `gamma=0.9`.

**Final blend:**

```
final_weights = (1 - rl_blend) * L2_weights + rl_blend * RL_weights
```

Default `rl_blend = 0.30` (30% RL, 70% L2-optimised). The RL agent provides trend-following in regime transitions while the L2-optimised weights provide stability.

**Output files:** `v4_regime_weights_optimised.png`, `v4_rl_qtable.png`

---

## 10. Portfolio Construction and Position Sizing

**Location:** `QuantumPortfolioConstructorV4._collapse()` in `quantum_portfolio.py`.

**Pre-filter before selection:**

```
sortino > 0         (positive risk-adjusted return)
max_dd > -0.60      (maximum 60% historical drawdown)
```

**Selection:** Top `n_holdings` (default 50) stocks by `|psi|^2`.

**Three-way weight blend (when `risk_parity=True`):**

```
raw_weight = born_weight   * 0.40
           + inv_vol_weight * 0.30
           + rp_weight      * 0.30
```

Where:
- `born_weight` = `P_i = |psi_i|^2 / sum |psi_j|^2` (Born-rule probability)
- `inv_vol_weight` = `1/vol_60_i / sum(1/vol_60_j)` (inverse 60-day vol)
- `rp_weight` = risk-parity (inverse variance), computed from `risk_parity_weights()` using the full historical covariance of the selected tickers

**Position cap:** Each weight is hard-capped at `max_position_pct / 100` (default 5%). After capping, weights are renormalised to sum to 1.

**Sector diversification constraint:** An additional sector-level cap of 35% prevents any single sector from dominating. If a sector exceeds this, the excess is redistributed proportionally to other sectors.

---

## 11. Cointegration Rebalancing Filter

**Location:** `QuarterlyRebalancer` in `stock_analysis.py`.

**Purpose:** Identify positions that have diverged abnormally from their sector benchmark, flagging them for trimming or rotation without requiring a full portfolio reconstruction.

**Sector ETF mapping:**

| Sector | ETF |
|--------|-----|
| Technology | XLK |
| Healthcare | XLV |
| Financial Services | XLF |
| Consumer Cyclical | XLY |
| Consumer Defensive | XLP |
| Industrials | XLI |
| Energy | XLE |
| Utilities | XLU |
| Real Estate | XLRE |
| Communication Services | XLC |
| Other / Unknown | SPY |

**Cointegration test:** Engle-Granger two-step test (`statsmodels.tsa.stattools.coint`) on log-price series over 2 years. A pair is flagged as cointegrated if `p-value < 0.05`.

**Spread Z-score calculation:**

1. Compute log prices for both the stock and its sector ETF
2. Estimate the hedge ratio `beta = Cov(stock, ETF) / Var(ETF)` on the most recent `lookback_days` (default 90)
3. Spread: `spread = log(stock) - beta * log(ETF)`
4. Rolling Z-score:

```
Z = (spread - rolling_mean(spread, lookback)) / rolling_std(spread, lookback)
```

**Signal generation:**

```
Z > +2.0  =>  TRIM       (overperforming sector by > 2 sigma)
Z < -2.0  =>  ROTATE_OUT (underperforming sector by > 2 sigma)
|Z| < 2.0 =>  HOLD
```

Signals are surfaced in the dashboard Holdings tab with explanations.

---

## 12. Bootstrap Monte Carlo Simulation

**Location:** `bootstrap_mc()` in `quantum_portfolio.py`.

**Method:** Block bootstrap — resample 21-day (one calendar month) return blocks from actual historical data, rather than drawing from a fitted Gaussian. This preserves:
- Serial autocorrelation (momentum effects within a month)
- Fat-tail return distributions (actual crash events recur in simulations)
- Cross-asset correlation structure (blocks are resampled jointly)

**Parameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| n_sims | 1000 | Number of independent simulation paths |
| horizon | 252 | Trading days per path (1 year) |
| block_size | 21 | Days per resampled block (~1 month) |
| initial | $100,000 | Starting portfolio value |
| psi_modulate | True | Apply |psi|^2 noise scaling (see below) |

**|psi|^2 modulation:**

High-confidence stocks (high `|psi|^2`) are simulated with less random block-order reshuffling. Concretely, each stock's block-start index pool is weighted by its `|psi|^2` value — the highest-|psi|^2 stocks pull from a more concentrated sample of their own best historical blocks. Low-|psi|^2 stocks draw uniformly, introducing more variance.

**Return metrics computed from the 1000 paths:**

| Metric | Calculation |
|--------|-------------|
| Median 1Y return | Median of terminal portfolio values |
| VaR 95% | 5th percentile of 1-year returns |
| CVaR 95% | Mean of returns below VaR 95% |
| Median max drawdown | Median of per-path peak-to-trough declines |

**Benchmark comparison:** The same simulation runs in parallel for a classical equal-weight variant (same tickers, equal weights) and SPY, enabling direct risk-adjusted comparison.

**Output charts:** `v4_mc_fan_chart.png`, `v4_mc_return_dist.png`, `v4_mc_drawdown_dist.png`

---

## 13. Transaction Cost Model

**Location:** `src/transaction_model.py`, `TransactionCostModel` class.

**Two-component model:**

```
total_cost = commission + market_impact
```

**Commission:**

```
commission = trade_value * (commission_bps / 10,000)
```

Default `commission_bps = 0.1` bps (institutional flat rate, not retail).

**Market impact** (Almgren-Chriss inspired):

```
impact_pct = max(impact_floor_pct / 100, vol_60 / impact_vol_div)
market_impact = trade_value * impact_pct
```

Default values: `impact_floor_pct = 0.10%`, `impact_vol_div = 10.0`

For a stock with 25% annualised volatility: `impact = max(0.10%, 25%/10) = 2.5%` per trade — meaning a $100,000 trade in this stock costs $2,500 in expected market impact on top of commissions.

**Turnover calculation:**

The model takes old and new weight vectors. For each ticker, `delta_weight = |new_w - old_w|`. The turnover (one-way) is the sum of all absolute weight changes.

**Annualised drag:**

```
annual_drag = single_rebalance_cost_pct * 4  (quarterly rebalancing)
```

**Usage context:** Costs are estimated for `AUM = $10,000,000` by default. The drag estimate feeds into the calibration metadata to give a realistic net-of-cost Sharpe expectation.

---

## 14. Export and IBKR Integration

**Calibration metadata (`calibration_meta.json`):**

All calibration outputs are bundled into a single JSON file written to `output/v3/`:

```json
{
  "run_timestamp": "...",
  "best_a": 1.25,
  "best_b": 1.15,
  "oos_valid": true,
  "n_folds_pass": 4,
  "avg_decay_pct": 6.2,
  "regime": "BULL",
  "rl_recommended": "BULL",
  "rl_blend": 0.30,
  "final_basis_weights": { "momentum": ..., "safety": ..., ... },
  "drawdown": { "quantum_median_mdd": ..., ... },
  "drawdown_hist": { "current_dd": ..., "max_dd_1y": ..., ... },
  "monte_carlo": { "quantum_median_return": ..., "var_95": ..., ... },
  "aum": 10000000
}
```

**IBKR order JSON (`ibkr_orders.json`):**

Each holding is converted to a share-count order:

```json
[
  { "ticker": "AAPL", "action": "BUY", "shares": 42, "target_weight": 0.032, ... },
  ...
]
```

Shares = `round(weight * AUM / last_price)`. The JSON is structured for direct consumption by the `ib_insync` TWS API wrapper, though the actual order submission is a manual step to preserve human oversight.

**Portfolio CSV (`ultimate_portfolio_MMDDYYYY.csv`):**

Full portfolio with all metrics:
`ticker, sector, weight, psi_sq, prob, sortino, win_rate, max_dd, momentum, vol_60, last_price, zscore, regime`

---

## 15. Default Parameters Reference

All configurable parameters and their defaults as set in the V4 notebook:

| Parameter | Default | Location |
|-----------|---------|----------|
| `N_HOLDINGS` | 50 | Notebook cell (Phase flags) |
| `MAX_POSITION_PCT` | 5.0% | Notebook |
| `MIN_MARKET_CAP` | $300M | Notebook |
| `MAX_TICKERS` | 1000 | Notebook |
| `AUM` | $10,000,000 | Notebook |
| `N_SIMS` | 1000 | Notebook |
| `HORIZON_DAYS` | 252 | Notebook |
| Risk-free rate | 4% annual | `stock_analysis.py`, `quantum_portfolio.py` |
| Sortino filter | > 0 | `_collapse()` |
| Drawdown filter | > -60% | `_collapse()` |
| Z-score threshold | 2.0 | `QuarterlyRebalancer` |
| Z-score lookback | 90 days | `QuarterlyRebalancer` |
| Regime cache TTL | 1 hour | `RegimeDetector` |
| Finviz cache TTL | 24 hours | `ComprehensiveStockAnalyzer` |
| FRED cache TTL | 1 hour | `RegimeDetector` |
| Phase default a | 1.0 | `quantum_portfolio.py` |
| Phase default b | 1.0 | `quantum_portfolio.py` |
| Phase grid a | [0.5 .. 2.0], 7 pts | CAL-A notebook cell |
| Phase grid b | [0.8 .. 1.5], 5 pts | CAL-A notebook cell |
| OOS decay threshold | 10% | CAL-B, `DECAY_THRESHOLD` |
| Dropout sigma | 5% | CAL-B, `NOISE_SIGMA` |
| Dropout resamples | 10 | CAL-B, `DROPOUT_N` |
| RL blend | 0.30 | CAL-C |
| Commission | 0.1 bps | `TransactionCostModel` |
| Impact floor | 0.10% | `TransactionCostModel` |
| Block bootstrap size | 21 days | `bootstrap_mc()` |
| Born/InvVol/RP blend | 40/30/30% | `QuantumPortfolioConstructorV4` |
