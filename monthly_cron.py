"""
monthly_cron.py — V4 Monthly Pipeline Runner with DataDriftMonitor
==================================================================
Intended usage (scheduled task or cron):
    python monthly_cron.py [--output-dir OUTPUT_DIR] [--dry-run]

Runs a compact version of the V4 pipeline:
  1. Downloads a fresh tick universe + prices
  2. Computes composite scores
  3. Checks for alpha decay via DataDriftMonitor
  4. If drift ≤ WARNING: saves new baseline and exits
  5. If drift = CRITICAL: prints alert and does NOT auto-update baseline
     (manual review required)
  6. Writes ranked CSV to OUTPUT_DIR/monthly_scores_{YYYYMM}.csv
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from datetime import datetime

# ── path resolution so script can be run from any cwd ──────────────────────
THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
SRC_DIR   = os.path.join(THIS_DIR, 'src')
CACHE_DIR = os.path.join(THIS_DIR, 'cache', 'v3')
OUTPUT_DIR_DEFAULT = os.path.join(THIS_DIR, 'output', 'v3')
BASELINE_PATH_DEFAULT = os.path.join(OUTPUT_DIR_DEFAULT, 'scores_baseline.json')

sys.path.insert(0, SRC_DIR)

try:
    from stock_analysis import ComprehensiveStockAnalyzer, DataDriftMonitor
except ImportError as e:
    print(f"[monthly_cron] Import error: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    output_dir: str = OUTPUT_DIR_DEFAULT,
    baseline_path: str = BASELINE_PATH_DEFAULT,
    dry_run: bool = False,
    min_market_cap: float = 1e9,
    max_tickers: int = 500,
) -> int:
    """
    Run the monthly pipeline and monitor for alpha decay.

    Returns
    -------
    int — exit code: 0 = OK/WARNING, 2 = CRITICAL drift detected
    """
    run_ts  = datetime.now()
    run_tag = run_ts.strftime('%Y%m')
    print(f"[monthly_cron] Run: {run_ts.isoformat()}")
    print(f"[monthly_cron] dry_run={dry_run}  output={output_dir}")

    # Step 1 — Fetch universe + prices
    print("\n── Step 1: Fetch universe ──────────────────────────────────────────")
    analyzer = ComprehensiveStockAnalyzer(cache_dir=CACHE_DIR)
    results = analyzer.run_full_analysis(
        min_market_cap=min_market_cap,
        max_tickers=max_tickers,
        cap_allocation='balanced',
    )

    if results is None or results.empty:
        print("[monthly_cron] No results returned — aborting.")
        return 1

    # Step 2 — Normalise scores dataframe
    print("\n── Step 2: Composite scores ────────────────────────────────────────")
    if 'ticker' not in results.columns and results.index.name == 'ticker':
        results = results.reset_index()
    if 'composite_score' not in results.columns:
        print("[monthly_cron] 'composite_score' column missing — check analyzer output.")
        return 1

    print(f"  Tickers scored: {len(results)}")
    print(f"  Score range:    [{results['composite_score'].min():.1f}, {results['composite_score'].max():.1f}]")

    # Step 3 — Drift check
    print("\n── Step 3: Alpha-decay monitor ─────────────────────────────────────")
    monitor = DataDriftMonitor()
    status  = monitor.check_drift(baseline_path, results)

    # Step 4 — Save outputs (unless dry-run or CRITICAL)
    print(f"\n── Step 4: Save outputs  (status={status}) ─────────────────────────")
    os.makedirs(output_dir, exist_ok=True)

    scores_out = os.path.join(output_dir, f'monthly_scores_{run_tag}.csv')
    if not dry_run:
        results.sort_values('composite_score', ascending=False).to_csv(scores_out, index=False)
        print(f"  Scores saved:   {scores_out}")
    else:
        print(f"  [dry-run] Would save scores to: {scores_out}")

    if status in ('DRIFT_OK', 'DRIFT_WARNING'):
        if not dry_run:
            monitor.save_baseline(results, baseline_path)
            print(f"  Baseline updated: {baseline_path}")
        else:
            print(f"  [dry-run] Would update baseline: {baseline_path}")
    else:
        # CRITICAL — print alert; do NOT auto-update baseline
        print("\n" + "=" * 60)
        print("  ⚠  CRITICAL DRIFT DETECTED")
        print("  Composite score distribution has shifted significantly.")
        print("  Review changes before accepting new baseline:")
        print(f"    baseline : {baseline_path}")
        print(f"    new run  : {scores_out}")
        print("  To accept manually: DataDriftMonitor().save_baseline(df, path)")
        print("=" * 60)

    # Step 5 — Summary
    top5 = results.nlargest(5, 'composite_score')[['ticker', 'composite_score']].to_string(index=False)
    print(f"\n── Top 5 this run:\n{top5}")
    print(f"\n[monthly_cron] Finished.  Status={status}")

    return 2 if status == 'DRIFT_CRITICAL' else 0


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V4 Monthly Pipeline Runner')
    parser.add_argument('--output-dir',    default=OUTPUT_DIR_DEFAULT,   help='Output directory')
    parser.add_argument('--baseline-path', default=BASELINE_PATH_DEFAULT, help='Drift baseline JSON path')
    parser.add_argument('--min-market-cap', type=float, default=1e9,     help='Min market cap filter')
    parser.add_argument('--max-tickers',    type=int,   default=500,      help='Max universe size')
    parser.add_argument('--dry-run',        action='store_true',          help='Do not write any files')
    args = parser.parse_args()

    exit_code = run_pipeline(
        output_dir=args.output_dir,
        baseline_path=args.baseline_path,
        dry_run=args.dry_run,
        min_market_cap=args.min_market_cap,
        max_tickers=args.max_tickers,
    )
    sys.exit(exit_code)
