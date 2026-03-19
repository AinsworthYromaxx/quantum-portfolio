"""
patch_quantum3d.py — appends the Quantum Evidence section to PortfolioReport_V4.ipynb
Run once: python patch_quantum3d.py
"""
import json, os

nb_path = os.path.join(os.path.dirname(__file__), "PortfolioReport_V4.ipynb")

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Guard: don't add twice
for c in nb["cells"]:
    if "Quantum Evidence" in "".join(c["source"]):
        print("Section already present — nothing to do.")
        raise SystemExit(0)

# ── Markdown header ────────────────────────────────────────────────────────────
md_src = (
    "## 10. Quantum Evidence \u2014 3D Wave-Function Visualizations\n"
    "\n"
    "Three interactive Plotly figures expose the quantum portfolio mechanics to any audience:\n"
    "\n"
    "| Figure | Axes | Story |\n"
    "|--------|------|-------|\n"
    "| **Wave Interference Surface** | \u03c6_momentum \u00d7 \u03c6_safety \u2192 |\u03c8|\u00b2 |"
    " Constructive interference concentrates probability at high-quality stocks |\n"
    "| **Phase Space Trajectories** | \u03c6_mom, \u03c6_sort, \u03c6_safety |"
    " Holdings converge from disorder \u2192 ground state |\n"
    "| **Energy Landscape** | Vol \u00d7 Return \u2192 \u2212|\u03c8|\u00b2 |"
    " Portfolio sits at the energy minimum |\n"
    "\n"
    "Use the **Phase slider** in Figure\u00a01 to animate the interference pattern"
    " across different *a* values."
)

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [md_src],
}

# ── Code cell ──────────────────────────────────────────────────────────────────
code_src = """\
# \u2500\u2500 Quantum 3D Visualizations \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
import sys, os
sys.path.insert(0, os.path.join('..', 'src'))
from v4_quantum_3d_plots import generate_3d_visuals, save_visuals

# \u2500\u2500 Assemble metrics_df from notebook state \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Prefer _cal_metrics (full universe from CAL-A) if available.
try:
    _q3d_metrics = _cal_metrics.copy()
except NameError:
    _q3d_metrics = portfolio.copy() if 'portfolio' in dir() else None

try:
    _q3d_portfolio = portfolio.copy()
except NameError:
    _q3d_portfolio = None

_q3d_a  = BEST_A        if 'BEST_A'        in dir() else 1.0
_q3d_b  = BEST_B        if 'BEST_B'        in dir() else 1.0
_q3d_bw = basis_weights if 'basis_weights' in dir() else None

assert _q3d_metrics   is not None, "No metrics \u2014 run through Phase 3 first"
assert _q3d_portfolio is not None, "No portfolio \u2014 run through Phase 8 first"

n_universe = len(_q3d_metrics)
n_holdings = len(_q3d_portfolio)
print(f"Generating 3D visuals: universe={n_universe} stocks, "
      f"portfolio={n_holdings} holdings, a={_q3d_a:.3f}, b={_q3d_b:.3f}")

fig1, fig2, fig3 = generate_3d_visuals(
    _q3d_metrics, _q3d_portfolio,
    best_a=_q3d_a, best_b=_q3d_b,
    basis_weights=_q3d_bw,
)

fig1.show()
fig2.show()
fig3.show()

# \u2500\u2500 Save HTML + PNG \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
_q3d_dir = os.path.join('..', 'output', 'v3')
save_visuals(fig1, fig2, fig3, _q3d_dir, also_png=True)

# \u2500\u2500 Save universe snapshot for dashboard live rendering \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
_snap_cols = [c for c in [
    'ticker', 'sector', 'industry', 'momentum', 'sortino',
    'win_rate', 'max_dd', 'vol_60', 'composite_score',
] if c in _q3d_metrics.columns]
_snapshot = _q3d_metrics[_snap_cols].copy()
_snap_path = os.path.join(_q3d_dir, 'v4_3d_snapshot.json')
_snapshot.to_json(_snap_path, orient='records', indent=2)
print(f"Snapshot saved: {_snap_path}")

# \u2500\u2500 Quantum concentration statistic \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
from v4_quantum_3d_plots import _extract_phi, _psi_sq_stocks, _DEFAULT_BW
_bw    = _q3d_bw or _DEFAULT_BW
_phi_u = _extract_phi(
    _q3d_metrics.set_index('ticker')
    if 'ticker' in _q3d_metrics.columns
    else _q3d_metrics
)
_psi_u   = _psi_sq_stocks(_phi_u, _bw, _q3d_a, _q3d_b)
_top_psi = sorted(_psi_u, reverse=True)[:n_holdings]
_conc    = sum(_top_psi) / (sum(_psi_u) + 1e-9) * 100
print(f"\\n\u2728 Quantum concentration: top-{n_holdings} holdings capture {_conc:.1f}% of total |\u03c8|\u00b2")
"""

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [code_src],
}

nb["cells"].append(md_cell)
nb["cells"].append(code_cell)

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done — notebook now has {len(nb['cells'])} cells.")
