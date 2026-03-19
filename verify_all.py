import sys, os, json

sys.path.insert(0, r'c:/Users/maxim/OneDrive/Bureau/Python/v4_ultimate/src')
import stock_analysis as sa
import transaction_model as tm

results = []

def check(label, cond, detail=''):
    status = 'OK' if cond else 'FAIL'
    results.append(f'[{status}] {label}' + (f'  ({detail})' if detail else ''))

# Task 1
check('T1 DataDriftMonitor class', hasattr(sa, 'DataDriftMonitor'))
m = sa.DataDriftMonitor()
check('T1 save_baseline method', hasattr(m, 'save_baseline'))
check('T1 check_drift method',   hasattr(m, 'check_drift'))
check('T1 monthly_cron.py exists',
      os.path.exists(r'c:/Users/maxim/OneDrive/Bureau/Python/v4_ultimate/monthly_cron.py'))

# Task 2
src = open(r'c:/Users/maxim/OneDrive/Bureau/Python/v4_ultimate/src/stock_analysis.py', encoding='utf-8').read()
check('T2 CLEAN_ENERGY_INDUSTRIES', 'CLEAN_ENERGY_INDUSTRIES' in src)
check('T2 utilities_bonus',         'utilities_bonus' in src)
check('T2 clean_energy_bonus',      'clean_energy_bonus' in src)
check('T2 roe_filter function',     'def roe_filter(' in src)

# Task 3
check('T3 fx_adjusted_returns',  hasattr(sa, 'fx_adjusted_returns'))

# Task 4
nb = json.load(open(r'c:/Users/maxim/OneDrive/Bureau/Python/v4_ultimate/notebooks/StockPickerV4_Ultimate.ipynb', encoding='utf-8'))
src7 = ''.join(nb['cells'][7]['source'])
check('T4 _Q80 dict (72-state)',    '_Q80: dict'    in src7)
check('T4 _encode_state function',  '_encode_state' in src7)
check('T4 _live_state encoding',    '_live_state'   in src7)
check('T4 VIX slice heatmap',       'VIX<15'        in src7)

# Task 5
t = tm.TransactionCostModel(aum=10_000_000)
check('T5 capacity_curve method', hasattr(t, 'capacity_curve'))
tm_src = open(r'c:/Users/maxim/OneDrive/Bureau/Python/v4_ultimate/src/transaction_model.py', encoding='utf-8').read()
check('T5 sqrt impact model',     'math.sqrt(notional' in tm_src)
check('T5 aum_levels param',      'aum_levels' in tm_src)

# Task 6
src5 = ''.join(nb['cells'][5]['source'])
check('T6 COVID window 2020-02-01', '2020-02-01' in src5)
check('T6 bear window 2022-06-01',  '2022-06-01' in src5)
check('T6 title updated',           'incl. crash periods' in src5)
n_wf = src5.count("('20")
check('T6 6 walk-forward windows',  n_wf >= 6, f'found {n_wf}')

# Task 7
qr = sa.QuarterlyRebalancer
check('T7 INDUSTRY_ETF_MAP exists', hasattr(qr, 'INDUSTRY_ETF_MAP'))
n_ind = len(qr.INDUSTRY_ETF_MAP)
check('T7 50+ industry entries',    n_ind >= 50, f'{n_ind} entries')
inst = qr.__new__(qr)  # skip __init__ holdings requirement
etf_a = qr.INDUSTRY_ETF_MAP.get('Semiconductors')
etf_b = qr.INDUSTRY_ETF_MAP.get('Biotechnology')
etf_c = qr.INDUSTRY_ETF_MAP.get('Solar')
check('T7 Semiconductors -> SOXX',  etf_a == 'SOXX', str(etf_a))
check('T7 Biotechnology  -> XBI',   etf_b == 'XBI',  str(etf_b))
check('T7 Solar          -> TAN',   etf_c == 'TAN',  str(etf_c))
check('T7 get_sector_etf fallback', 'def get_sector_etf' in src)

print('\n'.join(results))
all_ok = all('[OK]' in r for r in results)
print(f'\n{"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
sys.exit(0 if all_ok else 1)
