"""
Stage 2 - Anomaly register: separate ERRORS (fixed in clean.py) from REAL EVENTS (modelled/excluded).

    python src/anomalies.py   ->  reports/anomaly_register.csv, reports/anomaly_summary.json

Every flag carries a classification and the exact action the forecaster takes, so the register is
consistent with the model (same masks are used in features.Panel).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import LAST_OBS, M0, Panel, col, intermittency  # noqa: E402
from utils import int_to_ym  # noqa: E402

warnings.filterwarnings('ignore')
REP = ROOT / 'reports'
P = Panel()
tr = P.train
meta = P.meta
DATES = int_to_ym(np.arange(LAST_OBS + 1) + M0)
reg = []


def add(rows, months, atype, cls, reason, action, expected=None):
    for k, (i, m) in enumerate(zip(rows, months)):
        reg.append(dict(Date=DATES[m], SKU_ID=meta['SKU_ID'].iat[i], Region=meta['Region'].iat[i],
                        Product_Family=meta['Product_Family'].iat[i], Customer_Name=meta['Customer_Name'].iat[i],
                        Units_Sold=P.Ship[i, m], Unconstrained_Demand=P.Y[i, m],
                        Expected_Level=None if expected is None else round(float(expected[k]), 1),
                        Anomaly_Type=atype, Classification=cls,
                        Reason=reason if isinstance(reason, str) else reason[k], Action=action))


Y = P.Y[:, :LAST_OBS + 1]
lvl = pd.DataFrame(np.where(Y > 0, Y, np.nan).T).rolling(13, center=True, min_periods=3).median().T.values

# 1. one-off orders (real, non-recurring) -----------------------------------------------
r, m = np.where(P.spike[:, :LAST_OBS + 1])
add(r, m, 'One-off order spike', 'Real event', [f'{Y[i, j] / lvl[i, j]:.1f}x the series level' for i, j in zip(r, m)],
    'Kept in history; capped at 4x level for model training', lvl[r, m])

# 2. stockouts, back-order catch-up, unlisted chip-shortage months ------------------------
key = {k: i for i, k in enumerate(zip(meta['SKU_ID'], meta['Region']))}
for flag, atype, action in [
        (tr['Is_Stockout'], 'Stockout (listed)', 'Unconstrained demand = Requested_Units'),
        (tr['Demand_Adjustment'].eq('back-order catch-up removed'), 'Back-order catch-up',
         'Previous month back-orders removed from demand'),
        (tr['Demand_Adjustment'].str.startswith('unlisted'), 'Stockout (not in stockout file)',
         'Demand = shipments / regional fill rate; window excluded from training')]:
    s = tr[flag]
    rows = [key[k] for k in zip(s['SKU_ID'], s['Region'])]
    months = (s['Date'].str[:4].astype(int) * 12 + s['Date'].str[5:7].astype(int) - 1 - M0).values
    reason = (s['Reason'].fillna('Semiconductor allocation (EV-019)') + ', fill rate ' +
              s['Fill_Rate_Pct'].round(0).astype('Int64').astype(str).str.replace('<NA>', 'n/a') + '%').values
    add(rows, months, atype, 'Real event', reason, action)

# 3. COVID disruption: months where a series fell below 60% of its level ---------------------
for i, j in zip(*np.where(P.no_feat[:, :LAST_OBS + 1] & (P.shutdown[:, :LAST_OBS + 1] == 0))):
    if Y[i, j] < 0.6 * lvl[i, j]:
        add([i], [j], 'Disruption (COVID-19 lockdown)', 'Real event', 'Regional lockdown (events calendar)',
            'Excluded from training targets and level features', [lvl[i, j]])

# 4. customer events -----------------------------------------------------------------------
r, m = np.where(P.shutdown[:, :LAST_OBS + 1] > 0)
add(r, m, 'Customer event (strike)', 'Real event', 'Nexon Motor Group Europe labour strike (EV-038)',
    'Excluded from training; used as analog for Orion NA shutdown 2027-07..08', lvl[r, m])
h = np.where((meta['Customer_Name'] == 'Helios Auto Parts') & (meta['Region'] == 'India'))[0]
pre = np.nanmean(Y[h, col('2024-09'):col('2025-09')])
post = np.nanmean(Y[h, col('2025-09'):])
add(h, [col('2025-09')] * len(h), 'Structural break (capacity step-up)', 'Real event',
    f'Helios second Indian line (EV-052): +{(post / pre - 1) * 100:.0f}% level shift',
    'Kept; recent-level features carry the new level forward')

# 5. pre-buy before the announced 9% price rise ------------------------------------------------
tot = pd.Series(np.nansum(Y, 0), DATES)
norm = tot / tot.rolling(13, center=True, min_periods=7).median()
for d in ['2022-11', '2022-12', '2023-01', '2023-02']:
    reg.append(dict(Date=d, SKU_ID='ALL', Region='ALL', Product_Family='ALL', Customer_Name='ALL',
                    Units_Sold=None, Unconstrained_Demand=float(tot[d]), Expected_Level=None,
                    Anomaly_Type='Pre-buy / payback (9% price rise)', Classification='Real event',
                    Reason=f'Total demand {norm[d]:.2f}x trend (announcement EV-030)',
                    Action='Excluded from training targets (smaller 2027/28 rises showed no pre-buy)'))

# 6. Red Sea export delay ----------------------------------------------------------------------------
ex = meta['Sales_Channel'].values == 'Export'
for d in ['2024-01', '2024-02', '2024-03']:
    j = col(d)
    reg.append(dict(Date=d, SKU_ID='ALL', Region='ALL', Product_Family='ALL', Customer_Name='Export channel',
                    Units_Sold=None, Unconstrained_Demand=float(np.nansum(Y[ex, j])), Expected_Level=None,
                    Anomaly_Type='Logistics delay (Red Sea)', Classification='Real event',
                    Reason='Export shipments delayed then caught up (EV-039)', Action='Excluded from training targets'))

# 7. missing rows (not zeros) ------------------------------------------------------------------------
obs = ~np.isnan(Y)
first = obs.argmax(1)
last = LAST_OBS - obs[:, ::-1].argmax(1)
for i in range(len(meta)):
    if not obs[i].any():
        continue
    gaps = np.where(~obs[i, first[i]:last[i] + 1])[0] + first[i]
    for j in gaps:
        erp = meta['Region'].iat[i] == 'Latin America' and '2023-03' <= DATES[j] <= '2023-05'
        add([i], [j], 'Missing row', 'Missing data',
            'Latin America ERP migration (EV-033)' if erp else 'No row in extract (and not a zero)',
            'Left as missing - never imputed as 0')

# 8. order-cycle changes, customer switches, lost accounts -------------------------------------------
ic = intermittency(P)
chg = ic[(ic['Order_Cycle_Months'] != ic['Previous_Cycle_Months']) & ic['Previous_Cycle_Months'].notna()]
for _, c in chg.iterrows():
    i = key[(c['SKU_ID'], c['Region'])]
    add([i], [LAST_OBS - 23], 'Ordering-cycle change', 'Real event',
        f"Order every {int(c['Previous_Cycle_Months'])} -> every {int(c['Order_Cycle_Months'])} month(s)",
        'Phase features use only the latest 12 months')
cs = tr.sort_values('Date')[['SKU_ID', 'Region', 'Date', 'Customer_Name']]
cs['prev'] = cs.groupby(['SKU_ID', 'Region'])['Customer_Name'].shift()
sw = cs[cs['prev'].notna() & (cs['prev'] != cs['Customer_Name'])]
sw = sw.groupby(['SKU_ID', 'Region', 'prev', 'Customer_Name']).head(1)
for _, s in sw.iterrows():
    i = key[(s['SKU_ID'], s['Region'])]
    j = col(s['Date'])
    add([i], [j], 'Customer switch', 'Real event', f"{s['prev']} -> {s['Customer_Name']}",
        'Latest customer used as feature; history kept (demand continues)')
for i in np.where(obs.any(1) & (last < LAST_OBS - 6))[0]:
    if P.eol[i] <= last[i] + 1:
        continue
    add([i], [last[i]], 'Lost account / series ended', 'Real event',
        'No orders after this month and no EOL in master', 'Not forecast unless present in test.csv')

# 9. master data vs transactions -------------------------------------------------------------------
c = tr[tr['Lifecycle_Conflict'] != '']
add([key[k] for k in zip(c['SKU_ID'], c['Region'])], [col(d) for d in c['Date']],
    'Master data conflict', 'Master data conflict', c['Lifecycle_Conflict'].map(
        {'before_launch': 'Sale before sku_master Launch_Date', 'after_EOL': 'Sale after EOL'}).values,
    'Transactions kept as evidence; master flagged for correction')

# 10. remaining statistical outliers (robust z on log scale), not explained above ----------------------
flagged = {(a['SKU_ID'], a['Region'], a['Date']) for a in reg}
with np.errstate(all='ignore'):
    z = np.abs(np.log1p(Y) - np.log1p(lvl)) / 0.35
cand = (z > 4) & (Y > 0) & ~P.spike[:, :LAST_OBS + 1]
for i, j in zip(*np.where(cand)):
    if (meta['SKU_ID'].iat[i], meta['Region'].iat[i], DATES[j]) not in flagged:
        add([i], [j], 'Statistical outlier (unexplained)', 'Unexplained',
            f'{Y[i, j] / lvl[i, j]:.2f}x rolling median', 'Kept; L1 loss limits its influence', [lvl[i, j]])

out = pd.DataFrame(reg)
out.to_csv(REP / 'anomaly_register.csv', index=False)
summary = out.groupby(['Classification', 'Anomaly_Type']).size().rename('rows').reset_index()
(REP / 'anomaly_summary.json').write_text(json.dumps(summary.to_dict('records'), indent=2))
print(summary.to_string(index=False))
print(f'Total flags: {len(out):,}')
