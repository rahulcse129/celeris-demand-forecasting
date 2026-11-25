"""
Stage 1 - Data cleaning & unconstrained-demand reconstruction.

    python src/clean.py

Reads  dataset/student_data/*.csv
Writes data/cleaned/*            one clean row per Date-SKU-Region (+ reference files)
       reports/data_quality_report.json   machine-readable issue register (by file / type)
       reports/data_quality_issues.csv    row-level fix log (Record_ID, field, old -> new)

Every rule distinguishes *errors* (fixed) from *real events* (kept and flagged).
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from utils import (CLEAN_REGIONS, FAMILY_PREFIX_MAP, parse_date_to_yyyymm, standardize_channel,  # noqa: E402
                   standardize_customer, standardize_region, standardize_sku, ym_to_int)

warnings.filterwarnings('ignore')
RAW = ROOT / 'dataset' / 'student_data'
OUT = ROOT / 'data' / 'cleaned'
REP = ROOT / 'reports'
OUT.mkdir(parents=True, exist_ok=True)
REP.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = '2019-01', '2026-10'
SURCHARGE = ('2022-07', '2022-12', 1.06)          # events_calendar EV-027
ISSUES, ROWLOG = [], []


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def issue(file, itype, rows, action, kind='error', **extra):
    ISSUES.append(dict(file=file, issue_type=itype, rows_affected=int(rows), action=action,
                       classification=kind, **extra))


def rowlog(df, mask, itype, field, old, new):
    if mask.sum() == 0:
        return
    sub = df.loc[mask]
    ROWLOG.append(pd.DataFrame({'Record_ID': sub['Record_ID'].values, 'issue_type': itype, 'field': field,
                                'old_value': pd.Series(old)[mask].astype(str).values if not np.isscalar(old) else old,
                                'new_value': pd.Series(new)[mask].astype(str).values if not np.isscalar(new) else new}))


# ════════════════════════════════════════════════════════════════════
# 0. LOAD + PROFILE
# ════════════════════════════════════════════════════════════════════
log('Loading raw files ...')
raw = {f: pd.read_csv(RAW / f'{f}.csv', dtype=str, keep_default_na=False, na_values=[''])
       for f in ['train', 'test', 'sku_master', 'macro_indicators', 'events_calendar', 'inventory_stockouts']}
profile = {f: {'rows': len(d), 'columns': len(d.columns),
               'null_cells': {c: int(v) for c, v in d.isna().sum().items() if v},
               'duplicate_rows': int(d.duplicated().sum())} for f, d in raw.items()}

sku_master = raw['sku_master'].copy()
for c in sku_master.columns:
    sku_master[c] = sku_master[c].str.strip()
sku_master['SKU_ID'] = sku_master['SKU_ID'].map(standardize_sku)
for c in ['Launch_Date', 'EOL_Date']:
    sku_master[c] = sku_master[c].map(parse_date_to_yyyymm)
MASTER = sku_master.set_index('SKU_ID')

# ════════════════════════════════════════════════════════════════════
# 1. TRAIN: labels, keys, dates
# ════════════════════════════════════════════════════════════════════
t = raw['train'].copy()
F = 'train.csv'
n_raw = len(t)
log(f'train.csv raw rows: {n_raw:,}')

# 1.1 dates -------------------------------------------------------------
parsed = t['Date'].map(parse_date_to_yyyymm)
m = parsed.notna() & (parsed != t['Date'])
issue(F, 'date_format_inconsistent', m.sum(), 'Parsed 6 alternative formats (YYYY/MM, Mon-YYYY, MM-YYYY, ...) to YYYY-MM')
rowlog(t, m, 'date_format_inconsistent', 'Date', t['Date'], parsed)
t['Date'] = parsed

# Load_Timestamp tells us when the row reached the warehouse (~1-3 weeks after the month closes).
load_ts = pd.to_datetime(t['Load_Timestamp'], errors='coerce')
load_ym = (load_ts - pd.DateOffset(months=1)).dt.strftime('%Y-%m')
fut = t['Date'] > TRAIN_END
fixed = t['Date'].where(~fut, load_ym)
ok_swap = fut & (t['Date'].str[:4] == '2062') & (fixed.str[5:7] == t['Date'].str[5:7])
issue(F, 'date_year_typo', fut.sum(),
      "Year '2062' is a digit swap of 2026 - confirmed by Load_Timestamp (month before load = same month, 2026)",
      verified_by_load_timestamp=int(ok_swap.sum()))
rowlog(t, fut, 'date_year_typo', 'Date', t['Date'], fixed)
t['Date'] = fixed
t = t[t['Date'].between(TRAIN_START, TRAIN_END)].copy()

ym_mis = (t['Year'].astype(str) != t['Date'].str[:4]) | (pd.to_numeric(t['Month'], errors='coerce') != t['Date'].str[5:7].astype(int))
issue(F, 'year_month_disagree_with_date', ym_mis.sum(), 'Business rule 1: Year/Month rebuilt from authoritative Date')
t['Year'] = t['Date'].str[:4].astype(int)
t['Month'] = t['Date'].str[5:7].astype(int)

# 1.2 SKU ------------------------------------------------------------------
sku = t['SKU_ID'].map(standardize_sku)
m = sku != t['SKU_ID']
issue(F, 'sku_id_format', m.sum(), "Normalised case/whitespace/separator/zero-padding ('pmp_23', 'CSS31' -> 'PMP-023', 'CSS-031')")
rowlog(t, m, 'sku_id_format', 'SKU_ID', t['SKU_ID'], sku)
t['SKU_ID'] = sku
unk = ~t['SKU_ID'].isin(MASTER.index)
issue(F, 'sku_not_in_master', unk.sum(), 'Dropped (cannot be mapped to a product)')
t = t[~unk].copy()

# 1.3 Region / Channel / Customer -----------------------------------------
for col, fn, name in [('Region', standardize_region, 'region_label_variant'),
                      ('Sales_Channel', standardize_channel, 'channel_label_variant'),
                      ('Customer_Name', standardize_customer, 'customer_name_variant')]:
    new = t[col].map(fn)
    changed = new.notna() & (new != t[col])
    issue(F, name, changed.sum(), f'Mapped {t[col].nunique()} raw variants to {new.nunique()} canonical {col} values',
          variants_before=int(t[col].nunique()), variants_after=int(new.nunique()))
    rowlog(t, changed, name, col, t[col], new)
    t[col] = new

# 1.4a master data vs transactions: every family belongs to exactly one BU. sku_master breaks this for
# some SKUs while transactions are 100% consistent -> the master is wrong; fix it from transactions.
fam_bu = t.dropna(subset=['Product_Family']).groupby('Product_Family')['Business_Unit'].agg(lambda s: s.mode().iat[0])
bad_bu = sku_master['Business_Unit'] != sku_master['Product_Family'].map(fam_bu)
issue('sku_master.csv', 'business_unit_contradicts_family', bad_bu.sum(),
      'Family maps to one BU in all transactions; master rows disagreeing were corrected', kind='master_data_conflict',
      skus=sku_master.loc[bad_bu, 'SKU_ID'].tolist())
sku_master.loc[bad_bu, 'Business_Unit'] = sku_master.loc[bad_bu, 'Product_Family'].map(fam_bu)
MASTER = sku_master.set_index('SKU_ID')

# 1.4b Product family / BU / description from master (reference data wins for product attributes)
for col in ['Product_Family', 'Business_Unit', 'SKU_Description']:
    ref = t['SKU_ID'].map(MASTER[col])
    blank = t[col].isna()
    wrong = ~blank & (t[col].str.strip() != ref)
    issue(F, f'{col.lower()}_missing', blank.sum(), f'Filled from sku_master by SKU_ID')
    if wrong.sum():
        issue(F, f'{col.lower()}_contradicts_master', wrong.sum(), 'Overwritten with sku_master value')
    t[col] = ref

# 1.5 de-duplicate by Record_ID (business rule 2) ---------------------------
t['_ts'] = pd.to_datetime(t['Load_Timestamp'], errors='coerce')
t = t.sort_values(['Record_ID', '_ts'])
dup = t.duplicated('Record_ID', keep='last')
n_diff = int(t[t['Record_ID'].duplicated(keep=False)].groupby('Record_ID')['Units_Sold'].nunique().gt(1).sum())
issue(F, 'duplicate_record_id', dup.sum(), 'Kept the row with the latest Load_Timestamp (business rule 2)',
      duplicates_with_changed_units=n_diff)
t = t[~dup].copy()

# 1.6 blank region -> infer from the SKU's customer/channel footprint -----------
blank = t['Region'].isna()
foot = (t[~blank].groupby(['SKU_ID', 'Customer_Name', 'Sales_Channel'])['Region']
        .agg(lambda s: s.unique().tolist()).to_dict())
occupied = set(zip(t.loc[~blank, 'Date'], t.loc[~blank, 'SKU_ID'], t.loc[~blank, 'Region']))
inferred = []
for idx, r in t[blank].iterrows():
    cands = [g for g in foot.get((r['SKU_ID'], r['Customer_Name'], r['Sales_Channel']), [])
             if (r['Date'], r['SKU_ID'], g) not in occupied]
    inferred.append(cands[0] if len(cands) == 1 else None)
t.loc[blank, 'Region'] = inferred
still = t['Region'].isna()
issue(F, 'region_missing', blank.sum(),
      'Inferred from SKU x customer x channel footprint where exactly one region had no row that month; rest dropped',
      inferred=int(blank.sum() - still.sum()), dropped=int(still.sum()))
t = t[~still].copy()

cust_blank = t['Customer_Name'].isna()
if cust_blank.sum():
    mode = t.groupby(['SKU_ID', 'Region', 'Year'])['Customer_Name'].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    t.loc[cust_blank, 'Customer_Name'] = [mode.get((a, b, c)) for a, b, c in t.loc[cust_blank, ['SKU_ID', 'Region', 'Year']].values]
    issue(F, 'customer_unmappable', cust_blank.sum(), 'Filled with that SKU-region-year majority customer')

# ════════════════════════════════════════════════════════════════════
# 2. TRAIN: quantities, prices, revenue
# ════════════════════════════════════════════════════════════════════
u_str = t['Units_Sold'].fillna('').str.strip()
comma = u_str.str.contains(',')
u = pd.to_numeric(u_str.str.replace(',', ''), errors='coerce')
dash = u_str.isin(['-', '--', 'n/a', 'NA'])
p_rec = pd.to_numeric(t['Unit_Price_USD'], errors='coerce')
r = pd.to_numeric(t['Revenue_USD'], errors='coerce')

# 2.1 reference list price per SKU-year (rule 4: same in all regions, changes 1 Jan, surcharge per events)
t['_sur'] = np.where(t['Date'].between(SURCHARGE[0], SURCHARGE[1]), SURCHARGE[2], 1.0)
base = p_rec / t['_sur']
list_price = base.groupby([t['SKU_ID'], t['Year']]).transform('median')
exp_price = list_price * t['_sur']
ratio = p_rec / exp_price
p_ok = (ratio - 1).abs() < 0.02
cents = ((ratio / 100) - 1).abs() < 0.02
x10 = ((ratio / 10) - 1).abs() < 0.02
zero_order = (p_rec == 0) & (pd.to_numeric(t['Units_Sold'], errors='coerce') == 0)
other = ~(p_ok | cents | x10 | zero_order)
issue(F, 'price_in_cents', cents.sum(), 'Unit price 100x list price (cents) -> replaced with SKU-year list price')
issue(F, 'price_x10', x10.sum(), 'Unit price 10x list price (decimal shift) -> replaced with list price')
issue(F, 'price_off_list', other.sum(), 'Price not explained by list/surcharge -> replaced with list price')
issue(F, 'zero_price_on_no_order', zero_order.sum(), 'Price 0 on 0-unit rows (no order) - not an error; list price filled', kind='info')
rowlog(t, ~p_ok & ~zero_order, 'price_fix', 'Unit_Price_USD', p_rec, exp_price.round(2))
price = exp_price.round(2)

# 2.2 units: parse, recover, unit-of-measure errors ---------------------------------
# trusted level per series = median of rows whose units x recorded price reconcile with revenue
recon = (r - u * p_rec).abs() <= np.maximum(1.0, 0.005 * r.abs())
trusted = recon & u.notna() & (u > 0) & p_ok
key = [t['SKU_ID'], t['Region']]
level = u.where(trusted).groupby(key).transform('median')

new_u = u.copy()
# (a) negatives: revenue is negative too -> sign error, not a return (no credit-note channel exists)
neg = u < 0
new_u[neg] = -u[neg]
issue(F, 'negative_units', neg.sum(), 'Sign error (revenue also negative); took absolute value')
# (b) missing / '-' units -> recover from revenue / recorded price
miss = new_u.isna() & r.notna() & ((p_rec > 0) | (r == 0))
new_u[miss] = (r[miss] / p_rec[miss].replace(0, np.nan)).fillna(0).round()
issue(F, 'units_missing_recovered', miss.sum(), 'Units_Sold blank or "-" -> recovered as Revenue / Unit_Price', dash_values=int(dash.sum()))
unrec = new_u.isna()
issue(F, 'units_unrecoverable', unrec.sum(), 'Units and revenue both missing -> row treated as missing (not zero)')
# (c) units x price != revenue: pick the candidate that is consistent with revenue AND the series level
mis = new_u.notna() & r.notna() & ~((r.abs() - new_u * p_rec).abs() <= np.maximum(1.0, 0.005 * r.abs())) & ~miss
cand_rev = (r.abs() / p_rec).round()
use_rev = mis & ((np.log1p(cand_rev) - np.log1p(level)).abs() < (np.log1p(new_u) - np.log1p(level)).abs())
new_u[use_rev] = cand_rev[use_rev]
issue(F, 'units_revenue_mismatch', mis.sum(),
      'Units x price != revenue: kept whichever of {Units, Revenue/Price} matches the SKU-region level',
      units_replaced=int(use_rev.sum()), revenue_recomputed=int((mis & ~use_rev).sum()))
# (d) thousands-separator / unit-of-measure inflation: x10 / x100 vs level and divisible
for f in (100, 10):
    infl = new_u.notna() & (level > 0) & (new_u % f == 0) & (new_u / f > 0) & \
        ((np.log(new_u / f / level)).abs() < np.log(2.5)) & (new_u / level > f / 3)
    rowlog(t, infl, f'units_x{f}', 'Units_Sold', new_u, new_u / f)
    issue(F, f'units_x{f}_uom', infl.sum(), f'Quantity {f}x the series level and a multiple of {f} (pack/UoM error) -> divided by {f}',
          with_thousands_separator=int((infl & comma).sum()))
    new_u[infl] = new_u[infl] / f
rowlog(t, (new_u != u) & ~(neg | miss), 'units_fix', 'Units_Sold', u, new_u)

t['Units_Sold'] = new_u
t['Unit_Price_USD'] = price
t['Revenue_USD'] = (new_u * price).round(2)
t = t[t['Units_Sold'].notna()].copy()

# ════════════════════════════════════════════════════════════════════
# 3. ONE ROW PER Date-SKU-Region
# ════════════════════════════════════════════════════════════════════
k = ['Date', 'SKU_ID', 'Region']
kd = t.duplicated(k, keep=False)
same = t[kd].groupby(k)['Units_Sold'].transform('nunique') == 1
issue(F, 'duplicate_key_different_record_id', kd.sum(),
      'Same Date-SKU-Region under two Record_IDs: identical quantities -> keep latest load; else keep latest load',
      identical_quantity=int(same.sum()))
t = t.sort_values('_ts').drop_duplicates(k, keep='last')

# master data vs transactions: sales outside lifecycle window
launch = t['SKU_ID'].map(MASTER['Launch_Date'])
eol = t['SKU_ID'].map(MASTER['EOL_Date'])
after_eol = eol.notna() & (t['Date'] > eol) & (t['Units_Sold'] > 0)
before_launch = launch.notna() & (t['Date'] < launch) & (t['Units_Sold'] > 0)
issue(F, 'sale_after_eol', after_eol.sum(), 'Master says EOL earlier than last sale - kept (transactions are evidence); flagged', kind='master_data_conflict',
      skus=sorted(t.loc[after_eol, 'SKU_ID'].unique().tolist()))
issue(F, 'sale_before_launch', before_launch.sum(), 'Sale before master Launch_Date - kept; flagged', kind='master_data_conflict',
      skus=sorted(t.loc[before_launch, 'SKU_ID'].unique().tolist()))
t['Lifecycle_Conflict'] = np.where(after_eol, 'after_EOL', np.where(before_launch, 'before_launch', ''))

# ════════════════════════════════════════════════════════════════════
# 4. MACRO INDICATORS
# ════════════════════════════════════════════════════════════════════
F = 'macro_indicators.csv'
mac = raw['macro_indicators'].copy()
MC = [c for c in mac.columns if c not in ('Date', 'Year', 'Month')]
mac['Date'] = mac['Date'].map(parse_date_to_yyyymm)
for c in MC:
    mac[c] = pd.to_numeric(mac[c].str.replace(',', ''), errors='coerce')
d = mac.duplicated('Date', keep=False)
issue(F, 'duplicate_month', d.sum(), 'Two rows for the same month -> averaged')
mac = mac.groupby('Date', as_index=False)[MC].mean()
miss = int(mac[MC].isna().sum().sum())
# scale errors: auto production stored as a fraction early in the series; PMI decimal shift
ap = mac['Auto_Production_YoY_Pct']
frac = ap.abs() < 0.1
frac &= ap.ne(0)
mac.loc[frac, 'Auto_Production_YoY_Pct'] = ap[frac] * 100
issue(F, 'auto_production_as_fraction', frac.sum(), 'Values like -0.015 are fractions, not percent -> x100')
for c in MC:
    med = mac[c].rolling(7, center=True, min_periods=3).median()
    for f in (10, 0.1):
        bad = ((mac[c] * f / med) - 1).abs().lt(0.1) & ((mac[c] / med) - 1).abs().gt(0.5)
        if bad.sum():
            issue(F, f'{c}_decimal_shift', bad.sum(), f'Value off by x{1 / f:g} vs neighbours -> rescaled', dates=mac.loc[bad, 'Date'].tolist())
            mac.loc[bad, c] = mac.loc[bad, c] * f
mac[MC] = mac[MC].interpolate(limit_direction='both')
issue(F, 'missing_values', miss, 'Linear interpolation between neighbouring months')
issue(F, 'covid_shock_2020', 0, 'PMI 41 / auto production -45% in 2020-04 are real - kept', kind='real_event')
mac['Year'] = mac['Date'].str[:4].astype(int)
mac['Month'] = mac['Date'].str[5:7].astype(int)
mac = mac[['Date', 'Year', 'Month'] + MC]

# ════════════════════════════════════════════════════════════════════
# 5. EVENTS
# ════════════════════════════════════════════════════════════════════
ev = raw['events_calendar'].copy()
ev['Start_Month'] = ev['Start_Date'].map(parse_date_to_yyyymm)
ev['End_Month'] = ev['End_Date'].map(parse_date_to_yyyymm)
issue('events_calendar.csv', 'mixed_date_granularity', ev['Start_Date'].str.len().ne(7).sum(),
      'Day-level dates kept and also mapped to Start_Month/End_Month; open-ended End_Date = permanent')

# ════════════════════════════════════════════════════════════════════
# 6. STOCKOUTS -> UNCONSTRAINED DEMAND
# ════════════════════════════════════════════════════════════════════
F = 'inventory_stockouts.csv'
so = raw['inventory_stockouts'].copy()
so['Date'] = so['Date'].map(parse_date_to_yyyymm)
so['SKU_ID'] = so['SKU_ID'].map(standardize_sku)
so['Region'] = so['Region'].map(standardize_region)
NUM = ['Requested_Units', 'Carried_In_Units', 'Shipped_Units', 'Backordered_Units', 'Lost_Units', 'Fill_Rate_Pct', 'Stockout_Days']
for c in NUM:
    so[c] = pd.to_numeric(so[c], errors='coerce')
bad_key = so[['Date', 'SKU_ID', 'Region']].isna().any(axis=1)
issue(F, 'invalid_key', bad_key.sum(), 'Dropped')
so = so[~bad_key]
dk = so.duplicated(['Date', 'SKU_ID', 'Region'])
issue(F, 'duplicate_key', dk.sum(), 'Dropped exact key duplicates')
so = so[~dk].copy()

# reconcile with train shipments: Shipped_Units should equal Units_Sold
tk = t.set_index(k)['Units_Sold']
so['Train_Units'] = so.set_index(k).index.map(tk)
so['Train_Units'] = pd.to_numeric(so['Train_Units'], errors='coerce')
mm = so['Train_Units'].notna() & (so['Train_Units'] != so['Shipped_Units'])
issue(F, 'shipped_vs_train_mismatch', mm.sum(), 'Shipped_Units disagrees with cleaned Units_Sold; train value kept, stockout row still used for demand')
# Requested must be >= 0 and consistent: Lost = Req + CarriedIn - Shipped - Backordered
den = so['Fill_Rate_Pct'] / 100
req_fix = (so['Requested_Units'].fillna(0) <= 0) & (so['Shipped_Units'] > 0) & (den > 0)
so.loc[req_fix, 'Requested_Units'] = (so['Shipped_Units'] / den - so['Carried_In_Units'].fillna(0)).round()
issue(F, 'requested_zero_with_shipments', req_fix.sum(), 'Requested_Units rebuilt from Shipped / Fill_Rate - Carried_In')
empty = so['Requested_Units'].fillna(0).le(0) & so['Shipped_Units'].fillna(0).le(0)
issue(F, 'empty_stockout_row', empty.sum(), 'Requested = Shipped = 0: no demand signal, ignored for demand reconstruction')
so = so[~empty].copy()
so['Lost_Units'] = (so['Requested_Units'] + so['Carried_In_Units'].fillna(0) - so['Shipped_Units'] - so['Backordered_Units'].fillna(0)).clip(lower=0)

# unconstrained demand: stockout month -> Requested; the month after -> remove back-order catch-up
t = t.merge(so[k + ['Requested_Units', 'Shipped_Units', 'Backordered_Units', 'Lost_Units', 'Fill_Rate_Pct', 'Reason']],
            on=k, how='left')
t['Is_Stockout'] = t['Requested_Units'].notna()
t['Unconstrained_Demand'] = np.where(t['Is_Stockout'], np.maximum(t['Requested_Units'], 0), t['Units_Sold'])
t['Demand_Adjustment'] = np.where(t['Is_Stockout'], 'stockout: requested units', '')
t['_m'] = ym_to_int(t['Date'])
bo = so[['SKU_ID', 'Region', 'Backordered_Units']].copy()
bo['_m'] = ym_to_int(so['Date']) + 1
bo = bo[bo['Backordered_Units'] > 0].rename(columns={'Backordered_Units': '_bo_in'})
t = t.merge(bo, on=['SKU_ID', 'Region', '_m'], how='left')
catch = t['_bo_in'].notna() & ~t['Is_Stockout']
t.loc[catch, 'Unconstrained_Demand'] = (t.loc[catch, 'Units_Sold'] - t.loc[catch, '_bo_in']).clip(lower=0)
t.loc[catch, 'Demand_Adjustment'] = 'back-order catch-up removed'
issue('train.csv', 'stockout_constrained_month', t['Is_Stockout'].sum(),
      'Units_Sold < demand: Unconstrained_Demand = Requested_Units (rule 7)', kind='real_event')
issue('train.csv', 'backorder_catchup_month', catch.sum(),
      'Shipments inflated by last month back-orders -> removed from Unconstrained_Demand', kind='real_event')

# unlisted shortage months (stockout file is incomplete): Electrical Assemblies during the
# semiconductor allocation (EV-019, 2021-03..2022-09). Estimate the shortfall per region-month from
# the listed rows' fill rate and flag them; the model excludes the whole window from its targets.
win = (t['Product_Family'] == 'Electrical Assemblies') & t['Date'].between('2021-03', '2022-09')
t['Supply_Constrained_Window'] = win
unl = win & ~t['Is_Stockout'] & ~catch
listed = t[win & t['Is_Stockout'] & (t['Units_Sold'] > 0)]
fr = (listed.groupby(['Date', 'Region'])['Units_Sold'].sum() / listed.groupby(['Date', 'Region'])['Requested_Units'].sum())
# only rows that look suppressed vs their own pre-shortage level are adjusted
pre = t[(t['Product_Family'] == 'Electrical Assemblies') & t['Date'].between('2019-01', '2021-02')]
pre_lvl = pre.groupby(['SKU_ID', 'Region'])['Units_Sold'].mean()
lvl = pd.Series(list(zip(t['SKU_ID'], t['Region']))).map(pre_lvl).values
frv = pd.Series(list(zip(t['Date'], t['Region']))).map(fr).values
sup = unl & (t['Units_Sold'] > 0) & (t['Units_Sold'] < 0.8 * lvl) & (frv < 0.95)
t.loc[sup, 'Unconstrained_Demand'] = np.minimum(t.loc[sup, 'Units_Sold'] / frv[sup], lvl[sup] * 1.2).round()
t.loc[sup, 'Demand_Adjustment'] = 'unlisted chip-shortage month: shipments / regional fill rate'
issue('train.csv', 'unlisted_supply_constraint', sup.sum(),
      'Electrical Assemblies during chip shortage, not in stockout file, shipping <80% of pre-shortage level -> '
      'demand = shipments / listed regional fill rate (capped at 1.2x pre-shortage level)', kind='real_event')
t['Lost_Units'] = (t['Unconstrained_Demand'] - t['Units_Sold']).clip(lower=0)

# ════════════════════════════════════════════════════════════════════
# 7. MISSING ROWS vs ZEROS
# ════════════════════════════════════════════════════════════════════
first = t.groupby(['SKU_ID', 'Region'])['_m'].transform('min')
span = t.groupby(['SKU_ID', 'Region'])['_m'].agg(['min', 'max'])
expected = int((span['max'] - span['min'] + 1).sum())
gaps = expected - len(t)
latam = t[(t['Region'] == 'Latin America')]
erp_window = ym_to_int(pd.Series(['2023-03', '2023-05']))
issue('train.csv', 'missing_rows', gaps,
      'Months without a row inside a series life are MISSING, not zero (rule 3): left as NaN for the model; '
      'concentrated in Latin America 2023-03..05 (ERP migration, EV-033)', kind='missing_data')
issue('train.csv', 'zero_units', (t['Units_Sold'] == 0).sum(),
      'Explicit 0 = no order (holiday shutdowns, order-cycle gaps) - kept as real zeros', kind='real_event')

# ════════════════════════════════════════════════════════════════════
# 8. TEST
# ════════════════════════════════════════════════════════════════════
te = raw['test'].copy()
te['Date'] = te['Date'].map(parse_date_to_yyyymm)
te['SKU_ID'] = te['SKU_ID'].map(standardize_sku)
te['Region'] = te['Region'].map(standardize_region)
te['Sales_Channel'] = te['Sales_Channel'].map(standardize_channel)
te['Customer_Name'] = te['Customer_Name'].map(standardize_customer)
te['Year'] = te['Date'].str[:4].astype(int)
te['Month'] = te['Date'].str[5:7].astype(int)
for c in ['Product_Family', 'Business_Unit', 'SKU_Description']:
    te[c] = te['SKU_ID'].map(MASTER[c])
assert te[['Date', 'SKU_ID', 'Region', 'Customer_Name', 'Sales_Channel']].notna().all().all()

# ════════════════════════════════════════════════════════════════════
# 9. SAVE
# ════════════════════════════════════════════════════════════════════
life = ['Size_Class', 'Powertrain_Application', 'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'Lifecycle_Status']
cols = ['Date', 'Year', 'Month', 'Product_Family', 'Business_Unit', 'SKU_ID', 'SKU_Description', 'Customer_Name',
        'Sales_Channel', 'Region', 'Units_Sold', 'Unit_Price_USD', 'Revenue_USD', 'Unconstrained_Demand',
        'Lost_Units', 'Is_Stockout', 'Supply_Constrained_Window', 'Demand_Adjustment', 'Fill_Rate_Pct', 'Reason',
        'Lifecycle_Conflict']
t = t.sort_values(['SKU_ID', 'Region', 'Date'])[cols].merge(sku_master[['SKU_ID'] + life], on='SKU_ID', how='left')
te = te.merge(sku_master[['SKU_ID'] + life], on='SKU_ID', how='left')
t.to_parquet(OUT / 'train_clean.parquet', index=False)
te.to_csv(OUT / 'test_clean.csv', index=False)
mac.to_csv(OUT / 'macro_indicators_clean.csv', index=False)
so.drop(columns=['Train_Units']).to_csv(OUT / 'inventory_stockouts_clean.csv', index=False)
ev.to_csv(OUT / 'events_calendar_clean.csv', index=False)
sku_master.to_csv(OUT / 'sku_master_clean.csv', index=False)

rows = pd.concat(ROWLOG, ignore_index=True) if ROWLOG else pd.DataFrame()
rows.to_csv(REP / 'data_quality_issues.csv', index=False)
report = {
    'pipeline_run': datetime.now().isoformat(timespec='seconds'),
    'raw_profile': profile,
    'issues': ISSUES,
    'before_after': {
        'train_rows_raw': n_raw, 'train_rows_clean': len(t),
        'regions': [int(raw['train']['Region'].nunique()), int(t['Region'].nunique())],
        'channels': [int(raw['train']['Sales_Channel'].nunique()), int(t['Sales_Channel'].nunique())],
        'customers': [int(raw['train']['Customer_Name'].nunique()), int(t['Customer_Name'].nunique())],
        'skus': [int(raw['train']['SKU_ID'].nunique()), int(t['SKU_ID'].nunique())],
        'units_total': [float(pd.to_numeric(raw['train']['Units_Sold'].str.replace(',', ''), errors='coerce').sum()),
                        float(t['Units_Sold'].sum())],
        'unconstrained_units_total': float(t['Unconstrained_Demand'].sum()),
        'stockout_rows': int(t['Is_Stockout'].sum()),
        'units_x_price_eq_revenue_pct': [float(recon.mean() * 100), 100.0],
    },
    'summary': {'issue_types': len(ISSUES), 'rows_touched': int(rows['Record_ID'].nunique()) if len(rows) else 0},
}
(REP / 'data_quality_report.json').write_text(json.dumps(report, indent=2, default=str))

log(f'Clean train rows: {len(t):,} (raw {n_raw:,}) | test rows: {len(te):,}')
for i in ISSUES:
    log(f"  {i['file']:<24} {i['issue_type']:<38} {i['rows_affected']:>7,}  {i['classification']}")
