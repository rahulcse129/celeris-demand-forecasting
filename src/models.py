"""
Stage 4 - Rolling-origin backtest (19-month horizon) + final forecast.

    python src/models.py            # full run (backtest + final forecast)
    python src/models.py --fast     # one backtest fold (development)

Model: one global LightGBM, direct multi-horizon (h = 1..19), trained on scale-normalised
unconstrained demand with weight = series scale, objective L1  ->  equivalent to minimising WAPE.
P10/P90 from quantile LightGBMs on the same features. Lifecycle (EOL, launches via Replaces_SKU) and
known events (customer shutdown) applied after the model. Bottom-up => coherent at every level.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import (H, LAST_OBS, M0, Panel, build_training, col, feature_columns,  # noqa: E402
                      horizon_features, origin_state)
from utils import bias, int_to_ym, wape  # noqa: E402

warnings.filterwarnings('ignore')
FC = ROOT / 'data' / 'forecasts'
FEAT = ROOT / 'data' / 'features'
REP = ROOT / 'reports'
FC.mkdir(parents=True, exist_ok=True)

FAST = '--fast' in sys.argv
SEED = 42
FIRST_ORIGIN = col('2020-09')
FOLDS = [col('2023-03'), col('2024-03'), col('2025-03')]      # each forecasts 19 months ahead
PARAMS = dict(objective='l1', learning_rate=0.05, num_leaves=127, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, verbose=-1, num_threads=0, seed=SEED)
ROUNDS = 900
QROUNDS = 500


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fit(df, feats, objective='l1', alpha=None, rounds=ROUNDS):
    p = dict(PARAMS, objective=objective)
    if alpha is not None:
        p['alpha'] = alpha
    y = np.clip(df['y'].values / df['_scale'].values, 0, 12)
    ds = lgb.Dataset(df[feats], y, weight=df['_scale'].values, categorical_feature=CATS_IN(feats), free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=rounds)


def CATS_IN(feats):
    from features import CATS
    return [c for c in CATS if c in feats]


def train_origins(T_cut):
    """Origins whose first horizon is still observed; every 2nd month keeps the set compact."""
    return list(range(FIRST_ORIGIN, T_cut, 2))


def forecast_rows(P, T, rows_by_h=None):
    st = origin_state(P, T)
    frames = []
    for h in range(1, H + 1):
        rows = None if rows_by_h is None else rows_by_h.get(h)
        if rows is not None and not len(rows):
            continue
        frames.append(horizon_features(P, T, h, st, rows))
    df = pd.concat(frames, ignore_index=True)
    from features import CATS
    for c in CATS:
        df[c] = df[c].astype('category')
    return df, st


def align_cats(df, ref):
    from features import CATS
    for c in CATS:
        df[c] = pd.Categorical(df[c].astype(str), categories=ref[c].cat.categories)
    return df


# ════════════════════════════════════════════════════════════════════
# Domain post-processing
# ════════════════════════════════════════════════════════════════════
def shutdown_factor(P):
    """Analog for a customer plant shutdown: Nexon Motor Group Europe strike (2023-10..11)."""
    m = P.shutdown[:, :LAST_OBS + 1] > 0
    rows = np.where(m.any(1))[0]
    a, b = np.where(m[rows[0]])[0][[0, -1]]
    during = np.nansum(P.Y[rows, a:b + 1])
    normal = np.nansum(P.Y[rows, a - 3:a]) + np.nansum(P.Y[rows, b + 1:b + 4])
    return float(during / (b - a + 1) / (normal / 6))


def launch_ramp(P):
    """Successor demand per region relative to predecessor per-region level, by months since launch."""
    meta, out = P.meta, []
    sku = P.sku
    for s, r in sku[sku['Replaces_SKU'].notna()].iterrows():
        L = col(r['Launch_Date'])
        if L + 12 > LAST_OBS:
            continue
        pred_rows = np.where(meta['SKU_ID'] == r['Replaces_SKU'])[0]
        succ_rows = np.where(meta['SKU_ID'] == s)[0]
        lvl = np.nanmean(P.Y[pred_rows, L - 12:L])
        for k in range(0, 19):
            if L + k <= LAST_OBS:
                out.append(dict(SKU=s, k=k, ratio=np.nanmean(P.Y[succ_rows, L + k]) / lvl))
    d = pd.DataFrame(out)
    ramp = d.groupby('k')['ratio'].median()
    ramp = ramp.rolling(3, center=True, min_periods=1).mean()
    return ramp.reindex(range(0, 25)).ffill(), d


def postprocess(P, df, pred, q10, q90, factor, ramp, st, T):
    """Apply lifecycle rules, launches and known events. Returns arrays + reason codes."""
    rows, M = df['_row'].values, df['_M'].values
    pred, q10, q90 = np.nan_to_num(pred), np.nan_to_num(q10), np.nan_to_num(q90)
    reason = np.array([''] * len(df), dtype=object)
    # 1) launches with no history: inherit predecessor level x empirical ramp
    new = (st['nobs12'][rows] == 0) & P.meta['Replaces_SKU'].notna().values[rows]
    for i in np.where(new)[0]:
        rr = rows[i]
        pred_sku = P.meta['Replaces_SKU'].iat[rr]
        prow = np.where(P.meta['SKU_ID'].values == pred_sku)[0]
        same_reg = [p for p in prow if P.meta['Region'].iat[p] == P.meta['Region'].iat[rr]]
        base = np.nanmean(P.Yf[same_reg if same_reg else prow, T - 11:T + 1])     # only data known at origin
        if np.isnan(base):
            base = np.nanmean(P.Yf[prow, :T + 1])
        k = int(M[i] - P.launch[rr])
        pred[i] = np.nan_to_num(base) * ramp.get(k, ramp.iloc[-1]) if k >= 0 else 0.0
        q10[i], q90[i] = 0.5 * pred[i], 1.7 * pred[i]
        reason[i] = f'launch: {pred_sku} level x ramp(k={k})'
    # 2) customer shutdown in the forecast window (Orion Driveline NA retooling 2027-07..08)
    sd = P.shutdown[rows, M] > 0
    pred[sd] *= factor; q10[sd] *= factor; q90[sd] *= factor
    reason[sd] = f'customer shutdown x{factor:.2f} (Nexon strike analog)'
    # 3) lifecycle: nothing after EOL, nothing before launch (business rule 5)
    dead = (M > P.eol[rows]) | (M < P.launch[rows])
    pred[dead] = q10[dead] = q90[dead] = 0
    reason[dead] = 'after EOL / before launch -> 0'
    q10 = np.minimum(q10, pred)
    q90 = np.maximum(q90, pred)
    return pred, q10, q90, reason


# ════════════════════════════════════════════════════════════════════
# Baselines (same origin, same horizon)
# ════════════════════════════════════════════════════════════════════
def baselines(P, df, st):
    rows, M, h = df['_row'].values, df['_M'].values, df['h'].values
    sn = P.Yf[rows, M - 12 * np.ceil(h / 12).astype(int)]
    fallback = st['r12'][rows]
    return {
        'Seasonal_Naive': np.nan_to_num(np.where(np.isnan(sn), fallback, sn)),
        'Moving_Avg_12m': np.nan_to_num(fallback),
        'Croston_SBA': np.nan_to_num(np.where(np.isnan(st['sba'][rows]), fallback, st['sba'][rows])),
    }


# ════════════════════════════════════════════════════════════════════
# 1. ROLLING-ORIGIN BACKTEST
# ════════════════════════════════════════════════════════════════════
def backtest(P, factor, ramp):
    rng = np.random.default_rng(SEED)
    out = []
    folds = FOLDS[-1:] if FAST else FOLDS
    for T in folds:
        t0 = time.time()
        tr = build_training(P, train_origins(T), max_target=T, rng=rng, sample=0.7)
        feats = feature_columns(tr)
        model = fit(tr, feats)
        m10 = fit(tr, feats, 'quantile', 0.1, QROUNDS)
        m90 = fit(tr, feats, 'quantile', 0.9, QROUNDS)
        # evaluate on every observed (series, month) in the 19 months after T
        rows_by_h = {h: np.where(~np.isnan(P.Y[:, T + h]))[0] for h in range(1, H + 1) if T + h <= LAST_OBS}
        te, st = forecast_rows(P, T, rows_by_h)
        te = align_cats(te, tr)
        s = te['_scale'].values
        pred = np.clip(model.predict(te[feats]), 0, None) * s
        q10 = np.clip(m10.predict(te[feats]), 0, None) * s
        q90 = np.clip(m90.predict(te[feats]), 0, None) * s
        pred, q10, q90, _ = postprocess(P, te, pred, q10, q90, factor, ramp, st, T)
        res = te[['_row', '_M', 'h']].copy()
        res['Origin'] = int_to_ym(np.full(len(te), T) + M0)
        res['Date'] = int_to_ym(te['_M'].values + M0)
        res['Actual'] = P.Y[te['_row'], te['_M']]
        res['Actual_Shipped'] = P.Ship[te['_row'], te['_M']]
        for k, v in baselines(P, te, st).items():
            res[k] = v
        res['LightGBM_P50'] = pred
        res['P10'], res['P90'] = q10, q90
        out.append(res)
        log(f"  fold {res['Origin'].iat[0]}: train {len(tr):,} rows | WAPE LGB {wape(res.Actual, np.round(pred)):.4f} "
            f"| SNaive {wape(res.Actual, res.Seasonal_Naive):.4f} | MA12 {wape(res.Actual, res.Moving_Avg_12m):.4f} "
            f"| SBA {wape(res.Actual, res.Croston_SBA):.4f} | {time.time() - t0:.0f}s")
    bt = pd.concat(out, ignore_index=True)
    meta = P.meta.iloc[bt['_row']].reset_index(drop=True)
    for c in ['SKU_ID', 'Region', 'Product_Family', 'Sales_Channel', 'Customer_Name']:
        bt[c] = meta[c].values
    bt = bt.rename(columns={'h': 'Horizon'}).drop(columns=['_row', '_M'])
    bt['LightGBM_P50'] = np.round(bt['LightGBM_P50'])
    return bt, feats


def metrics_table(bt):
    models = ['Seasonal_Naive', 'Moving_Avg_12m', 'Croston_SBA', 'LightGBM_P50']
    overall = {m: dict(wape=round(wape(bt.Actual, bt[m]), 4), bias=round(bias(bt.Actual, bt[m]), 4),
                       mae=round(float((bt.Actual - bt[m]).abs().mean()), 2)) for m in models}
    per_fold = {o: {m: round(wape(g.Actual, g[m]), 4) for m in models} for o, g in bt.groupby('Origin')}

    def by(key):
        return {str(k): dict(wape=round(wape(g.Actual, g.LightGBM_P50), 4), bias=round(bias(g.Actual, g.LightGBM_P50), 4),
                             wape_snaive=round(wape(g.Actual, g.Seasonal_Naive), 4), actual_units=float(g.Actual.sum()))
                for k, g in bt.groupby(key)}
    cover = float(((bt.Actual >= bt.P10) & (bt.Actual <= bt.P90)).mean())
    agg = {}
    for lvl in ['Product_Family', 'Region', 'Sales_Channel']:
        g = bt.groupby(['Origin', 'Date', lvl])[['Actual', 'LightGBM_P50']].sum()
        agg[lvl] = round(wape(g.Actual, g.LightGBM_P50), 4)
    g = bt.groupby(['Origin', 'Date'])[['Actual', 'LightGBM_P50']].sum()
    agg['Total'] = round(wape(g.Actual, g.LightGBM_P50), 4)
    return dict(overall=overall, per_fold=per_fold, by_horizon=by('Horizon'), by_family=by('Product_Family'),
                by_region=by('Region'), by_channel=by('Sales_Channel'), aggregate_level_wape=agg,
                interval_P10_P90_coverage=round(cover, 4),
                target='Unconstrained_Demand (stockout-corrected)',
                folds=sorted(bt['Origin'].unique().tolist()), horizon_months=H)


# ════════════════════════════════════════════════════════════════════
# 2. FINAL FORECAST (origin 2026-10)
# ════════════════════════════════════════════════════════════════════
def final_forecast(P, factor, ramp):
    rng = np.random.default_rng(SEED + 1)
    T = LAST_OBS
    tr = build_training(P, train_origins(T), max_target=T, rng=rng, sample=0.7)
    feats = feature_columns(tr)
    log(f'  final training set: {len(tr):,} rows x {len(feats)} features')
    model = fit(tr, feats)
    m10 = fit(tr, feats, 'quantile', 0.1, QROUNDS)
    m90 = fit(tr, feats, 'quantile', 0.9, QROUNDS)

    test = P.test.copy()
    test['_row'] = [P.idx[k] for k in zip(test['SKU_ID'], test['Region'])]
    test['_M'] = (pd.Series(test['Date']).str[:4].astype(int) * 12 + test['Date'].str[5:7].astype(int) - 1 - M0).values
    rows_by_h = {h: test.loc[test['_M'] == T + h, '_row'].values for h in range(1, H + 1)}
    te, st = forecast_rows(P, T, rows_by_h)
    te = align_cats(te, tr)
    s = te['_scale'].values
    raw = np.clip(model.predict(te[feats]), 0, None)
    pred = raw * s
    q10 = np.clip(m10.predict(te[feats]), 0, None) * s
    q90 = np.clip(m90.predict(te[feats]), 0, None) * s
    pred, q10, q90, reason = postprocess(P, te, pred.copy(), q10, q90, factor, ramp, st, T)
    te['P50'], te['P10'], te['P90'], te['Adjustment'] = pred, q10, q90, reason
    te['Adj_Factor'] = np.where(raw * s > 0, pred / np.maximum(raw * s, 1e-9), 0.0)
    te['Raw_Model'] = raw
    te['SKU_ID'] = P.meta['SKU_ID'].values[te['_row']]
    te['Date'] = int_to_ym(te['_M'].values + M0)

    fc = test.merge(te[['_row', '_M', 'P50', 'P10', 'P90', 'Adjustment']], on=['_row', '_M'], how='left')
    assert fc['P50'].notna().all() and len(fc) == len(P.test)
    fc['Units_Sold_Forecast'] = np.round(fc['P50']).astype(int)
    fc['Forecast_P10'] = np.floor(fc['P10']).astype(int)
    fc['Forecast_P90'] = np.ceil(fc['P90']).astype(int)
    fc['Forecast_P50'] = fc['Units_Sold_Forecast']
    # revenue: list price at the end of history x next Jan increases from the events calendar
    price = P.train[P.train['Date'] >= '2026-01'].groupby('SKU_ID')['Unit_Price_USD'].median()
    pinc = {'2027': 1.045, '2028': 1.045 * 1.04}
    pr = fc['SKU_ID'].map(price)
    pr = pr.fillna(fc['SKU_ID'].map(P.sku['Replaces_SKU']).map(price))
    fc['Unit_Price_USD'] = (pr * fc['Date'].str[:4].map(pinc).fillna(1.0)).round(2)
    fc['Revenue_Forecast_USD'] = (fc['Units_Sold_Forecast'] * fc['Unit_Price_USD']).round(2)
    imp = pd.DataFrame({'feature': feats, 'importance': model.feature_importance('gain')}).sort_values('importance', ascending=False)
    return fc, te, model, feats, imp


def hierarchy(fc):
    """Bottom-up aggregates for every level; integer SKU forecasts sum exactly to each node."""
    out = []
    levels = {'Total': [], 'Product_Family': ['Product_Family'], 'Region': ['Region'],
              'Sales_Channel': ['Sales_Channel'], 'Customer_Name': ['Customer_Name'],
              'Family_x_Region': ['Product_Family', 'Region']}
    for name, keys in levels.items():
        g = fc.groupby(['Date'] + keys)[['Units_Sold_Forecast', 'Forecast_P10', 'Forecast_P90']].sum().reset_index()
        g['Level'] = name
        g['Node'] = g[keys].astype(str).agg(' | '.join, axis=1) if keys else 'All'
        g = g.rename(columns={'Units_Sold_Forecast': 'Bottom_Up_Sum'})
        g['Reconciled_Sum'] = g['Bottom_Up_Sum']
        out.append(g[['Level', 'Node', 'Date', 'Bottom_Up_Sum', 'Reconciled_Sum', 'Forecast_P10', 'Forecast_P90']])
    return pd.concat(out, ignore_index=True)


def main():
    t0 = time.time()
    log('Building panel ...')
    P = Panel()
    factor = shutdown_factor(P)
    ramp, ramp_obs = launch_ramp(P)
    log(f'  {len(P.keys):,} series | shutdown factor {factor:.2f} | launch ramp k0..6: '
        f'{np.round(ramp.iloc[:7].values, 2).tolist()}')

    log('Rolling-origin backtest (19-month horizon) ...')
    bt, _ = backtest(P, factor, ramp)
    met = metrics_table(bt)
    for m, v in met['overall'].items():
        log(f'  {m:<16} WAPE {v["wape"]:.4f}  bias {v["bias"]:+.4f}')
    log(f"  P10-P90 coverage {met['interval_P10_P90_coverage']:.3f} | aggregate WAPE {met['aggregate_level_wape']}")
    if FAST:
        return

    log('Final model (origin 2026-10) ...')
    fc, te, model, feats, imp = final_forecast(P, factor, ramp)

    sub = fc[['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast']]
    raw_test = pd.read_csv(ROOT / 'dataset' / 'student_data' / 'test.csv')
    assert len(sub) == len(raw_test) and sub['Units_Sold_Forecast'].ge(0).all() and sub.notna().all().all()
    sub.to_csv(ROOT / 'submission.csv', index=False)

    cols = ['Date', 'SKU_ID', 'Region', 'Product_Family', 'Business_Unit', 'Sales_Channel', 'Customer_Name',
            'Units_Sold_Forecast', 'Forecast_P10', 'Forecast_P50', 'Forecast_P90', 'Unit_Price_USD',
            'Revenue_Forecast_USD', 'Adjustment']
    fc[cols].to_csv(FC / 'forecast_detail.csv', index=False)
    hierarchy(fc).to_csv(FC / 'hierarchical_reconciliation.csv', index=False)
    bt.to_parquet(FC / 'backtest_results.parquet', index=False)
    imp.to_csv(FC / 'feature_importance.csv', index=False)
    model.save_model(str(FC / 'lgb_model.txt'))
    (FC / 'model_features.json').write_text(json.dumps(feats))
    te.drop(columns=['P50', 'P10', 'P90']).to_parquet(FEAT / 'test_features.parquet', index=False)
    ramp_obs.to_csv(FC / 'launch_ramp_analogs.csv', index=False)

    met['test_forecast_summary'] = dict(rows=len(fc), total_units=int(fc['Units_Sold_Forecast'].sum()),
                                        total_revenue_usd=float(fc['Revenue_Forecast_USD'].sum()),
                                        p10_total=int(fc['Forecast_P10'].sum()), p90_total=int(fc['Forecast_P90'].sum()),
                                        adjusted_rows=fc['Adjustment'].fillna('').ne('').groupby(
                                            fc['Adjustment'].fillna('').str.split(':').str[0]).sum().to_dict())
    met['domain_parameters'] = dict(shutdown_factor=round(factor, 3),
                                    launch_ramp=[round(float(x), 3) for x in ramp.values])
    met['model'] = dict(type='LightGBM global, direct multi-horizon', objective='L1 on scale-normalised target, weight=scale (WAPE-aligned)',
                        features=len(feats), rounds=ROUNDS, train_rows=None, runtime_sec=round(time.time() - t0))
    (REP / 'model_metrics.json').write_text(json.dumps(met, indent=2))
    hist = P.train.groupby('Date')['Unconstrained_Demand'].sum()
    log(f"Submission written: {len(sub):,} rows | 19-month total {met['test_forecast_summary']['total_units']:,} units "
        f"(last 19 months of history: {int(hist.iloc[-19:].sum()):,}) | {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
