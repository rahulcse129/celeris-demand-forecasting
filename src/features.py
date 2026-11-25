"""
Stage 3 - Feature library for the direct multi-horizon forecaster.

Design: every training row is (series i, forecast origin T, horizon h) with target month M = T + h.
All series features are computed ONLY from data observed up to T, so the backtest and the
production forecast (T = 2026-10, h = 1..19) use exactly the same code path - no leakage.

    python src/features.py      -> builds driver analyses (macro lead/lag, intermittency classes)
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from utils import int_to_ym, ym_to_int  # noqa: E402

warnings.filterwarnings('ignore')
CLEAN = ROOT / 'data' / 'cleaned'
FEAT = ROOT / 'data' / 'features'
FEAT.mkdir(parents=True, exist_ok=True)

M0 = ym_to_int(pd.Series(['2019-01']))[0]
LAST_OBS = ym_to_int(pd.Series(['2026-10']))[0] - M0       # column index of 2026-10
N_MONTHS = ym_to_int(pd.Series(['2028-05']))[0] - M0 + 1
H = 19
MACRO = ['ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct', 'Vehicle_Inventory_Days', 'Raw_Material_Cost_Index',
         'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct', 'Fuel_Price_Index', 'EV_Adoption_Index']
CATS = ['Product_Family', 'Region', 'Sales_Channel', 'Customer_Name', 'Powertrain_Application', 'Size_Class']


def macro_short(name):
    return name.split('_')[0] + '_' + name.split('_')[1][:4]


def col(ym):
    return int(ym_to_int(pd.Series([ym]))[0] - M0)


# ════════════════════════════════════════════════════════════════════
# PANEL
# ════════════════════════════════════════════════════════════════════
class Panel:
    """Dense S x months arrays + per-series metadata."""

    def __init__(self):
        tr = pd.read_parquet(CLEAN / 'train_clean.parquet')
        te = pd.read_csv(CLEAN / 'test_clean.csv', dtype={'EOL_Date': str, 'Replaces_SKU': str})
        self.sku = pd.read_csv(CLEAN / 'sku_master_clean.csv', dtype=str).set_index('SKU_ID')
        self.events = pd.read_csv(CLEAN / 'events_calendar_clean.csv', dtype=str)
        mac = pd.read_csv(CLEAN / 'macro_indicators_clean.csv')
        self.train, self.test = tr, te

        keys = pd.concat([tr[['SKU_ID', 'Region']], te[['SKU_ID', 'Region']]]).drop_duplicates()
        keys = keys.sort_values(['SKU_ID', 'Region']).reset_index(drop=True)
        self.keys = keys
        self.idx = {k: i for i, k in enumerate(zip(keys['SKU_ID'], keys['Region']))}
        S = len(keys)
        r = np.array([self.idx[k] for k in zip(tr['SKU_ID'], tr['Region'])])
        c = (ym_to_int(tr['Date']) - M0).values
        self.Y = np.full((S, N_MONTHS), np.nan)
        self.Y[r, c] = tr['Unconstrained_Demand'].values
        self.Ship = np.full((S, N_MONTHS), np.nan)
        self.Ship[r, c] = tr['Units_Sold'].values

        # metadata: latest customer / channel (test overrides train - accounts can switch)
        last = tr.sort_values('Date').groupby(['SKU_ID', 'Region']).tail(1).set_index(['SKU_ID', 'Region'])
        tmeta = te.groupby(['SKU_ID', 'Region'])[['Customer_Name', 'Sales_Channel']].first()
        meta = keys.set_index(['SKU_ID', 'Region'])
        for cname in ['Customer_Name', 'Sales_Channel']:
            meta[cname] = tmeta[cname].reindex(meta.index).fillna(last[cname].reindex(meta.index))
        meta = meta.reset_index()
        for cname in ['Product_Family', 'Business_Unit', 'Powertrain_Application', 'Size_Class',
                      'Launch_Date', 'EOL_Date', 'Replaces_SKU', 'Lifecycle_Status']:
            meta[cname] = meta['SKU_ID'].map(self.sku[cname])
        self.meta = meta
        self.launch = np.where(meta['Launch_Date'].notna(), ym_to_int(meta['Launch_Date'].fillna('1900-01')) - M0, -999)
        self.eol = np.where(meta['EOL_Date'].notna(), ym_to_int(meta['EOL_Date'].fillna('2100-01')) - M0, 999)

        mac = mac.set_index('Date').reindex(int_to_ym(np.arange(N_MONTHS) + M0))
        self.macro = mac[MACRO].interpolate(limit_direction='both').values   # N_MONTHS x 8

        self._event_masks()
        self._clean_series()

    # ------------------------------------------------------------------
    def _months(self, start, end):
        a = col(start[:7])
        b = col(end[:7]) if isinstance(end, str) and end else N_MONTHS - 1
        return max(a, 0), min(b, N_MONTHS - 1)

    def _event_masks(self):
        """Holiday intensity (share of month), disruption & customer-event masks (S x months)."""
        S = len(self.keys)
        reg = self.meta['Region'].values
        cust = self.meta['Customer_Name'].values
        chan = self.meta['Sales_Channel'].values
        fam = self.meta['Product_Family'].values
        self.holiday = np.zeros((S, N_MONTHS))
        self.no_train = np.zeros((S, N_MONTHS), bool)     # target months excluded from training
        self.no_feat = np.zeros((S, N_MONTHS), bool)      # history months hidden from level features
        self.shutdown = np.zeros((S, N_MONTHS))           # known customer shutdowns (incl. forecast window)
        for _, e in self.events.iterrows():
            st, en, et = e['Start_Date'], e['End_Date'], e['Event_Type']
            scope, val = e['Scope_Type'], str(e['Scope_Value'])
            if scope == 'Region':
                rows = reg == val
            elif scope == 'Channel':
                rows = chan == val
            elif scope == 'Product_Family':
                rows = fam == val
            elif scope == 'Customer':
                cn, rg = [s.strip() for s in val.split('|')]
                rows = (cust == cn) & (reg == rg)
            else:
                rows = np.ones(S, bool)
            if et == 'Holiday':
                if len(st) == 10:   # day-level -> share of each month covered
                    for d in pd.date_range(st, en):
                        c = col(d.strftime('%Y-%m'))
                        if 0 <= c < N_MONTHS:
                            self.holiday[rows, c] += 1 / d.days_in_month
                else:
                    a, b = self._months(st, en)
                    self.holiday[rows, a:b + 1] += 1
            elif et == 'Disruption':                     # COVID lockdowns
                a, b = self._months(st, en)
                self.no_train[rows, a:b + 2] = True
                self.no_feat[rows, a:b + 1] = True
            elif et == 'Customer Event' and 'strike' in e['Description'].lower() or \
                    et == 'Customer Event' and 'shutdown' in e['Description'].lower():
                a, b = self._months(st, en)
                self.shutdown[rows, a:b + 1] = 1
                if b <= LAST_OBS:
                    self.no_train[rows, a:b + 1] = True
                    self.no_feat[rows, a:b + 1] = True
            elif et == 'Logistics':                      # Red Sea: delay + catch-up
                a, b = self._months(st, en)
                self.no_train[rows, a - 1:b + 1] = True
            elif et == 'Supply Constraint':              # chip shortage: demand reconstructed, still uncertain
                a, b = self._months(st, en)
                self.no_train[rows, a:b + 1] = True
            elif et == 'Announcement' and 'price increase' in e['Description'].lower():
                a, _ = self._months(st, st)              # pre-buy before the 9% rise, payback after
                self.no_train[:, a:a + 4] = True
        self.holiday = np.clip(self.holiday, 0, 1)

    def _clean_series(self):
        """Yc: one-off spikes capped (model target). Yf: Yc with disruption months hidden (features)."""
        Y = self.Y[:, :LAST_OBS + 1]
        nz = np.where(Y > 0, Y, np.nan)
        med = pd.DataFrame(nz.T).rolling(13, center=True, min_periods=3).median().T.values
        med = np.where(np.isnan(med), np.nanmedian(nz, axis=1, keepdims=True), med)
        cap = 4.0 * med
        self.spike = (Y > cap) & (Y > 20)
        Yc = np.where(self.spike, cap, Y)
        self.Yc = np.full_like(self.Y, np.nan)
        self.Yc[:, :LAST_OBS + 1] = Yc
        self.Yf = np.where(self.no_feat, np.nan, self.Yc)


# ════════════════════════════════════════════════════════════════════
# FEATURES
# ════════════════════════════════════════════════════════════════════
def _nanmean(a, axis=1):
    with np.errstate(all='ignore'):
        return np.nanmean(a, axis=axis)


def _win(Y, T, k):
    return Y[:, max(T - k + 1, 0):T + 1]


def origin_state(P, T):
    """Per-series statistics known at origin T (independent of horizon)."""
    Yf = P.Yf
    S = Yf.shape[0]
    st = {}
    for k in (3, 6, 12, 24):
        st[f'r{k}'] = _nanmean(_win(Yf, T, k))
    prev = Yf[:, max(T - 23, 0):max(T - 11, 0)]
    st['r12_prev'] = _nanmean(prev) if prev.shape[1] else np.full(S, np.nan)
    w12 = _win(Yf, T, 12)
    with np.errstate(all='ignore'):
        st['nz12'] = np.nanmean(np.where(np.isnan(w12), np.nan, w12 > 0), axis=1)
        st['nzmean12'] = np.nanmean(np.where(w12 > 0, w12, np.nan), axis=1)
        st['std12'] = np.nanstd(w12, axis=1)
        st['nobs12'] = np.sum(~np.isnan(w12), axis=1)
    # EWMA level
    ew = np.full(S, np.nan)
    for t in range(max(T - 35, 0), T + 1):
        v = Yf[:, t]
        ew = np.where(np.isnan(ew), v, np.where(np.isnan(v), ew, 0.25 * v + 0.75 * ew))
    st['ewm'] = ew
    # Croston-SBA rate (intermittent-demand method) over last 36 months
    z = np.full(S, np.nan); p = np.full(S, np.nan); gap = np.ones(S)
    for t in range(max(T - 35, 0), T + 1):
        v = Yf[:, t]
        hit = v > 0
        z = np.where(hit, np.where(np.isnan(z), v, 0.15 * v + 0.85 * z), z)
        p = np.where(hit, np.where(np.isnan(p), gap, 0.15 * gap + 0.85 * p), p)
        gap = np.where(hit, 1, gap + 1)
    st['sba'] = (1 - 0.15 / 2) * z / p
    # recency / age
    obs = ~np.isnan(P.Yc[:, :T + 1])
    pos = P.Yc[:, :T + 1] > 0
    st['first_obs'] = np.where(obs.any(1), obs.argmax(1), T + 1)
    st['last_nz'] = np.where(pos.any(1), T - pos[:, ::-1].argmax(1), -999)
    scale = st['r12'].copy()
    for alt in ('r24', 'sba', 'ewm'):
        scale = np.where(np.isnan(scale), st[alt], scale)
    st['scale'] = np.maximum(np.nan_to_num(scale, nan=1.0), 1.0)

    # family x region aggregates (computed on rows observed in both windows to avoid mix effects)
    grp = P.meta['Product_Family'] + '|' + P.meta['Region']
    codes, uniq = pd.factorize(grp)
    st['fr_id'] = codes
    G = len(uniq)
    hist = np.nan_to_num(Yf[:, :T + 1])
    agg = np.zeros((G, T + 1))
    np.add.at(agg, codes, hist)
    seas = np.ones((G, 12))
    yrs = agg[:, max(0, T - 47):T + 1]
    base_m = max(0, T - 47)
    for mth in range(12):
        cols_m = [c for c in range(yrs.shape[1]) if (base_m + c) % 12 == mth]
        seas[:, mth] = yrs[:, cols_m].mean(1) / np.maximum(yrs.mean(1), 1e-6)
    st['fr_seas'] = seas
    st['fr_trend'] = agg[:, T - 11:T + 1].sum(1) / np.maximum(agg[:, T - 23:T - 11].sum(1), 1)
    st['fr_recent'] = agg[:, T - 5:T + 1].sum(1) / np.maximum(agg[:, T - 17:T - 11].sum(1), 1)
    return st


def horizon_features(P, T, h, st, rows=None):
    """Feature frame for target month M = T + h (all series, or `rows`)."""
    Yf = P.Yf
    S = Yf.shape[0]
    rows = np.arange(S) if rows is None else rows
    M = T + h
    s = st['scale'][rows]
    f = {'h': np.full(len(rows), h), 'target_month': np.full(len(rows), (M % 12) + 1),
         'origin_month': np.full(len(rows), (T % 12) + 1)}
    for k in ('r3', 'r6', 'r24', 'ewm', 'sba', 'nzmean12', 'std12', 'r12_prev'):
        f[f'n_{k}'] = st[k][rows] / s
    f['nz12'] = st['nz12'][rows]
    f['nobs12'] = st['nobs12'][rows]
    f['log_scale'] = np.log1p(s)
    f['trend_12_12'] = st['r12'][rows] / np.maximum(st['r12_prev'][rows], 1)
    f['last'] = Yf[rows, T] / s
    f['last2'] = Yf[rows, T - 1] / s

    # order-cycle phase features: what happened in the same phase of a P-month cycle
    for per in (2, 3, 4, 6):
        ts = [t for t in range(T - 11, T + 1) if (M - t) % per == 0]
        f[f'pm{per}'] = _nanmean(Yf[np.ix_(rows, ts)]) / s
        with np.errstate(all='ignore'):
            f[f'pz{per}'] = np.nanmean(np.where(np.isnan(Yf[np.ix_(rows, ts)]), np.nan, Yf[np.ix_(rows, ts)] == 0), axis=1)
        f[f'sn{per}'] = Yf[rows, M - per * int(np.ceil(h / per))] / s
    since = M - st['last_nz'][rows]
    f['since_last_order'] = np.where(st['last_nz'][rows] < 0, 99, since)
    for per in (2, 3):
        f[f'on_phase{per}'] = (since % per == 0).astype(float)

    # same calendar month in previous years (seasonality / recurring shutdowns)
    yrs = [M - 12 * k for k in range(1, 5) if M - 12 * k <= T and M - 12 * k >= 0]
    sm = Yf[np.ix_(rows, yrs)]
    f['pm12'] = _nanmean(sm) / s
    with np.errstate(all='ignore'):
        f['pz12'] = np.nanmean(np.where(np.isnan(sm), np.nan, sm == 0), axis=1)
    f['sn12'] = Yf[rows, M - 12 * int(np.ceil(h / 12))] / s
    allh = Yf[rows, :T + 1]
    cal = allh[:, [t for t in range(T + 1) if t % 12 == M % 12]]
    with np.errstate(all='ignore'):
        f['seas_idx'] = _nanmean(cal) / np.maximum(_nanmean(allh), 1e-6)
    # family x region level: smoother seasonality and the group's own momentum
    fr = st['fr_id'][rows]
    f['fr_seas'] = st['fr_seas'][fr, M % 12]
    f['fr_trend'] = st['fr_trend'][fr]
    f['fr_recent'] = st['fr_recent'][fr]

    # events at target month
    f['holiday'] = P.holiday[rows, M]
    f['holiday_prev'] = P.holiday[rows, M - 12]
    f['shutdown'] = P.shutdown[rows, M]

    # lifecycle
    f['age'] = np.minimum(M - st['first_obs'][rows], 120)
    f['since_launch'] = np.clip(M - P.launch[rows], -24, 120)
    f['to_eol'] = np.clip(P.eol[rows] - M, -12, 60)

    # macro drivers (known for the whole window). The target is level-normalised, so the driver is how
    # far the indicator (1-month lead) sits from its average over the window the level was measured on.
    # Raw macro LEVELS were tested and dropped: they act as time proxies and extrapolate badly.
    for j, name in enumerate(MACRO):
        f[f'w_{macro_short(name)}'] = P.macro[M - 1, j] - P.macro[T - 12:T, j].mean()
    ev_sign = P.meta['Powertrain_Application'].map({'EV': 1.0, 'ICE': -1.0, 'Universal': 0.0}).values[rows]
    f['ev_x_powertrain'] = ev_sign * (P.macro[M, 7] - P.macro[T, 7])

    df = pd.DataFrame(f)
    for c in CATS:
        df[c] = P.meta[c].values[rows]
    df['_row'] = rows
    df['_T'] = T
    df['_M'] = M
    df['_scale'] = s
    return df


def build_training(P, origins, max_target, rng=None, sample=1.0):
    """Stack (series, origin, horizon) rows with known targets <= max_target."""
    frames = []
    for T in origins:
        st = origin_state(P, T)
        active = (st['nobs12'] > 0)
        for h in range(1, H + 1):
            M = T + h
            if M > max_target:
                break
            y = P.Yc[:, M]
            ok = active & ~np.isnan(y) & ~P.no_train[:, M]
            rows = np.where(ok)[0]
            if sample < 1.0 and rng is not None:
                rows = rows[rng.random(len(rows)) < sample]
            if not len(rows):
                continue
            df = horizon_features(P, T, h, st, rows)
            df['y'] = y[rows]
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in CATS:
        out[c] = out[c].astype('category')
    return out


def feature_columns(df):
    return [c for c in df.columns if not c.startswith('_') and c != 'y']


# ════════════════════════════════════════════════════════════════════
# DRIVER ANALYSIS (run as script)
# ════════════════════════════════════════════════════════════════════
def driver_analysis(P):
    """Which indicator leads which family, at what lag, and is it stable over time?"""
    fam = P.meta['Product_Family'].values
    dates = int_to_ym(np.arange(LAST_OBS + 1) + M0)
    rows, stab = [], []
    for fname in sorted(set(fam)):
        y = np.nansum(P.Yf[fam == fname, :LAST_OBS + 1], axis=0)
        yoy = pd.Series(y).pct_change(12)
        for j, mname in enumerate(MACRO):
            x = pd.Series(P.macro[:LAST_OBS + 1, j])
            xch = x.diff(12) if mname != 'Auto_Production_YoY_Pct' else x
            for lag in range(0, 13):
                cc = yoy.corr(xch.shift(lag))
                rows.append(dict(Product_Family=fname, Indicator=mname, Lag=lag, Corr=round(float(cc), 3)))
            # stability: best lag in 2020-2022 vs 2023-2026
            for period, (a, b) in {'2020-2022': ('2020-01', '2022-12'), '2023-2026': ('2023-01', '2026-10')}.items():
                m = (dates >= a) & (dates <= b)
                cs = [yoy[m].corr(xch.shift(l)[m]) for l in range(13)]
                k = int(np.nanargmax(np.abs(cs)))
                stab.append(dict(Product_Family=fname, Indicator=mname, Period=period, Best_Lag=k, Corr=round(float(cs[k]), 3)))
    ll = pd.DataFrame(rows)
    best = ll.loc[ll.groupby(['Product_Family', 'Indicator'])['Corr'].apply(lambda s: s.abs().idxmax())]
    ll.to_csv(FEAT / 'macro_lead_lag.csv', index=False)
    best.to_csv(FEAT / 'macro_best_lag.csv', index=False)
    pd.DataFrame(stab).to_csv(FEAT / 'macro_lag_stability.csv', index=False)

    # EV transition: demand index by powertrain
    pt = P.meta['Powertrain_Application'].values
    ev = []
    for p in ('EV', 'ICE', 'Universal'):
        y = np.nansum(P.Yf[pt == p, :LAST_OBS + 1], axis=0)
        for i, d in enumerate(dates):
            ev.append(dict(Date=d, Powertrain=p, Units=float(y[i]), EV_Adoption_Index=float(P.macro[i, 7])))
    pd.DataFrame(ev).to_csv(FEAT / 'ev_transition.csv', index=False)
    return best


def intermittency(P):
    """Syntetos-Boylan class + detected order cycle per series (last 24 months)."""
    out = []
    for i, (sku, reg) in enumerate(zip(P.meta['SKU_ID'], P.meta['Region'])):
        y = P.Yf[i, LAST_OBS - 23:LAST_OBS + 1]
        y = y[~np.isnan(y)]
        nzi = np.where(y > 0)[0]
        if len(nzi) < 2:
            out.append(dict(SKU_ID=sku, Region=reg, ADI=np.nan, CV2=np.nan, Demand_Class='Insufficient history',
                            Order_Cycle_Months=np.nan, Zero_Pct=np.nan))
            continue
        adi = float(np.diff(nzi).mean())
        nzv = y[nzi]
        cv2 = float((nzv.std() / nzv.mean()) ** 2)
        cls = ('Smooth' if cv2 < 0.49 else 'Erratic') if adi < 1.32 else ('Intermittent' if cv2 < 0.49 else 'Lumpy')
        cyc = int(np.median(np.diff(nzi)))
        # cycle change: compare first and second 12 months
        y_old = P.Yf[i, LAST_OBS - 47:LAST_OBS - 23]
        nz_old = np.where(y_old > 0)[0]
        cyc_old = int(np.median(np.diff(nz_old))) if len(nz_old) > 1 else np.nan
        out.append(dict(SKU_ID=sku, Region=reg, ADI=round(adi, 2), CV2=round(cv2, 3), Demand_Class=cls,
                        Order_Cycle_Months=cyc, Previous_Cycle_Months=cyc_old,
                        Zero_Pct=round(100 * (y == 0).mean(), 1)))
    return pd.DataFrame(out)


if __name__ == '__main__':
    P = Panel()
    print(f'Panel: {P.Y.shape[0]:,} series x {N_MONTHS} months; spikes capped: {int(P.spike.sum()):,}')
    best = driver_analysis(P)
    print(best.sort_values('Corr', key=abs, ascending=False).head(12).to_string(index=False))
    ic = intermittency(P)
    ic.to_csv(FEAT / 'intermittency_classes.csv', index=False)
    print(ic['Demand_Class'].value_counts().to_string())
    print('Order cycles (months):', ic['Order_Cycle_Months'].value_counts().to_dict())
    print('Series whose cycle changed vs prior 24m:',
          int((ic['Order_Cycle_Months'] != ic['Previous_Cycle_Months']).sum()))
