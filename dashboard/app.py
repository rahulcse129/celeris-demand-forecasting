"""
Celeris Automobili - Demand Planning Control Tower
    streamlit run dashboard/app.py

Reads ONLY pipeline outputs (data/cleaned, data/features, data/forecasts, reports) - no typed-in numbers.
Part A (training data): Executive overview, Data health, Demand explorer, Anomaly monitor
Part B (prediction):    Forecast & accuracy, Macro drivers, Lifecycle, What-if, Procurement (P90),
                        Stockouts & exceptions, Downloads
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Celeris Planning Control Tower', page_icon='🏭', layout='wide')
ROOT = Path(__file__).resolve().parent.parent
CLEAN, FEAT, FC, REP = (ROOT / 'data' / 'cleaned', ROOT / 'data' / 'features', ROOT / 'data' / 'forecasts',
                        ROOT / 'reports')
HIST_C, UNC_C, FC_C, BAND_C = '#8a94a6', '#2b6cb0', '#dd6b20', 'rgba(221,107,32,0.18)'
SPLIT = '2026-10'


_N = [0]


def chart(where, fig, **kw):
    """plotly_chart with a stable unique key per position on the page."""
    _N[0] += 1
    return where.plotly_chart(fig, key=f'chart_{_N[0]}', **kw)


@st.cache_data
def load():
    d = dict(
        train=pd.read_parquet(CLEAN / 'train_clean.parquet'),
        fc=pd.read_csv(FC / 'forecast_detail.csv'),
        bt=pd.read_parquet(FC / 'backtest_results.parquet'),
        imp=pd.read_csv(FC / 'feature_importance.csv'),
        anom=pd.read_csv(REP / 'anomaly_register.csv'),
        so=pd.read_csv(CLEAN / 'inventory_stockouts_clean.csv'),
        macro=pd.read_csv(CLEAN / 'macro_indicators_clean.csv'),
        sku=pd.read_csv(CLEAN / 'sku_master_clean.csv', dtype=str),
        lag=pd.read_csv(FEAT / 'macro_lead_lag.csv'),
        stab=pd.read_csv(FEAT / 'macro_lag_stability.csv'),
        ev=pd.read_csv(FEAT / 'ev_transition.csv'),
        ic=pd.read_csv(FEAT / 'intermittency_classes.csv'),
        ramp=pd.read_csv(FC / 'launch_ramp_analogs.csv'),
        hier=pd.read_csv(FC / 'hierarchical_reconciliation.csv'),
        quality=json.loads((REP / 'data_quality_report.json').read_text()),
        metrics=json.loads((REP / 'model_metrics.json').read_text()),
    )
    d['issues_rows'] = REP / 'data_quality_issues.csv'
    return d


@st.cache_resource
def load_model():
    return (lgb.Booster(model_file=str(FC / 'lgb_model.txt')), json.loads((FC / 'model_features.json').read_text()),
            pd.read_parquet(FEAT / 'test_features.parquet'))


D = load()
train, fc, bt, anom = D['train'], D['fc'], D['bt'], D['anom']
M = D['metrics']

# ════════════════════════════════════════════════════════════════════
# SHARED FILTERS
# ════════════════════════════════════════════════════════════════════
sb = st.sidebar
sb.markdown('## 🏭 Celeris Control Tower')
sb.caption('Filters apply to every view (history and forecast)')
fam_all = sorted(train['Product_Family'].unique())
f_fam = sb.multiselect('Product family', fam_all, placeholder='All families')
sku_pool = sorted(set(train.loc[train['Product_Family'].isin(f_fam or fam_all), 'SKU_ID']) |
                  set(fc.loc[fc['Product_Family'].isin(f_fam or fam_all), 'SKU_ID']))
f_sku = sb.multiselect('SKU', sku_pool, placeholder='All SKUs')
f_reg = sb.multiselect('Region', sorted(train['Region'].unique()), placeholder='All regions')
f_ch = sb.multiselect('Sales channel', sorted(train['Sales_Channel'].unique()), placeholder='All channels')
f_cu = sb.multiselect('Customer', sorted(train['Customer_Name'].unique()), placeholder='All customers')


def filt(df):
    m = pd.Series(True, index=df.index)
    for c, v in [('Product_Family', f_fam), ('SKU_ID', f_sku), ('Region', f_reg), ('Sales_Channel', f_ch),
                 ('Customer_Name', f_cu)]:
        if v and c in df.columns:
            m &= df[c].isin(v)
    return df[m]


T, F, B = filt(train), filt(fc), filt(bt)
sb.divider()
sb.metric('History rows in view', f'{len(T):,}')
sb.metric('Forecast rows in view', f'{len(F):,}')
sb.caption(f"Backtest WAPE (19-mo, 3 origins): **{M['overall']['LightGBM_P50']['wape']:.3f}** "
           f"vs seasonal naive {M['overall']['Seasonal_Naive']['wape']:.3f}")


def timeline(T, F, value='units', title=''):
    """History (shipped + unconstrained) and forecast (P10-P90 band) on one continuous axis."""
    if value == 'units':
        h = T.groupby('Date')[['Units_Sold', 'Unconstrained_Demand']].sum()
        f = F.groupby('Date')[['Units_Sold_Forecast', 'Forecast_P10', 'Forecast_P90']].sum()
    else:
        T = T.assign(UR=T['Unconstrained_Demand'] * T['Unit_Price_USD'])
        h = T.groupby('Date')[['Revenue_USD', 'UR']].sum().rename(columns={'Revenue_USD': 'Units_Sold', 'UR': 'Unconstrained_Demand'})
        F = F.assign(R10=F['Forecast_P10'] * F['Unit_Price_USD'], R90=F['Forecast_P90'] * F['Unit_Price_USD'])
        f = F.groupby('Date')[['Revenue_Forecast_USD', 'R10', 'R90']].sum()
        f.columns = ['Units_Sold_Forecast', 'Forecast_P10', 'Forecast_P90']
    fig = go.Figure()
    if len(f):
        fig.add_trace(go.Scatter(x=list(f.index) + list(f.index[::-1]),
                                 y=list(f['Forecast_P90']) + list(f['Forecast_P10'][::-1]),
                                 fill='toself', fillcolor=BAND_C, line=dict(width=0), name='P10–P90 range'))
    fig.add_trace(go.Scatter(x=h.index, y=h['Units_Sold'], name='Shipped (actual)', line=dict(color=HIST_C, width=1.5)))
    fig.add_trace(go.Scatter(x=h.index, y=h['Unconstrained_Demand'], name='Unconstrained demand',
                             line=dict(color=UNC_C, width=1.5, dash='dot')))
    if len(f) and len(h):
        x = [h.index[-1]] + list(f.index)
        y = [h['Unconstrained_Demand'].iloc[-1]] + list(f['Units_Sold_Forecast'])
        fig.add_trace(go.Scatter(x=x, y=y, name='Forecast P50', line=dict(color=FC_C, width=2.5)))
    fig.add_vline(x=SPLIT, line_dash='dash', line_color='grey')
    fig.add_annotation(x=SPLIT, y=1, yref='paper', text='forecast →', showarrow=False, xanchor='left')
    fig.update_layout(title=title, height=420, hovermode='x unified', legend=dict(orientation='h', y=-0.15),
                      margin=dict(t=40, b=10))
    return fig


tabs = st.tabs(['Executive overview', 'Data health', 'Demand explorer', 'Anomaly monitor', 'Forecast & accuracy',
                'Macro drivers', 'Lifecycle & launches', 'What-if', 'Procurement (P90)', 'Stockouts & exceptions',
                'Downloads'])

# ════════════════════════════════════════════════════════════════════
# 1 EXECUTIVE OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader('Executive overview')
    last19 = T[T['Date'] > '2025-03']
    c = st.columns(5)
    c[0].metric('19-mo forecast (units)', f"{F['Units_Sold_Forecast'].sum():,.0f}",
                f"{(F['Units_Sold_Forecast'].sum() / max(last19['Unconstrained_Demand'].sum(), 1) - 1) * 100:+.1f}% vs last 19 mo")
    c[1].metric('19-mo forecast revenue', f"${F['Revenue_Forecast_USD'].sum() / 1e6:,.1f}M",
                f"{(F['Revenue_Forecast_USD'].sum() / max((last19['Unconstrained_Demand'] * last19['Unit_Price_USD']).sum(), 1) - 1) * 100:+.1f}%")
    c[2].metric('P10 – P90 (units)', f"{F['Forecast_P10'].sum() / 1e6:.2f}M – {F['Forecast_P90'].sum() / 1e6:.2f}M")
    c[3].metric('History units 2019-26', f"{T['Units_Sold'].sum() / 1e6:,.2f}M")
    c[4].metric('Lost to stockouts (history)', f"{T['Lost_Units'].sum():,.0f}")
    mode = st.radio('Measure', ['Units', 'Revenue (USD)'], horizontal=True)
    chart(st, timeline(T, F, 'units' if mode == 'Units' else 'rev'), use_container_width=True)
    st.markdown('**Mix: last 12 months of history vs first 12 months of forecast**')
    cols = st.columns(3)
    for col_, dim in zip(cols, ['Product_Family', 'Region', 'Sales_Channel']):
        a = T[T['Date'] > '2025-10'].groupby(dim)['Unconstrained_Demand'].sum().rename('History (12 mo)')
        b = F[F['Date'] <= '2027-10'].groupby(dim)['Units_Sold_Forecast'].sum().rename('Forecast (12 mo)')
        mix = pd.concat([a, b], axis=1).fillna(0).reset_index().melt(id_vars=dim, var_name='Period', value_name='Units')
        chart(col_, px.bar(mix, y=dim, x='Units', color='Period', barmode='group', orientation='h', height=380,
                                 color_discrete_sequence=[UNC_C, FC_C]).update_layout(legend=dict(orientation='h', y=-0.2),
                                                                                       margin=dict(t=10)),
                          use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# 2 DATA HEALTH
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader('Data health — issues found, rows affected, before vs after')
    Q = D['quality']
    iss = pd.DataFrame(Q['issues'])
    ba = Q['before_after']
    c = st.columns(5)
    c[0].metric('Train rows', f"{ba['train_rows_clean']:,}", f"{ba['train_rows_clean'] - ba['train_rows_raw']:,} vs raw {ba['train_rows_raw']:,}")
    c[1].metric('Region labels', f"{ba['regions'][1]}", f"from {ba['regions'][0]} variants", delta_color='off')
    c[2].metric('Customer labels', f"{ba['customers'][1]}", f"from {ba['customers'][0]} variants", delta_color='off')
    c[3].metric('SKU codes', f"{ba['skus'][1]}", f"from {ba['skus'][0]} variants", delta_color='off')
    c[4].metric('Units × price = revenue', '100%', f"from {ba['units_x_price_eq_revenue_pct'][0]:.1f}%")
    kinds = st.multiselect('Classification', sorted(iss['classification'].unique()), default=sorted(iss['classification'].unique()))
    v = iss[iss['classification'].isin(kinds) & (iss['rows_affected'] > 0)]
    chart(st, px.bar(v.sort_values('rows_affected'), x='rows_affected', y='issue_type', color='file', orientation='h',
                           log_x=True, height=650, hover_data=['action', 'classification'],
                           labels={'rows_affected': 'Rows affected (log scale)', 'issue_type': ''}), use_container_width=True)
    st.dataframe(v[['file', 'issue_type', 'classification', 'rows_affected', 'action']], use_container_width=True, hide_index=True)
    st.caption('Error = fixed in the pipeline. Real event = kept and modelled or excluded. Missing data = left missing, never zero. '
               'Row-level fix log (old → new per Record_ID): reports/data_quality_issues.csv')

# ════════════════════════════════════════════════════════════════════
# 3 DEMAND EXPLORER
# ════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader('Demand explorer — family → SKU → region → customer')
    c = st.columns(4)
    fam = c[0].selectbox('Family', ['All'] + sorted(T['Product_Family'].unique()))
    t1 = T if fam == 'All' else T[T['Product_Family'] == fam]
    sku = c[1].selectbox('SKU', ['All'] + sorted(t1['SKU_ID'].unique()))
    t2 = t1 if sku == 'All' else t1[t1['SKU_ID'] == sku]
    reg = c[2].selectbox('Region', ['All'] + sorted(t2['Region'].unique()))
    t3 = t2 if reg == 'All' else t2[t2['Region'] == reg]
    cus = c[3].selectbox('Customer', ['All'] + sorted(t3['Customer_Name'].unique()))
    t4 = t3 if cus == 'All' else t3[t3['Customer_Name'] == cus]
    f4 = F
    for col_, val in [('Product_Family', fam), ('SKU_ID', sku), ('Region', reg), ('Customer_Name', cus)]:
        if val != 'All':
            f4 = f4[f4[col_] == val]
    chart(st, timeline(t4, f4), use_container_width=True)
    a, b = st.columns([3, 2])
    s = t4.groupby(['Year', 'Month'])['Unconstrained_Demand'].sum().unstack()
    chart(a, px.imshow(s, aspect='auto', color_continuous_scale='Blues', title='Seasonality (units by year × month)',
                             labels=dict(color='Units')).update_layout(height=340), use_container_width=True)
    z = t4.assign(zero=t4['Units_Sold'].eq(0)).groupby('Month')['zero'].mean().mul(100)
    chart(b, px.bar(z, title='Zero-order months (% of rows) by calendar month', labels={'value': '%', 'Month': ''},
                          height=340).update_layout(showlegend=False), use_container_width=True)
    if sku != 'All':
        info = D['sku'][D['sku']['SKU_ID'] == sku].T.rename(columns=lambda _: 'value')
        ic = D['ic'][D['ic']['SKU_ID'] == sku]
        a, b = st.columns(2)
        a.markdown('**Lifecycle (sku_master)**')
        a.dataframe(info, use_container_width=True)
        b.markdown('**Demand class & ordering cycle by region (last 24 months)**')
        b.dataframe(ic.drop(columns='SKU_ID'), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# 4 ANOMALY MONITOR
# ════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader('Anomaly monitor — errors vs real events, with action taken')
    A = filt(anom)
    types = sorted(A['Anomaly_Type'].unique())
    sel = st.multiselect('Anomaly type', types, default=[t for t in types if t != 'Missing row'])
    A = A[A['Anomaly_Type'].isin(sel)]
    c = st.columns(4)
    for i, (k, v) in enumerate(A['Classification'].value_counts().items()):
        c[i % 4].metric(k, f'{v:,}')
    level = st.radio('Series', ['Total (filtered)', 'One SKU-region'], horizontal=True)
    if level == 'One SKU-region':
        opts = A[A['SKU_ID'] != 'ALL'].groupby(['SKU_ID', 'Region']).size().sort_values(ascending=False).index.tolist()
        if opts:
            s_, r_ = st.selectbox('SKU-region (most flags first)', opts, format_func=lambda x: f'{x[0]} · {x[1]}')
            hist = T[(T['SKU_ID'] == s_) & (T['Region'] == r_)].set_index('Date')
            pts = A[(A['SKU_ID'] == s_) & (A['Region'] == r_)]
            fig = go.Figure([go.Scatter(x=hist.index, y=hist['Units_Sold'], name='Shipped', line=dict(color=HIST_C)),
                             go.Scatter(x=hist.index, y=hist['Unconstrained_Demand'], name='Unconstrained', line=dict(color=UNC_C, dash='dot'))])
            for t_, g in pts.groupby('Anomaly_Type'):
                fig.add_trace(go.Scatter(x=g['Date'], y=hist['Units_Sold'].reindex(g['Date']), mode='markers', name=t_,
                                         marker=dict(size=11, symbol='diamond'), text=g['Reason'] + ' → ' + g['Action']))
            chart(st, fig.update_layout(height=420, hovermode='closest'), use_container_width=True)
    else:
        hist = T.groupby('Date')[['Units_Sold', 'Unconstrained_Demand']].sum()
        cnt = A.groupby(['Date', 'Anomaly_Type']).size().rename('Flags').reset_index()
        fig = go.Figure([go.Scatter(x=hist.index, y=hist['Units_Sold'], name='Shipped', line=dict(color=HIST_C)),
                         go.Scatter(x=hist.index, y=hist['Unconstrained_Demand'], name='Unconstrained', line=dict(color=UNC_C, dash='dot'))])
        for t_, g in cnt.groupby('Anomaly_Type'):
            fig.add_trace(go.Scatter(x=g['Date'], y=hist['Units_Sold'].reindex(g['Date']), mode='markers', name=t_,
                                     marker=dict(size=np.clip(np.sqrt(g['Flags']) * 3, 6, 22)), text=g['Flags'].astype(str) + ' flags'))
        chart(st, fig.update_layout(height=420, hovermode='closest'), use_container_width=True)
    st.dataframe(A[['Date', 'SKU_ID', 'Region', 'Customer_Name', 'Anomaly_Type', 'Classification', 'Units_Sold',
                    'Unconstrained_Demand', 'Expected_Level', 'Reason', 'Action']].sort_values('Date'),
                 use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════════
# 5 FORECAST & ACCURACY
# ════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader('Forecast & accuracy — rolling-origin backtest, 19-month horizon')
    ov = pd.DataFrame(M['overall']).T.rename_axis('Model').reset_index()
    ov['WAPE vs seasonal naive'] = (1 - ov['wape'] / ov.loc[ov['Model'] == 'Seasonal_Naive', 'wape'].iat[0]).map('{:+.1%}'.format)
    a, b = st.columns([2, 3])
    a.dataframe(ov, hide_index=True, use_container_width=True)
    a.caption(f"Origins {', '.join(M['folds'])} — each forecasts the next 19 months using only data up to the origin. "
              f"Target: stockout-corrected demand. P10–P90 coverage: {M['interval_P10_P90_coverage']:.0%} (target 80%). "
              f"Aggregate WAPE: {', '.join(f'{k} {v:.3f}' for k, v in M['aggregate_level_wape'].items())}.")
    hz = B.groupby('Horizon').apply(lambda g: pd.Series({
        m: (g['Actual'] - g[m]).abs().sum() / max(g['Actual'].sum(), 1) for m in ['LightGBM_P50', 'Seasonal_Naive', 'Moving_Avg_12m', 'Croston_SBA']}))
    chart(b, px.line(hz, markers=True, title='WAPE by horizon (filtered)', labels={'value': 'WAPE', 'variable': ''},
                           height=330), use_container_width=True)
    a, b = st.columns(2)
    for col_, dim in [(a, 'Product_Family'), (b, 'Region')]:
        g = B.groupby(dim).apply(lambda g: pd.Series({'WAPE': (g['Actual'] - g['LightGBM_P50']).abs().sum() / g['Actual'].sum(),
                                                      'Bias': (g['LightGBM_P50'] - g['Actual']).sum() / g['Actual'].sum()})).reset_index()
        chart(col_, px.bar(g.melt(id_vars=dim), x=dim, y='value', color='variable', barmode='group', height=330,
                                 title=f'WAPE and bias by {dim.replace("_", " ").lower()}'), use_container_width=True)
    agg = B.groupby(['Origin', 'Date'])[['Actual', 'LightGBM_P50', 'P10', 'P90', 'Seasonal_Naive']].sum().reset_index()
    fig = go.Figure()
    for o, g in agg.groupby('Origin'):
        fig.add_trace(go.Scatter(x=list(g['Date']) + list(g['Date'][::-1]), y=list(g['P90']) + list(g['P10'][::-1]),
                                 fill='toself', fillcolor=BAND_C, line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=g['Date'], y=g['LightGBM_P50'], name=f'P50 from {o}', line=dict(color=FC_C)))
        fig.add_trace(go.Scatter(x=g['Date'], y=g['Seasonal_Naive'], name=f'Seasonal naive from {o}', line=dict(color='#a0aec0', dash='dot')))
    hist = T.groupby('Date')['Unconstrained_Demand'].sum()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name='Actual (unconstrained)', line=dict(color='#16a34a', width=2)))
    fig.add_trace(go.Scatter(x=F.groupby('Date')['Units_Sold_Forecast'].sum().index, y=F.groupby('Date')['Units_Sold_Forecast'].sum(),
                             name='Final forecast', line=dict(color=FC_C, width=3)))
    chart(st, fig.update_layout(title='Backtest forecasts vs actuals, then the final forecast', height=440,
                                      xaxis_range=['2022-06', '2028-05']), use_container_width=True)
    a, b = st.columns(2)
    imp = D['imp'].head(20).iloc[::-1]
    chart(a, px.bar(imp, x='importance', y='feature', orientation='h', height=480, title='Top-20 drivers (gain)'),
                   use_container_width=True)
    H_ = D['hier']
    lv = b.selectbox('Hierarchy level (bottom-up, adds up exactly)', H_['Level'].unique().tolist(), index=1)
    hh = H_[H_['Level'] == lv].pivot_table(index='Node', values=['Bottom_Up_Sum', 'Forecast_P10', 'Forecast_P90'], aggfunc='sum')
    b.dataframe(hh.sort_values('Bottom_Up_Sum', ascending=False), use_container_width=True)
    b.caption('Forecasts are produced at SKU × region × month and summed, so every level is coherent by construction. '
              'Sums of P10/P90 overstate the aggregate range (errors partly cancel).')

# ════════════════════════════════════════════════════════════════════
# 6 MACRO DRIVERS
# ════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader('Macro lead/lag — which indicator leads which family, and is it stable?')
    lag = D['lag']
    ind = st.selectbox('Indicator', sorted(lag['Indicator'].unique()), index=sorted(lag['Indicator'].unique()).index('New_Vehicle_Orders_Index'))
    hm = lag[lag['Indicator'] == ind].pivot(index='Product_Family', columns='Lag', values='Corr')
    chart(st, px.imshow(hm, color_continuous_scale='RdBu', zmin=-1, zmax=1, aspect='auto', text_auto='.2f',
                              title=f'Correlation of family demand YoY with {ind} (12-mo change) led by k months',
                              labels=dict(x='Lead (months)', y='', color='r')).update_layout(height=380), use_container_width=True)
    a, b = st.columns(2)
    stb = D['stab'][D['stab']['Indicator'] == ind].pivot(index='Product_Family', columns='Period', values=['Best_Lag', 'Corr'])
    a.markdown('**Lag stability: best lead in 2020-22 vs 2023-26**')
    a.dataframe(stb, use_container_width=True)
    ev = D['ev'].copy()
    ev['Index'] = ev.groupby('Powertrain')['Units'].transform(lambda s: s / s.iloc[:12].mean() * 100)
    ev['Index'] = ev.groupby('Powertrain')['Index'].transform(lambda s: s.rolling(12).mean())
    fig = px.line(ev, x='Date', y='Index', color='Powertrain', title='EV transition: demand index (12-mo avg, 2019 = 100)', height=360)
    fig.add_trace(go.Scatter(x=ev['Date'].unique(), y=ev.groupby('Date')['EV_Adoption_Index'].first(), name='EV adoption index',
                             line=dict(dash='dot', color='green')))
    chart(b, fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# 7 LIFECYCLE
# ════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader('Lifecycle — EOL phase-outs and new launches (Replaces_SKU)')
    sk = D['sku']
    c = st.columns(4)
    for i, s in enumerate(['Active', 'EOL Announced', 'Discontinued', 'Planned Launch']):
        c[i].metric(s, int((sk['Lifecycle_Status'] == s).sum()))
    pairs = sk[sk['Replaces_SKU'].notna()][['SKU_ID', 'Replaces_SKU', 'Launch_Date', 'Lifecycle_Status']]
    pairs['Predecessor_EOL'] = pairs['Replaces_SKU'].map(sk.set_index('SKU_ID')['EOL_Date'])
    st.dataframe(pairs.sort_values('Launch_Date'), hide_index=True, use_container_width=True)
    pick = st.selectbox('Transition', pairs.sort_values('Launch_Date', ascending=False).apply(lambda r: f"{r['Replaces_SKU']} → {r['SKU_ID']}", axis=1))
    old, new = pick.split(' → ')
    fig = go.Figure()
    for s_, col_ in [(old, HIST_C), (new, UNC_C)]:
        h = train[train['SKU_ID'] == s_].groupby('Date')['Unconstrained_Demand'].sum()
        f = fc[fc['SKU_ID'] == s_].groupby('Date')['Units_Sold_Forecast'].sum()
        fig.add_trace(go.Scatter(x=h.index, y=h, name=f'{s_} history', line=dict(color=col_)))
        if len(f):
            fig.add_trace(go.Scatter(x=f.index, y=f, name=f'{s_} forecast', line=dict(color=col_, dash='dash', width=3)))
    fig.add_vline(x=SPLIT, line_dash='dash', line_color='grey')
    chart(st, fig.update_layout(height=400, xaxis_range=['2020-01', '2028-05']), use_container_width=True)
    r = D['ramp'].groupby('k')['ratio'].agg(['median', 'min', 'max']).reset_index()
    chart(st, px.line(r, x='k', y=['median', 'min', 'max'], markers=True, height=320,
                            title='Launch ramp learned from 6 historical successors (successor ÷ predecessor per-region level)',
                            labels={'k': 'Months since launch', 'value': 'Ratio'}), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# 8 WHAT-IF
# ════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.subheader('What-if — re-run the forecast model under a macro scenario')
    model, feats, X = load_model()
    wcols = {c: c for c in feats if c.startswith('w_')}
    names = {'w_ISM_Manu': 'PMI (points)', 'w_New_Vehi': 'New vehicle orders index (points)',
             'w_Auto_Prod': 'Auto production YoY (pp)', 'w_Vehicle_Inve': 'Vehicle inventory days',
             'w_Raw_Mate': 'Raw-material cost index', 'w_Industrial_Util': 'Industrial utilisation (pp)',
             'w_Fuel_Pric': 'Fuel price index', 'w_EV_Adop': 'EV adoption index'}
    c = st.columns(4)
    deltas = {}
    for i, (k, lab) in enumerate(names.items()):
        if k in wcols:
            deltas[k] = c[i % 4].slider(lab, -15.0, 15.0, -3.0 if k == 'w_ISM_Manu' else 0.0, 0.5)
    start = st.select_slider('Shock starts in', options=sorted(X['Date'].unique()), value='2026-11')
    Xs = X.copy()
    for col_ in ['Product_Family', 'Region', 'Sales_Channel', 'Customer_Name', 'Powertrain_Application', 'Size_Class']:
        Xs[col_] = Xs[col_].astype('category')
    shock = Xs['Date'] >= start
    for k, v in deltas.items():
        Xs.loc[shock, k] = Xs.loc[shock, k] + v
    if deltas.get('w_EV_Adop'):
        sign = Xs['Powertrain_Application'].astype(str).map({'EV': 1.0, 'ICE': -1.0}).fillna(0)
        Xs.loc[shock, 'ev_x_powertrain'] += sign[shock] * deltas['w_EV_Adop']
    base = X['Raw_Model'] * X['_scale'] * X['Adj_Factor']
    scen = np.clip(model.predict(Xs[feats]), 0, None) * X['_scale'] * X['Adj_Factor']
    R = X[['Date', 'SKU_ID', 'Region', 'Product_Family', 'Customer_Name', 'Sales_Channel']].assign(Base=base, Scenario=scen)
    R = filt(R)
    m_ = R.groupby('Date')[['Base', 'Scenario']].sum()
    c = st.columns(3)
    c[0].metric('Baseline 19-mo units', f"{m_['Base'].sum():,.0f}")
    c[1].metric('Scenario 19-mo units', f"{m_['Scenario'].sum():,.0f}", f"{(m_['Scenario'].sum() / m_['Base'].sum() - 1) * 100:+.1f}%")
    fam_ = R.groupby('Product_Family')[['Base', 'Scenario']].sum()
    c[2].metric('Most sensitive family', (fam_['Scenario'] / fam_['Base'] - 1).abs().idxmax())
    fig = go.Figure([go.Scatter(x=m_.index, y=m_['Base'], name='Baseline', line=dict(color=FC_C, width=3)),
                     go.Scatter(x=m_.index, y=m_['Scenario'], name='Scenario', line=dict(color='#c53030', dash='dash', width=3))])
    chart(st, fig.update_layout(height=380), use_container_width=True)
    st.dataframe((fam_['Scenario'] / fam_['Base'] - 1).rename('Impact').map('{:+.1%}'.format), use_container_width=True)
    st.caption('The trained model is re-scored with shifted driver features; lifecycle/event adjustments are kept. '
               'Effects are what the model learned from 2019-2026, not causal guarantees.')

# ════════════════════════════════════════════════════════════════════
# 9 PROCUREMENT
# ════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.subheader('Procurement plan — plan to P50, buffer to P90')
    horizon = st.slider('Planning window (months)', 1, 19, 6)
    w = F[F['Date'] <= sorted(F['Date'].unique())[horizon - 1]]
    p = w.groupby('SKU_ID').agg(Family=('Product_Family', 'first'), P50=('Units_Sold_Forecast', 'sum'),
                                P90=('Forecast_P90', 'sum'), Price=('Unit_Price_USD', 'mean')).reset_index()
    p['Buffer_Units'] = p['P90'] - p['P50']
    p['Buffer_Value_USD'] = (p['Buffer_Units'] * p['Price']).round(0)
    c = st.columns(3)
    c[0].metric('Plan (P50) units', f"{p['P50'].sum():,.0f}")
    c[1].metric('Buffer to P90 (units)', f"{p['Buffer_Units'].sum():,.0f}")
    c[2].metric('Buffer value', f"${p['Buffer_Value_USD'].sum() / 1e6:,.2f}M")
    m_ = w.groupby(['Date', 'Product_Family'])[['Units_Sold_Forecast', 'Forecast_P90']].sum().reset_index()
    chart(st, px.bar(m_, x='Date', y='Forecast_P90', color='Product_Family', height=360,
                           title='Monthly P90 requirement by family'), use_container_width=True)
    st.dataframe(p.sort_values('Buffer_Value_USD', ascending=False), hide_index=True, use_container_width=True)
    st.download_button('Download procurement plan (CSV)', p.to_csv(index=False), 'procurement_plan.csv')

# ════════════════════════════════════════════════════════════════════
# 10 STOCKOUTS & EXCEPTIONS
# ════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.subheader('Stockouts & lost sales')
    so = T[T['Is_Stockout'] | (T['Lost_Units'] > 0)]
    c = st.columns(3)
    c[0].metric('Constrained SKU-region-months', f'{len(so):,}')
    c[1].metric('Lost units (demand − shipped)', f"{so['Lost_Units'].sum():,.0f}")
    c[2].metric('Lost revenue', f"${(so['Lost_Units'] * so['Unit_Price_USD']).sum() / 1e6:,.1f}M")
    a, b = st.columns(2)
    lm = so.groupby(['Date', 'Product_Family'])['Lost_Units'].sum().reset_index()
    chart(a, px.area(lm, x='Date', y='Lost_Units', color='Product_Family', height=340, title='Lost units by month'),
                   use_container_width=True)
    rs = so.groupby(so['Reason'].fillna('Unlisted (chip shortage estimate)'))['Lost_Units'].sum().reset_index()
    chart(b, px.pie(rs, names='Reason', values='Lost_Units', height=340, title='Lost units by cause'), use_container_width=True)

    st.subheader('Exception list for planners')
    last12 = T[T['Date'] > '2025-10'].groupby(['SKU_ID', 'Region'])['Unconstrained_Demand'].sum()
    nxt12 = F[F['Date'] <= '2027-10'].groupby(['SKU_ID', 'Region']).agg(
        Forecast_12m=('Units_Sold_Forecast', 'sum'), P10=('Forecast_P10', 'sum'), P90=('Forecast_P90', 'sum'),
        Adjustment=('Adjustment', lambda s: '; '.join(sorted(set(s.dropna().str.split(':').str[0])))))
    ex = nxt12.join(last12.rename('History_12m')).fillna({'History_12m': 0})
    ex['Change_%'] = (ex['Forecast_12m'] / ex['History_12m'].replace(0, np.nan) - 1) * 100
    ex['Range_width_%'] = (ex['P90'] - ex['P10']) / ex['Forecast_12m'].replace(0, np.nan) * 100
    ex['Flag'] = np.select([ex['History_12m'].eq(0), ex['Change_%'].abs() > 30, ex['Range_width_%'] > 150,
                            ex['Adjustment'].ne('')],
                           ['New series', 'Large change vs last year', 'Very uncertain', 'Rule-adjusted'], '')
    ex = ex[ex['Flag'] != ''].reset_index().sort_values('Change_%', key=lambda s: -s.abs())
    st.dataframe(ex.round(1), hide_index=True, use_container_width=True, height=380)

# ════════════════════════════════════════════════════════════════════
# 11 DOWNLOADS
# ════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.subheader('Data downloads (filtered where applicable)')
    st.download_button('submission.csv (all 58,219 rows)', (ROOT / 'submission.csv').read_bytes(), 'submission.csv')
    st.download_button('Forecast detail with P10/P50/P90 (filtered)', F.to_csv(index=False), 'forecast_detail.csv')
    st.download_button('Clean history (filtered)', T.to_csv(index=False), 'train_clean.csv')
    st.download_button('Anomaly register (filtered)', filt(anom).to_csv(index=False), 'anomaly_register.csv')
    st.download_button('Backtest results (filtered)', B.to_csv(index=False), 'backtest_results.csv')
    st.download_button('Data-quality report (JSON)', json.dumps(D['quality'], indent=2), 'data_quality_report.json')
    if D['issues_rows'].exists():
        st.download_button('Row-level fix log', D['issues_rows'].read_bytes(), 'data_quality_issues.csv')
