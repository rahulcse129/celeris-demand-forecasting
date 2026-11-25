"""Builds notebooks/celeris_demand_forecasting_walkthrough.ipynb (reads pipeline outputs only)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(s):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': s.strip().splitlines(True)}


def code(s):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.strip().splitlines(True)}


cells = [
    md("""
# Celeris Automobili — demand forecasting walkthrough
Reads the outputs of `run_pipeline` (no numbers typed in). Sections: data quality → anomalies →
drivers → backtest → final forecast.
"""),
    code("""
import json, pandas as pd, numpy as np
pd.set_option('display.width', 200)
q = json.load(open('../reports/data_quality_report.json'))
issues = pd.DataFrame(q['issues'])
print(q['before_after'])
issues.sort_values('rows_affected', ascending=False)[['file','issue_type','classification','rows_affected','action']]
"""),
    md("## 2. Anomaly register — errors vs real events"),
    code("""
a = pd.read_csv('../reports/anomaly_register.csv')
a.groupby(['Classification','Anomaly_Type']).size().rename('rows')
"""),
    md("## 3. Drivers — which indicator leads which family, at what lag, and is it stable?"),
    code("""
best = pd.read_csv('../data/features/macro_best_lag.csv')
stab = pd.read_csv('../data/features/macro_lag_stability.csv')
display(best.sort_values('Corr', key=abs, ascending=False).head(10))
stab[stab.Indicator=='New_Vehicle_Orders_Index'].pivot(index='Product_Family', columns='Period', values=['Best_Lag','Corr'])
"""),
    md("## 4. Rolling-origin backtest (3 origins × 19-month horizon)"),
    code("""
m = json.load(open('../reports/model_metrics.json'))
display(pd.DataFrame(m['overall']).T)
print('P10-P90 coverage', m['interval_P10_P90_coverage'], '| aggregate WAPE', m['aggregate_level_wape'])
pd.DataFrame(m['by_horizon']).T[['wape','wape_snaive','bias']].plot(title='WAPE by horizon', figsize=(9,3));
"""),
    md("## 5. Final forecast (Nov 2026 – May 2028)"),
    code("""
fc = pd.read_csv('../data/forecasts/forecast_detail.csv')
sub = pd.read_csv('../submission.csv')
assert len(sub) == 58219 and sub.Units_Sold_Forecast.ge(0).all()
print(m['test_forecast_summary'])
fc.groupby('Date')[['Forecast_P10','Units_Sold_Forecast','Forecast_P90']].sum().plot(figsize=(9,3));
"""),
    code("""
fc[fc.Adjustment.notna()].groupby(fc.Adjustment.str.split(':').str[0]).size().rename('rows adjusted by rule')
"""),
]
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
      'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks' / 'celeris_demand_forecasting_walkthrough.ipynb'
out.write_text(json.dumps(nb, indent=1))
print('wrote', out)
