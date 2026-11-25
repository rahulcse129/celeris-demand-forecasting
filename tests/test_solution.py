"""
Integrity tests for the pipeline outputs and the core cleaning helpers.
    python -m pytest tests -q        (or)        python tests/test_solution.py
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from utils import parse_date_to_yyyymm, standardize_customer, standardize_region, standardize_sku  # noqa: E402

RAW = ROOT / 'dataset' / 'student_data'
CLEAN = ROOT / 'data' / 'cleaned'
FC = ROOT / 'data' / 'forecasts'
REP = ROOT / 'reports'
REGIONS = {'India', 'Europe', 'North America', 'Asia Pacific', 'Latin America', 'Middle East & Africa'}


class TestHelpers(unittest.TestCase):
    def test_dates(self):
        for raw in ['2024-02', '2024/02', 'Feb-2024', 'Feb 2024', '02-2024', '02/2024', '2024-02-01']:
            self.assertEqual(parse_date_to_yyyymm(raw), '2024-02', raw)

    def test_labels(self):
        self.assertEqual(standardize_sku(' pmp_23'), 'PMP-023')
        self.assertEqual(standardize_sku('CSS31'), 'CSS-031')
        self.assertEqual(standardize_region('LatAm'), 'Latin America')
        self.assertEqual(standardize_region('ME&A'), 'Middle East & Africa')
        self.assertEqual(standardize_customer('HELIOS  AUTO PARTS Inc.'), 'Helios Auto Parts')


class TestCleanData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = pd.read_parquet(CLEAN / 'train_clean.parquet')

    def test_one_row_per_key(self):
        self.assertFalse(self.t.duplicated(['Date', 'SKU_ID', 'Region']).any())

    def test_labels_canonical(self):
        self.assertTrue(set(self.t['Region']) <= REGIONS)
        self.assertEqual(self.t['Sales_Channel'].nunique(), 4)
        self.assertEqual(self.t['Customer_Name'].nunique(), 12)
        self.assertEqual(self.t.groupby('Product_Family')['Business_Unit'].nunique().max(), 1)

    def test_dates_in_range(self):
        self.assertEqual(self.t['Date'].min(), '2019-01')
        self.assertEqual(self.t['Date'].max(), '2026-10')
        self.assertTrue((self.t['Date'].str[:4].astype(int) == self.t['Year']).all())

    def test_reconciles(self):
        self.assertTrue((self.t['Units_Sold'] >= 0).all())
        err = (self.t['Units_Sold'] * self.t['Unit_Price_USD'] - self.t['Revenue_USD']).abs()
        self.assertLess(err.max(), 0.02)

    def test_unconstrained_ge_shipped_in_stockouts(self):
        s = self.t[self.t['Is_Stockout']]
        self.assertTrue((s['Unconstrained_Demand'] >= s['Units_Sold'] - 1e-9).mean() > 0.99)


class TestSubmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sub = pd.read_csv(ROOT / 'submission.csv')
        cls.test = pd.read_csv(RAW / 'test.csv')

    def test_format(self):
        self.assertEqual(list(self.sub.columns), ['Date', 'SKU_ID', 'Region', 'Units_Sold_Forecast'])
        self.assertEqual(len(self.sub), len(self.test))
        self.assertTrue((self.sub[['Date', 'SKU_ID', 'Region']].values == self.test[['Date', 'SKU_ID', 'Region']].values).all())

    def test_values(self):
        v = self.sub['Units_Sold_Forecast']
        self.assertEqual(v.isna().sum(), 0)
        self.assertTrue((v >= 0).all())
        self.assertTrue(np.issubdtype(v.dtype, np.integer))

    def test_lifecycle(self):
        sku = pd.read_csv(RAW / 'sku_master.csv', dtype=str).set_index('SKU_ID')
        eol = self.sub['SKU_ID'].map(sku['EOL_Date'])
        after = eol.notna() & (self.sub['Date'] > eol)
        self.assertGreater(after.sum(), 0)
        self.assertTrue((self.sub.loc[after, 'Units_Sold_Forecast'] == 0).all())
        launches = self.sub['SKU_ID'].isin(sku.index[sku['Lifecycle_Status'] == 'Planned Launch'])
        self.assertGreater(self.sub.loc[launches, 'Units_Sold_Forecast'].sum(), 0)

    def test_quantiles_ordered(self):
        f = pd.read_csv(FC / 'forecast_detail.csv')
        self.assertTrue((f['Forecast_P10'] <= f['Forecast_P50']).all())
        self.assertTrue((f['Forecast_P50'] <= f['Forecast_P90']).all())

    def test_hierarchy_coherent(self):
        f = pd.read_csv(FC / 'forecast_detail.csv')
        h = pd.read_csv(FC / 'hierarchical_reconciliation.csv')
        fam = h[h['Level'] == 'Product_Family'].groupby('Date')['Bottom_Up_Sum'].sum()
        tot = h[h['Level'] == 'Total'].set_index('Date')['Bottom_Up_Sum']
        self.assertTrue((fam == tot).all())
        self.assertEqual(tot.sum(), f['Units_Sold_Forecast'].sum())


class TestReports(unittest.TestCase):
    def test_quality_report(self):
        q = json.loads((REP / 'data_quality_report.json').read_text())
        self.assertGreater(len(q['issues']), 20)
        self.assertTrue({'file', 'issue_type', 'rows_affected', 'action', 'classification'} <= set(q['issues'][0]))

    def test_anomaly_register(self):
        a = pd.read_csv(REP / 'anomaly_register.csv')
        self.assertTrue({'Anomaly_Type', 'Classification', 'Reason', 'Action'} <= set(a.columns))
        self.assertGreater(a['Classification'].nunique(), 2)

    def test_backtest_beats_baselines(self):
        m = json.loads((REP / 'model_metrics.json').read_text())['overall']
        best_baseline = min(m[k]['wape'] for k in ['Seasonal_Naive', 'Moving_Avg_12m', 'Croston_SBA'])
        self.assertLess(m['LightGBM_P50']['wape'], best_baseline)


if __name__ == '__main__':
    unittest.main(verbosity=2)
