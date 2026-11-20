"""
Automated Integrity & Verification Test Suite for Celeris Demand Forecasting Control Tower.
Standard library unittest - runs without external dependencies.
"""

import os
import unittest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_DIR = os.path.join(DATA_DIR, "cleaned")
FEAT_DIR = os.path.join(DATA_DIR, "features")
FORECAST_DIR = os.path.join(DATA_DIR, "forecasts")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_DIR, "dataset", "student_data", "sample_submission.csv")
TEST_RAW_PATH = os.path.join(BASE_DIR, "dataset", "student_data", "test.csv")


class TestCelerisSolution(unittest.TestCase):

    def test_01_submission_format_and_row_count(self):
        """Verify submission.csv exists and has exactly 58,219 rows matching test."""
        self.assertTrue(os.path.exists(SUBMISSION_PATH), "submission.csv is missing!")
        sub = pd.read_csv(SUBMISSION_PATH)
        self.assertEqual(len(sub), 58219, f"Expected 58,219 rows, got {len(sub)}")
        expected_cols = ["Date", "SKU_ID", "Region", "Units_Sold_Forecast"]
        self.assertEqual(list(sub.columns), expected_cols, f"Columns mismatch: {sub.columns}")

    def test_02_submission_validity(self):
        """Verify submission has zero null values and strictly non-negative predictions."""
        sub = pd.read_csv(SUBMISSION_PATH)
        self.assertEqual(sub["Units_Sold_Forecast"].isnull().sum(), 0, "Submission contains NaN values!")
        self.assertEqual((sub["Units_Sold_Forecast"] < 0).sum(), 0, "Submission contains negative values!")

    def test_03_submission_alignment(self):
        """Verify submission keys match test.csv."""
        if os.path.exists(TEST_RAW_PATH):
            raw_test = pd.read_csv(TEST_RAW_PATH)
            sub = pd.read_csv(SUBMISSION_PATH)
            self.assertEqual(len(raw_test), len(sub), "Row counts do not match!")
            self.assertTrue((raw_test["Date"].values == sub["Date"].values).all(), "Dates do not match!")

    def test_04_data_hygiene(self):
        """Verify clean dataset has standardized regions and valid dates."""
        clean_path = os.path.join(CLEAN_DIR, "train_clean.parquet")
        if os.path.exists(clean_path):
            df = pd.read_parquet(clean_path)
            valid_regions = {"India", "Europe", "North America", "Asia Pacific", "Latin America", "Middle East & Africa"}
            self.assertTrue(set(df["Region"].unique()).issubset(valid_regions), "Unsanitized regions detected!")
            self.assertTrue(np.issubdtype(df["Units_Sold"].dtype, np.number), "Units_Sold is not numeric!")
            self.assertLessEqual(df["Year"].max(), 2026, "Corrupted future dates detected (>2026)!")

    def test_05_anomaly_register(self):
        """Verify anomaly detection produced valid register with categorization."""
        anom_path = os.path.join(REPORTS_DIR, "anomaly_register.csv")
        if os.path.exists(anom_path):
            anom = pd.read_csv(anom_path)
            self.assertGreater(len(anom), 5000, f"Expected >5000 anomalies, found {len(anom)}")
            self.assertIn("Anomaly_Type", anom.columns)

    def test_06_hierarchical_reconciliation(self):
        """Verify hierarchical reconciliation file exists and has coherent aggregates."""
        rec_path = os.path.join(FORECAST_DIR, "hierarchical_reconciliation.csv")
        if os.path.exists(rec_path):
            rec = pd.read_csv(rec_path)
            self.assertGreater(len(rec), 0, "Reconciliation table is empty!")
            diff = (rec["Bottom_Up_Sum"] - rec["Reconciled_Sum"]).abs()
            self.assertTrue((diff < 1e-3).all(), "Reconciliation exhibits mathematical discrepancy!")

    def test_07_eol_enforcement(self):
        """Verify post-EOL SKUs receive zero forecast."""
        sub = pd.read_csv(SUBMISSION_PATH)
        eol_mask = (sub["SKU_ID"] == "PTC-005") & (sub["Date"] >= "2027-06")
        if eol_mask.sum() > 0:
            self.assertTrue(
                (sub.loc[eol_mask, "Units_Sold_Forecast"] == 0).all(),
                "Post-EOL SKU has non-zero predictions!"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
