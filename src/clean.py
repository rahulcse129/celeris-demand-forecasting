"""
One-command data cleaning pipeline for MKS Hackathon.
Usage: python src/clean.py

Reads raw data from dataset/student_data/
Writes cleaned data to data/cleaned/
Writes data quality report to reports/data_quality_report.json
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
from utils import (
    standardize_region, standardize_channel, standardize_customer,
    parse_date_to_yyyymm, infer_family_from_sku,
    CLEAN_REGIONS, CLEAN_CHANNELS, CANONICAL_CUSTOMERS, FAMILY_PREFIX_MAP
)

warnings.filterwarnings('ignore')

RAW_DIR = PROJECT_ROOT / 'dataset' / 'student_data'
CLEAN_DIR = PROJECT_ROOT / 'data' / 'cleaned'
REPORT_DIR = PROJECT_ROOT / 'reports'

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

quality_report = {
    'pipeline_run': datetime.now().isoformat(),
    'files': {},
    'summary': {}
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════
log("Loading raw data files...")

train_raw = pd.read_csv(RAW_DIR / 'train.csv', dtype=str)
test_raw = pd.read_csv(RAW_DIR / 'test.csv', dtype=str)
sku_master = pd.read_csv(RAW_DIR / 'sku_master.csv', dtype=str)
macro = pd.read_csv(RAW_DIR / 'macro_indicators.csv', dtype=str)
events = pd.read_csv(RAW_DIR / 'events_calendar.csv', dtype=str)
stockouts = pd.read_csv(RAW_DIR / 'inventory_stockouts.csv', dtype=str)

log(f"  train: {len(train_raw):,} rows | test: {len(test_raw):,} rows")


# ═══════════════════════════════════════════════════════════════════════
# 2. PROFILE RAW TRAIN DATA
# ═══════════════════════════════════════════════════════════════════════
log("Profiling raw train data...")

train_profile = {
    'raw_rows': len(train_raw),
    'raw_columns': list(train_raw.columns),
    'null_counts': train_raw.isnull().sum().to_dict(),
    'blank_counts': {c: int((train_raw[c].fillna('').str.strip() == '').sum()) for c in train_raw.columns},
    'unique_counts': {c: int(train_raw[c].nunique()) for c in train_raw.columns},
}
quality_report['files']['train_raw'] = train_profile


# ═══════════════════════════════════════════════════════════════════════
# 3. CLEAN train.csv
# ═══════════════════════════════════════════════════════════════════════
log("Cleaning train.csv...")

train = train_raw.copy()
issues = []

# ── 3.1 Fix Dates ──────────────────────────────────────────────────────
log("  3.1 Standardizing dates...")
train['Date_Clean'] = train['Date'].apply(parse_date_to_yyyymm)
date_mismatches = int((train['Date'] != train['Date_Clean']).sum())
issues.append({'type': 'date_format_inconsistency', 'rows_affected': date_mismatches,
               'action': 'Parsed all dates to YYYY-MM format'})
train['Date'] = train['Date_Clean']
train.drop(columns=['Date_Clean'], inplace=True)

# Check for any dates that couldn't be parsed to YYYY-MM
bad_date_mask = ~train['Date'].str.match(r'^\d{4}-\d{2}$', na=False)
bad_date_count = int(bad_date_mask.sum())
if bad_date_count > 0:
    log(f"  WARNING: {bad_date_count} rows with unparseable dates. Samples: {train.loc[bad_date_mask, 'Date'].head(5).tolist()}")
    # Try to salvage or drop
    train = train[~bad_date_mask].copy()
    issues.append({'type': 'unparseable_dates', 'rows_removed': bad_date_count,
                   'action': 'Dropped rows with dates that could not be parsed to YYYY-MM'})

# Extract correct Year/Month from authoritative Date
train['Year'] = train['Date'].str[:4].astype(int)
train['Month'] = train['Date'].str[5:7].astype(int)

# ── 3.1b Fix future dates beyond training period ──────────────────────
# Training data should be 2019-01 to 2026-10 max
# Year 2062 is likely a typo for 2026 (swapped last two digits)
log("  3.1b Fixing future dates (e.g., 2062 → 2026)...")
future_mask = train['Year'] > 2026
future_count = int(future_mask.sum())
if future_count > 0:
    # Attempt: swap to reasonable year. 2062 → 2026
    train.loc[future_mask & (train['Year'] == 2062), 'Year'] = 2026
    # Reconstruct date
    still_future = train['Year'] > 2026
    train = train[~still_future].copy()
    # Rebuild Date from fixed Year/Month
    train['Date'] = train['Year'].astype(str) + '-' + train['Month'].astype(str).str.zfill(2)
    issues.append({'type': 'future_dates_fixed', 'rows_affected': future_count,
                   'action': 'Fixed year 2062→2026 (digit swap); dropped any remaining future dates'})

# ── 3.2 Standardize Regions ───────────────────────────────────────────
log("  3.2 Standardizing regions...")
region_before = train['Region'].nunique()
train['Region'] = train['Region'].apply(standardize_region)
blank_regions = int(train['Region'].isna().sum())
issues.append({'type': 'region_inconsistency', 'variants_before': region_before,
               'variants_after': train['Region'].nunique(), 'blank_regions': blank_regions,
               'action': f'Mapped {region_before} variants to {train["Region"].nunique()} clean regions'})

# ── 3.3 Standardize Sales Channels ────────────────────────────────────
log("  3.3 Standardizing sales channels...")
channel_before = train['Sales_Channel'].nunique()
train['Sales_Channel'] = train['Sales_Channel'].apply(standardize_channel)
issues.append({'type': 'channel_inconsistency', 'variants_before': channel_before,
               'variants_after': train['Sales_Channel'].nunique(),
               'action': f'Mapped {channel_before} variants to {train["Sales_Channel"].nunique()} clean channels'})

# ── 3.4 Standardize Customer Names ────────────────────────────────────
log("  3.4 Standardizing customer names...")
customer_before = train['Customer_Name'].nunique()
train['Customer_Name'] = train['Customer_Name'].apply(standardize_customer)
customer_after = train['Customer_Name'].nunique()
issues.append({'type': 'customer_name_typos', 'variants_before': customer_before,
               'variants_after': customer_after,
               'action': f'Fuzzy-matched {customer_before} variants to {customer_after} clean names'})

# ── 3.5 Fix blank Product_Family ──────────────────────────────────────
log("  3.5 Fixing blank Product_Family...")
blank_family_mask = train['Product_Family'].fillna('').str.strip() == ''
blank_family_count = int(blank_family_mask.sum())
train.loc[blank_family_mask, 'Product_Family'] = train.loc[blank_family_mask, 'SKU_ID'].apply(infer_family_from_sku)
issues.append({'type': 'blank_product_family', 'rows_affected': blank_family_count,
               'action': 'Inferred from SKU_ID prefix'})

# ── 3.6 Convert numeric columns ──────────────────────────────────────
log("  3.6 Converting numeric columns...")
train['Units_Sold'] = pd.to_numeric(train['Units_Sold'], errors='coerce')
train['Unit_Price_USD'] = pd.to_numeric(train['Unit_Price_USD'], errors='coerce')
train['Revenue_USD'] = pd.to_numeric(train['Revenue_USD'], errors='coerce')

numeric_nulls = int(train[['Units_Sold', 'Unit_Price_USD', 'Revenue_USD']].isna().any(axis=1).sum())
issues.append({'type': 'numeric_conversion_errors', 'rows_affected': numeric_nulls,
               'action': 'Coerced to numeric; NaN where unparseable'})

# ── 3.7 De-duplicate by Record_ID ─────────────────────────────────────
log("  3.7 De-duplicating by Record_ID...")
rows_before_dedup = len(train)
# Parse Load_Timestamp for sorting
train['Load_Timestamp_Parsed'] = pd.to_datetime(train['Load_Timestamp'], errors='coerce')
# Sort by timestamp descending so first occurrence is the latest
train = train.sort_values('Load_Timestamp_Parsed', ascending=False)
# Keep first (latest) for each Record_ID
dup_mask = train.duplicated(subset=['Record_ID'], keep='first')
duplicates_removed = int(dup_mask.sum())
train = train[~dup_mask].copy()
issues.append({'type': 'duplicate_record_ids', 'rows_removed': duplicates_removed,
               'rows_before': rows_before_dedup, 'rows_after': len(train),
               'action': 'Kept row with latest Load_Timestamp per Record_ID'})
train.drop(columns=['Load_Timestamp_Parsed'], inplace=True)

# ── 3.8 Revenue Reconciliation ────────────────────────────────────────
log("  3.8 Reconciling revenue...")
train['Computed_Revenue'] = train['Units_Sold'] * train['Unit_Price_USD']
revenue_mismatch = train[
    (train['Computed_Revenue'].notna()) & (train['Revenue_USD'].notna()) &
    (abs(train['Computed_Revenue'] - train['Revenue_USD']) > 0.01)
]
revenue_mismatch_count = len(revenue_mismatch)

# Check for unit errors (factor of 10, 100, 1000)
unit_error_rows = []
for factor in [10, 100, 1000]:
    mask = (
        (train['Computed_Revenue'].notna()) & (train['Revenue_USD'].notna()) &
        (abs(train['Computed_Revenue'] * factor - train['Revenue_USD']) < 1.0)
    )
    unit_error_rows.append(int(mask.sum()))
    # Fix: multiply Units_Sold by factor
    if mask.sum() > 0:
        train.loc[mask, 'Units_Sold'] = train.loc[mask, 'Units_Sold'] * factor
        train.loc[mask, 'Computed_Revenue'] = train.loc[mask, 'Units_Sold'] * train.loc[mask, 'Unit_Price_USD']

# Also check reverse (Units_Sold too large)
for factor in [10, 100, 1000]:
    mask = (
        (train['Computed_Revenue'].notna()) & (train['Revenue_USD'].notna()) &
        (abs(train['Computed_Revenue'] / factor - train['Revenue_USD']) < 1.0)
    )
    if mask.sum() > 0:
        train.loc[mask, 'Units_Sold'] = train.loc[mask, 'Units_Sold'] / factor
        train.loc[mask, 'Computed_Revenue'] = train.loc[mask, 'Units_Sold'] * train.loc[mask, 'Unit_Price_USD']

# Recompute revenue
train['Revenue_USD'] = train['Units_Sold'] * train['Unit_Price_USD']
train.drop(columns=['Computed_Revenue'], inplace=True)

issues.append({'type': 'revenue_mismatch', 'rows_affected': revenue_mismatch_count,
               'unit_errors_x10': unit_error_rows[0], 'unit_errors_x100': unit_error_rows[1],
               'unit_errors_x1000': unit_error_rows[2],
               'action': 'Fixed unit scaling errors; recomputed Revenue = Units * Price'})

# ── 3.9 Validate SKU_IDs against master ───────────────────────────────
log("  3.9 Validating SKU_IDs against master...")
master_skus = set(sku_master['SKU_ID'].unique())
train_skus = set(train['SKU_ID'].unique())
skus_not_in_master = train_skus - master_skus
skus_in_master_not_train = master_skus - train_skus

issues.append({'type': 'sku_validation', 'train_unique_skus': len(train_skus),
               'master_skus': len(master_skus), 'in_train_not_master': len(skus_not_in_master),
               'in_master_not_train': len(skus_in_master_not_train),
               'sample_not_in_master': sorted(list(skus_not_in_master))[:10],
               'action': 'Flagged SKUs not in master for investigation'})

# ── 3.10 Remove rows with invalid regions (after standardization) ─────
log("  3.10 Removing rows with invalid/blank regions...")
invalid_region_mask = ~train['Region'].isin(CLEAN_REGIONS) | train['Region'].isna()
invalid_region_count = int(invalid_region_mask.sum())
train = train[~invalid_region_mask].copy()
issues.append({'type': 'invalid_region_removed', 'rows_removed': invalid_region_count,
               'action': 'Removed rows where Region could not be mapped to a clean value'})

# ── 3.11 Aggregate to Date-SKU-Region (Core requirement) ─────────────
log("  3.11 Aggregating to one row per Date-SKU-Region...")
rows_before_agg = len(train)

# Keep the most common categorical values during aggregation
agg_dict = {
    'Units_Sold': 'sum',
    'Revenue_USD': 'sum',
    'Unit_Price_USD': 'mean',
    'Product_Family': 'first',
    'Business_Unit': 'first',
    'SKU_Description': 'first',
    'Customer_Name': 'first',
    'Sales_Channel': 'first',
    'Year': 'first',
    'Month': 'first',
}

train_agg = train.groupby(['Date', 'SKU_ID', 'Region'], as_index=False).agg(agg_dict)
issues.append({'type': 'aggregation', 'rows_before': rows_before_agg,
               'rows_after': len(train_agg),
               'action': 'Aggregated to one row per Date-SKU-Region (summed Units_Sold and Revenue)'})

train = train_agg.copy()

# ── 3.12 Merge lifecycle info from sku_master ─────────────────────────
log("  3.12 Merging SKU lifecycle info...")
sku_cols = ['SKU_ID', 'Size_Class', 'Powertrain_Application', 'Launch_Date', 'EOL_Date',
            'Replaces_SKU', 'Lifecycle_Status']
sku_merge = sku_master[sku_cols].copy()
train = train.merge(sku_merge, on='SKU_ID', how='left')

log(f"  Clean train: {len(train):,} rows")


# ═══════════════════════════════════════════════════════════════════════
# 4. CLEAN test.csv
# ═══════════════════════════════════════════════════════════════════════
log("Cleaning test.csv...")

test = test_raw.copy()
test['Date'] = test['Date'].apply(parse_date_to_yyyymm)
test['Year'] = test['Date'].str.split('-').str[0].astype(int)
test['Month'] = test['Date'].str.split('-').str[1].astype(int)
test['Region'] = test['Region'].apply(standardize_region)
test['Sales_Channel'] = test['Sales_Channel'].apply(standardize_channel)
test['Customer_Name'] = test['Customer_Name'].apply(standardize_customer)

# Fix blank Product_Family
blank_fam_test = test['Product_Family'].fillna('').str.strip() == ''
test.loc[blank_fam_test, 'Product_Family'] = test.loc[blank_fam_test, 'SKU_ID'].apply(infer_family_from_sku)

# Merge lifecycle
test = test.merge(sku_merge, on='SKU_ID', how='left')

log(f"  Clean test: {len(test):,} rows")


# ═══════════════════════════════════════════════════════════════════════
# 5. CLEAN macro_indicators.csv
# ═══════════════════════════════════════════════════════════════════════
log("Cleaning macro_indicators.csv...")

macro['Date'] = macro['Date'].apply(parse_date_to_yyyymm)
macro['Year'] = macro['Date'].str.split('-').str[0].astype(int)
macro['Month'] = macro['Date'].str.split('-').str[1].astype(int)

# Convert numeric columns
numeric_cols = ['ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct', 'Vehicle_Inventory_Days',
                'Raw_Material_Cost_Index', 'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
                'Fuel_Price_Index', 'EV_Adoption_Index']
for col in numeric_cols:
    macro[col] = pd.to_numeric(macro[col], errors='coerce')

# Check for nulls and interpolate
macro_nulls = int(macro[numeric_cols].isna().sum().sum())
if macro_nulls > 0:
    macro[numeric_cols] = macro[numeric_cols].interpolate(method='linear')

quality_report['files']['macro_indicators'] = {
    'rows': len(macro),
    'null_values_found': macro_nulls,
    'date_range': f"{macro['Date'].min()} to {macro['Date'].max()}",
    'action': 'Interpolated missing values'
}

log(f"  Clean macro: {len(macro)} rows, {macro_nulls} nulls interpolated")


# ═══════════════════════════════════════════════════════════════════════
# 6. CLEAN inventory_stockouts.csv
# ═══════════════════════════════════════════════════════════════════════
log("Cleaning inventory_stockouts.csv...")

stockouts['Date'] = stockouts['Date'].apply(parse_date_to_yyyymm)
stockouts['Region'] = stockouts['Region'].apply(standardize_region)

for col in ['Requested_Units', 'Carried_In_Units', 'Shipped_Units',
            'Backordered_Units', 'Lost_Units', 'Fill_Rate_Pct', 'Stockout_Days']:
    stockouts[col] = pd.to_numeric(stockouts[col], errors='coerce')

log(f"  Clean stockouts: {len(stockouts):,} rows")


# ═══════════════════════════════════════════════════════════════════════
# 7. CLEAN events_calendar.csv
# ═══════════════════════════════════════════════════════════════════════
log("Cleaning events_calendar.csv...")

events['Start_Date'] = events['Start_Date'].apply(parse_date_to_yyyymm)
events['End_Date'] = events['End_Date'].apply(parse_date_to_yyyymm)

log(f"  Clean events: {len(events)} rows")


# ═══════════════════════════════════════════════════════════════════════
# 8. CREATE UNCONSTRAINED DEMAND
# ═══════════════════════════════════════════════════════════════════════
log("Creating unconstrained demand column...")

# Merge stockout info to identify constrained months
stockout_demand = stockouts[['Date', 'SKU_ID', 'Region', 'Requested_Units']].copy()
stockout_demand = stockout_demand.rename(columns={'Requested_Units': 'Unconstrained_Demand'})

train = train.merge(stockout_demand, on=['Date', 'SKU_ID', 'Region'], how='left')
# Where we have stockout info, use Requested_Units as true demand; otherwise use Units_Sold
train['Unconstrained_Demand'] = train['Unconstrained_Demand'].fillna(train['Units_Sold'])

constrained_rows = int(train['Unconstrained_Demand'].ne(train['Units_Sold']).sum())
log(f"  Corrected {constrained_rows:,} rows with unconstrained demand from stockout data")

issues.append({'type': 'unconstrained_demand', 'rows_corrected': constrained_rows,
               'action': 'Used Requested_Units from stockout data as true demand where available'})


# ═══════════════════════════════════════════════════════════════════════
# 9. SAVE CLEANED FILES
# ═══════════════════════════════════════════════════════════════════════
log("Saving cleaned files...")

train.to_csv(CLEAN_DIR / 'train_clean.csv', index=False)
test.to_csv(CLEAN_DIR / 'test_clean.csv', index=False)
macro.to_csv(CLEAN_DIR / 'macro_indicators_clean.csv', index=False)
stockouts.to_csv(CLEAN_DIR / 'inventory_stockouts_clean.csv', index=False)
events.to_csv(CLEAN_DIR / 'events_calendar_clean.csv', index=False)
sku_master.to_csv(CLEAN_DIR / 'sku_master_clean.csv', index=False)

log(f"  All files saved to {CLEAN_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# 10. DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════
log("Generating data quality report...")

quality_report['files']['train_cleaned'] = {
    'clean_rows': len(train),
    'unique_dates': int(train['Date'].nunique()),
    'date_range': f"{train['Date'].min()} to {train['Date'].max()}",
    'unique_skus': int(train['SKU_ID'].nunique()),
    'unique_regions': int(train['Region'].nunique()),
    'regions': sorted(train['Region'].unique().tolist()),
    'unique_customers': int(train['Customer_Name'].nunique()),
    'unique_channels': int(train['Sales_Channel'].nunique()),
    'unique_families': int(train['Product_Family'].nunique()),
    'families': sorted(train['Product_Family'].dropna().unique().tolist()),
}

quality_report['files']['test_cleaned'] = {
    'clean_rows': len(test),
    'unique_dates': int(test['Date'].nunique()),
    'date_range': f"{test['Date'].min()} to {test['Date'].max()}",
}

quality_report['issues'] = issues
quality_report['summary'] = {
    'total_issues_found': len(issues),
    'total_rows_raw': train_profile['raw_rows'],
    'total_rows_clean': len(train),
    'rows_removed': train_profile['raw_rows'] - len(train),
    'cleaning_pipeline': 'src/clean.py',
    'output_directory': str(CLEAN_DIR),
}

with open(REPORT_DIR / 'data_quality_report.json', 'w') as f:
    json.dump(quality_report, f, indent=2, default=str)

log(f"  Data quality report saved to {REPORT_DIR / 'data_quality_report.json'}")


# ═══════════════════════════════════════════════════════════════════════
# 11. SUMMARY
# ═══════════════════════════════════════════════════════════════════════
log("=" * 60)
log("CLEANING PIPELINE COMPLETE")
log("=" * 60)
log(f"  Raw train rows:   {train_profile['raw_rows']:>10,}")
log(f"  Clean train rows: {len(train):>10,}")
log(f"  Rows removed:     {train_profile['raw_rows'] - len(train):>10,}")
log(f"  Issues found:     {len(issues):>10}")
log(f"  Date range:       {train['Date'].min()} → {train['Date'].max()}")
log(f"  Regions:          {sorted(train['Region'].unique().tolist())}")
log(f"  Families:         {train['Product_Family'].nunique()}")
log(f"  SKUs:             {train['SKU_ID'].nunique()}")
log(f"  Customers:        {train['Customer_Name'].nunique()}")
log("")
for issue in issues:
    log(f"  [{issue['type']}] {issue['action']}")
