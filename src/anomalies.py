"""
Anomaly Detection & Analysis for MKS Hackathon.
Usage: python src/anomalies.py

Reads cleaned data from data/cleaned/
Writes anomaly register to reports/anomaly_register.csv
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

CLEAN_DIR = PROJECT_ROOT / 'data' / 'cleaned'
REPORT_DIR = PROJECT_ROOT / 'reports'


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


log("Loading cleaned data...")
train = pd.read_csv(CLEAN_DIR / 'train_clean.csv')
stockouts = pd.read_csv(CLEAN_DIR / 'inventory_stockouts_clean.csv')
events = pd.read_csv(CLEAN_DIR / 'events_calendar_clean.csv')
sku_master = pd.read_csv(CLEAN_DIR / 'sku_master_clean.csv')

anomaly_register = []


# ═══════════════════════════════════════════════════════════════════════
# 1. STATISTICAL OUTLIER DETECTION (IQR per SKU-Region)
# ═══════════════════════════════════════════════════════════════════════
log("1. Statistical outlier detection (IQR method)...")

for (sku, region), grp in train.groupby(['SKU_ID', 'Region']):
    if len(grp) < 6:
        continue
    q1 = grp['Units_Sold'].quantile(0.25)
    q3 = grp['Units_Sold'].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        continue
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    outliers = grp[(grp['Units_Sold'] < lower) | (grp['Units_Sold'] > upper)]
    for _, row in outliers.iterrows():
        anomaly_register.append({
            'Date': row['Date'],
            'SKU_ID': sku,
            'Region': region,
            'Units_Sold': row['Units_Sold'],
            'Anomaly_Type': 'Statistical Outlier (IQR)',
            'Lower_Bound': round(lower, 1),
            'Upper_Bound': round(upper, 1),
            'Reason': 'Outside 3×IQR range',
            'Action': 'Flag for review',
            'Category': 'Unknown'
        })

log(f"  Found {len(anomaly_register)} IQR outliers")


# ═══════════════════════════════════════════════════════════════════════
# 2. STRUCTURAL BREAKS (COVID, Chip Shortage, Inflation)
# ═══════════════════════════════════════════════════════════════════════
log("2. Detecting structural breaks...")

# Monthly totals
monthly_total = train.groupby('Date')['Units_Sold'].sum().reset_index()
monthly_total = monthly_total.sort_values('Date')
monthly_total['Pct_Change'] = monthly_total['Units_Sold'].pct_change()

# Large month-over-month drops
for _, row in monthly_total.iterrows():
    if pd.notna(row['Pct_Change']):
        if row['Pct_Change'] < -0.3:
            anomaly_register.append({
                'Date': row['Date'],
                'SKU_ID': 'ALL',
                'Region': 'ALL',
                'Units_Sold': row['Units_Sold'],
                'Anomaly_Type': 'Structural Break - Demand Drop',
                'Lower_Bound': None,
                'Upper_Bound': None,
                'Reason': f"{row['Pct_Change']*100:.1f}% month-over-month drop",
                'Action': 'Model as event (COVID/disruption)',
                'Category': 'Real Event'
            })
        elif row['Pct_Change'] > 0.5:
            anomaly_register.append({
                'Date': row['Date'],
                'SKU_ID': 'ALL',
                'Region': 'ALL',
                'Units_Sold': row['Units_Sold'],
                'Anomaly_Type': 'Structural Break - Demand Surge',
                'Lower_Bound': None,
                'Upper_Bound': None,
                'Reason': f"+{row['Pct_Change']*100:.1f}% month-over-month surge",
                'Action': 'Model as recovery/event',
                'Category': 'Real Event'
            })

log(f"  Total anomalies after structural breaks: {len(anomaly_register)}")


# ═══════════════════════════════════════════════════════════════════════
# 3. PRE-BUY BEFORE PRICE INCREASES
# ═══════════════════════════════════════════════════════════════════════
log("3. Detecting pre-buy patterns before price changes...")

price_events = events[events['Event_Type'].str.contains('Price', case=False, na=False)]
for _, evt in price_events.iterrows():
    start = evt['Start_Date']
    if not isinstance(start, str) or len(start) < 7:
        continue
    # Look at 1-2 months before the price change
    year, month = int(start[:4]), int(start[5:7])
    for offset in [1, 2]:
        pre_m = month - offset
        pre_y = year
        if pre_m <= 0:
            pre_m += 12
            pre_y -= 1
        pre_date = f"{pre_y}-{pre_m:02d}"
        # Check if demand in that month was above normal
        pre_demand = train[train['Date'] == pre_date]
        if len(pre_demand) == 0:
            continue
        total_pre = pre_demand['Units_Sold'].sum()
        # Compare with same month prior year
        compare_date = f"{pre_y-1}-{pre_m:02d}"
        compare_demand = train[train['Date'] == compare_date]['Units_Sold'].sum()
        if compare_demand > 0 and total_pre / compare_demand > 1.3:
            anomaly_register.append({
                'Date': pre_date,
                'SKU_ID': 'ALL',
                'Region': evt.get('Scope_Value', 'ALL'),
                'Units_Sold': total_pre,
                'Anomaly_Type': 'Pre-Buy Pattern',
                'Lower_Bound': None,
                'Upper_Bound': None,
                'Reason': f"Pre-buy before price change ({evt['Description'][:60]})",
                'Action': 'Model as event-driven demand',
                'Category': 'Real Event'
            })

log(f"  Total anomalies after pre-buy: {len(anomaly_register)}")


# ═══════════════════════════════════════════════════════════════════════
# 4. STOCKOUT PERIODS (constrained supply)
# ═══════════════════════════════════════════════════════════════════════
log("4. Flagging stockout periods...")

for _, row in stockouts.iterrows():
    anomaly_register.append({
        'Date': row['Date'],
        'SKU_ID': row['SKU_ID'],
        'Region': row['Region'],
        'Units_Sold': row.get('Shipped_Units', None),
        'Anomaly_Type': 'Supply Constraint / Stockout',
        'Lower_Bound': None,
        'Upper_Bound': row.get('Requested_Units', None),
        'Reason': f"Stockout: {row.get('Reason', 'Unknown')}; Fill rate: {row.get('Fill_Rate_Pct', '?')}%",
        'Action': 'Use Requested_Units as unconstrained demand',
        'Category': 'Real Event'
    })

log(f"  Total anomalies after stockouts: {len(anomaly_register)}")


# ═══════════════════════════════════════════════════════════════════════
# 5. LIFECYCLE VIOLATIONS
# ═══════════════════════════════════════════════════════════════════════
log("5. Checking lifecycle violations...")

# Sales after EOL
eol_skus = sku_master[sku_master['EOL_Date'].notna() & (sku_master['EOL_Date'] != '')][['SKU_ID', 'EOL_Date']]
if len(eol_skus) > 0:
    train_with_eol = train.merge(eol_skus, on='SKU_ID', how='inner', suffixes=('', '_master'))
    if 'EOL_Date_master' in train_with_eol.columns:
        after_eol = train_with_eol[train_with_eol['Date'] > train_with_eol['EOL_Date_master']]
    elif 'EOL_Date' in train_with_eol.columns:
        after_eol = train_with_eol[train_with_eol['Date'] > train_with_eol['EOL_Date']]
    else:
        after_eol = pd.DataFrame()
    
    for _, row in after_eol.iterrows():
        anomaly_register.append({
            'Date': row['Date'],
            'SKU_ID': row['SKU_ID'],
            'Region': row['Region'],
            'Units_Sold': row['Units_Sold'],
            'Anomaly_Type': 'Lifecycle Violation - After EOL',
            'Lower_Bound': None,
            'Upper_Bound': None,
            'Reason': 'Sales recorded after End-of-Life date',
            'Action': 'Investigate; may be aftermarket/spares',
            'Category': 'Error'
        })

log(f"  Total anomalies after lifecycle: {len(anomaly_register)}")


# ═══════════════════════════════════════════════════════════════════════
# 6. ZERO-DEMAND MONTHS ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
log("6. Analyzing zero-demand patterns...")

zero_months = train[train['Units_Sold'] == 0]
log(f"  Zero-demand rows: {len(zero_months):,} ({100*len(zero_months)/len(train):.1f}% of clean train)")


# ═══════════════════════════════════════════════════════════════════════
# 7. CUSTOMER SWITCHES
# ═══════════════════════════════════════════════════════════════════════
log("7. Detecting customer switches...")

sku_region_customers = train.groupby(['SKU_ID', 'Region', 'Date'])['Customer_Name'].first().reset_index()
sku_region_customers = sku_region_customers.sort_values(['SKU_ID', 'Region', 'Date'])

# Detect when customer changes for a SKU-Region
switches = []
for (sku, region), grp in sku_region_customers.groupby(['SKU_ID', 'Region']):
    customers = grp['Customer_Name'].values
    dates = grp['Date'].values
    for i in range(1, len(customers)):
        if customers[i] != customers[i-1] and pd.notna(customers[i]) and pd.notna(customers[i-1]):
            switches.append({
                'Date': dates[i],
                'SKU_ID': sku,
                'Region': region,
                'From_Customer': customers[i-1],
                'To_Customer': customers[i],
            })
            anomaly_register.append({
                'Date': dates[i],
                'SKU_ID': sku,
                'Region': region,
                'Units_Sold': None,
                'Anomaly_Type': 'Customer Switch',
                'Lower_Bound': None,
                'Upper_Bound': None,
                'Reason': f"Customer changed: {customers[i-1]} → {customers[i]}",
                'Action': 'May affect demand pattern',
                'Category': 'Real Event'
            })

log(f"  Customer switches found: {len(switches)}")
log(f"  Total anomalies: {len(anomaly_register)}")


# ═══════════════════════════════════════════════════════════════════════
# 8. SAVE ANOMALY REGISTER
# ═══════════════════════════════════════════════════════════════════════
log("Saving anomaly register...")

anomaly_df = pd.DataFrame(anomaly_register)
anomaly_df.to_csv(REPORT_DIR / 'anomaly_register.csv', index=False)

log(f"  Saved {len(anomaly_df)} anomalies to {REPORT_DIR / 'anomaly_register.csv'}")

# Summary
log("=" * 60)
log("ANOMALY DETECTION COMPLETE")
log("=" * 60)
type_counts = anomaly_df['Anomaly_Type'].value_counts()
for atype, count in type_counts.items():
    log(f"  {atype}: {count}")
