"""
Feature Engineering for MKS Hackathon.
Usage: python src/features.py

Reads cleaned data from data/cleaned/
Writes feature-engineered data to data/features/
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = PROJECT_ROOT / 'data' / 'cleaned'
FEATURE_DIR = PROJECT_ROOT / 'data' / 'features'
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


log("Loading cleaned data...")
train = pd.read_csv(CLEAN_DIR / 'train_clean.csv', low_memory=False)
test = pd.read_csv(CLEAN_DIR / 'test_clean.csv', low_memory=False)
macro = pd.read_csv(CLEAN_DIR / 'macro_indicators_clean.csv', low_memory=False)
events = pd.read_csv(CLEAN_DIR / 'events_calendar_clean.csv', low_memory=False)

# Track exact original test row index
train['is_train'] = 1
train['test_row_id'] = -1

test['is_train'] = 0
test['test_row_id'] = np.arange(len(test))
test['Units_Sold'] = np.nan
test['Revenue_USD'] = np.nan
test['Unit_Price_USD'] = np.nan
test['Unconstrained_Demand'] = np.nan

# Align columns perfectly without duplication
df = pd.concat([train, test[train.columns]], ignore_index=True)
df = df.sort_values(['SKU_ID', 'Region', 'Date']).reset_index(drop=True)
log(f"  Combined dataset: {len(df):,} rows (train: {len(train):,}, test: {len(test):,})")

# ═══════════════════════════════════════════════════════════════════════
# 1. CALENDAR FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("1. Generating calendar features...")

df['Year'] = df['Date'].str[:4].astype(int)
df['Month'] = df['Date'].str[5:7].astype(int)
df['Date_Num'] = df['Year'] * 12 + df['Month']
df['Quarter'] = (df['Month'] - 1) // 3 + 1
df['Is_Q4'] = (df['Quarter'] == 4).astype(int)
df['Is_December'] = (df['Month'] == 12).astype(int)
df['Is_January'] = (df['Month'] == 1).astype(int)
df['Half_Year'] = (df['Month'] > 6).astype(int)

# Cyclical encoding of month
df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
df['Days_In_Month'] = df['Month'].map(days_in_month)

# ═══════════════════════════════════════════════════════════════════════
# 2. HISTORICAL SUMMARY STATISTICS (per SKU-Region & per SKU)
# ═══════════════════════════════════════════════════════════════════════
log("2. Computing historical summary statistics...")

train_only = df[df['is_train'] == 1].copy()

# SKU-Region stats using standard list aggregation
sku_reg_stats = train_only.groupby(['SKU_ID', 'Region'])['Units_Sold'].agg(['mean', 'median', 'std', 'max', 'min', 'count']).reset_index()
sku_reg_stats.columns = ['SKU_ID', 'Region', 'Hist_Mean_Demand', 'Hist_Median_Demand', 'Hist_Std_Demand', 'Hist_Max_Demand', 'Hist_Min_Demand', 'Hist_Count']

# SKU overall stats
sku_stats = train_only.groupby('SKU_ID')['Units_Sold'].agg(['mean', 'median']).reset_index()
sku_stats.columns = ['SKU_ID', 'SKU_Overall_Mean', 'SKU_Overall_Median']

# Product Family stats
fam_stats = train_only.groupby('Product_Family')['Units_Sold'].agg(['mean']).reset_index()
fam_stats.columns = ['Product_Family', 'Family_Mean_Demand']

# Recent 12-month mean (as of end of train, 2025-11 to 2026-10)
recent_train = train_only[train_only['Date'] >= '2025-11']
recent_sku_reg = recent_train.groupby(['SKU_ID', 'Region'])['Units_Sold'].agg(['mean']).reset_index()
recent_sku_reg.columns = ['SKU_ID', 'Region', 'Recent_12m_Mean']

df = df.merge(sku_reg_stats, on=['SKU_ID', 'Region'], how='left')
df = df.merge(sku_stats, on=['SKU_ID'], how='left')
df = df.merge(fam_stats, on=['Product_Family'], how='left')
df = df.merge(recent_sku_reg, on=['SKU_ID', 'Region'], how='left')

# ── Expert: Inherit historical demand for new launches via Replaces_SKU ──
log("  Applying Replaces_SKU demand inheritance for new launches...")
pred_sku_stats = sku_stats.set_index('SKU_ID').to_dict('index')
pred_reg_stats = sku_reg_stats.set_index(['SKU_ID', 'Region']).to_dict('index')

rep_mask = df['Replaces_SKU'].notna() & (df['Replaces_SKU'].str.strip() != '') & (df['Hist_Mean_Demand'].isna() | (df['Hist_Mean_Demand'] == 0))
for idx in df[rep_mask].index:
    rep_sku = str(df.loc[idx, 'Replaces_SKU']).strip()
    reg = df.loc[idx, 'Region']
    if (rep_sku, reg) in pred_reg_stats:
        rep_info = pred_reg_stats[(rep_sku, reg)]
        df.loc[idx, 'Hist_Mean_Demand'] = rep_info['Hist_Mean_Demand']
        df.loc[idx, 'Hist_Median_Demand'] = rep_info['Hist_Median_Demand']
        df.loc[idx, 'Hist_Std_Demand'] = rep_info['Hist_Std_Demand']
        df.loc[idx, 'Recent_12m_Mean'] = rep_info['Hist_Mean_Demand']
    elif rep_sku in pred_sku_stats:
        rep_info = pred_sku_stats[rep_sku]
        df.loc[idx, 'Hist_Mean_Demand'] = rep_info['SKU_Overall_Mean']
        df.loc[idx, 'Hist_Median_Demand'] = rep_info['SKU_Overall_Median']
        df.loc[idx, 'Recent_12m_Mean'] = rep_info['SKU_Overall_Mean']

# Impute remaining missing values with broader hierarchal fallbacks
df['Recent_12m_Mean'] = df['Recent_12m_Mean'].fillna(df['Hist_Mean_Demand']).fillna(df['SKU_Overall_Mean']).fillna(0)
df['Hist_Mean_Demand'] = df['Hist_Mean_Demand'].fillna(df['SKU_Overall_Mean']).fillna(0)
df['Hist_Median_Demand'] = df['Hist_Median_Demand'].fillna(df['SKU_Overall_Median']).fillna(0)
df['Hist_Std_Demand'] = df['Hist_Std_Demand'].fillna(0)
df['Hist_Max_Demand'] = df['Hist_Max_Demand'].fillna(df['Hist_Mean_Demand'])
df['Hist_Min_Demand'] = df['Hist_Min_Demand'].fillna(0)

# ── Intermittent Demand: Croston-SBA Rate ──
log("  Computing Croston-SBA intermittent demand rates...")
croston_dict = {}
for (sku, reg), grp in train_only.groupby(['SKU_ID', 'Region']):
    vals = grp.sort_values('Date_Num')['Units_Sold'].values
    non_zeros = np.where(vals > 0)[0]
    if len(non_zeros) == 0:
        croston_dict[(sku, reg)] = 0.0
    else:
        alpha = 0.1
        z = float(vals[non_zeros[0]])
        p = 1.0
        last_idx = non_zeros[0]
        for i_nz in non_zeros[1:]:
            interval = float(i_nz - last_idx)
            z = alpha * float(vals[i_nz]) + (1.0 - alpha) * z
            p = alpha * interval + (1.0 - alpha) * p
            last_idx = i_nz
        rate = (1.0 - alpha / 2.0) * (z / max(p, 0.1))
        croston_dict[(sku, reg)] = float(rate)

df['Croston_SBA_Rate'] = [croston_dict.get((s, r), np.nan) for s, r in zip(df['SKU_ID'], df['Region'])]
# For new launches, inherit predecessor Croston rate if available
for idx in df[df['Croston_SBA_Rate'].isna() & df['Replaces_SKU'].notna()].index:
    rep_sku = str(df.loc[idx, 'Replaces_SKU']).strip()
    reg = df.loc[idx, 'Region']
    if (rep_sku, reg) in croston_dict:
        df.loc[idx, 'Croston_SBA_Rate'] = croston_dict[(rep_sku, reg)]

df['Croston_SBA_Rate'] = df['Croston_SBA_Rate'].fillna(df['Recent_12m_Mean'])

# ═══════════════════════════════════════════════════════════════════════
# 3. LAG FEATURES & ROLLING STATS
# ═══════════════════════════════════════════════════════════════════════
log("3. Computing lag and rolling features...")

df['Date_Num'] = df['Year'] * 12 + df['Month']

# Fast lookup dict from train
train_lookup = train_only.set_index(['SKU_ID', 'Region', 'Date_Num'])['Units_Sold'].to_dict()

for lag in [1, 2, 3, 6, 12]:
    log(f"  Computing lag-{lag}...")
    lag_keys = list(zip(df['SKU_ID'], df['Region'], df['Date_Num'] - lag))
    df[f'Lag_{lag}'] = [train_lookup.get(k, np.nan) for k in lag_keys]
    # Check Replaces_SKU inheritance if still NaN
    nan_rep_mask = df[f'Lag_{lag}'].isna() & df['Replaces_SKU'].notna()
    if nan_rep_mask.sum() > 0:
        rep_keys = list(zip(df.loc[nan_rep_mask, 'Replaces_SKU'], df.loc[nan_rep_mask, 'Region'], df.loc[nan_rep_mask, 'Date_Num'] - lag))
        df.loc[nan_rep_mask, f'Lag_{lag}'] = [train_lookup.get(k, np.nan) for k in rep_keys]
    df[f'Lag_{lag}'] = df[f'Lag_{lag}'].fillna(df['Recent_12m_Mean'])

log("  Computing rolling stats for train...")
for window in [3, 6, 12]:
    df[f'Rolling_{window}m_mean'] = df['Lag_1']
    train_idx = df[df['is_train'] == 1].index
    lag_grouped = df.loc[train_idx].groupby(['SKU_ID', 'Region'])['Lag_1']
    df.loc[train_idx, f'Rolling_{window}m_mean'] = lag_grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
    df.loc[train_idx, f'Rolling_{window}m_std'] = lag_grouped.transform(lambda s: s.rolling(window, min_periods=1).std()).fillna(0)
    df.loc[train_idx, f'Rolling_{window}m_max'] = lag_grouped.transform(lambda s: s.rolling(window, min_periods=1).max())
    df.loc[train_idx, f'Rolling_{window}m_min'] = lag_grouped.transform(lambda s: s.rolling(window, min_periods=1).min())

    test_idx = df[df['is_train'] == 0].index
    df.loc[test_idx, f'Rolling_{window}m_mean'] = df.loc[test_idx, 'Recent_12m_Mean']
    df.loc[test_idx, f'Rolling_{window}m_std'] = df.loc[test_idx, 'Hist_Std_Demand']
    df.loc[test_idx, f'Rolling_{window}m_max'] = df.loc[test_idx, 'Hist_Max_Demand']
    df.loc[test_idx, f'Rolling_{window}m_min'] = df.loc[test_idx, 'Hist_Min_Demand']

# ═══════════════════════════════════════════════════════════════════════
# 4. INTERMITTENCY CLASSIFICATION (Syntetos-Boylan)
# ═══════════════════════════════════════════════════════════════════════
log("4. Intermittency classification (ADI x CV2)...")

intermittency = []
for (sku, region), grp in train_only.groupby(['SKU_ID', 'Region']):
    demands = grp['Units_Sold'].values
    non_zero = demands[demands > 0]
    
    if len(non_zero) == 0:
        adi = 999.0
        cv2 = 0.0
    else:
        nz_idx = np.where(demands > 0)[0]
        adi = float(np.diff(nz_idx).mean()) if len(nz_idx) > 1 else float(len(demands))
        mean_nz = non_zero.mean()
        std_nz = non_zero.std()
        cv2 = float((std_nz / mean_nz) ** 2) if mean_nz > 0 else 0.0
    
    if adi < 1.32 and cv2 < 0.49:
        dclass = 'Smooth'
    elif adi < 1.32 and cv2 >= 0.49:
        dclass = 'Erratic'
    elif adi >= 1.32 and cv2 < 0.49:
        dclass = 'Intermittent'
    else:
        dclass = 'Lumpy'
    
    intermittency.append({
        'SKU_ID': sku,
        'Region': region,
        'ADI': round(adi, 2),
        'CV2': round(cv2, 4),
        'Demand_Class': dclass,
        'Zero_Pct': round(100.0 * (1.0 - len(non_zero) / max(len(demands), 1)), 1)
    })

intermittency_df = pd.DataFrame(intermittency)
df = df.merge(intermittency_df, on=['SKU_ID', 'Region'], how='left')
df['ADI'] = df['ADI'].fillna(1.0)
df['CV2'] = df['CV2'].fillna(0.0)
df['Demand_Class'] = df['Demand_Class'].fillna('Smooth')
df['Zero_Pct'] = df['Zero_Pct'].fillna(0.0)

log(f"  Demand classes: {df['Demand_Class'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════════════════════
# 5. MACRO INDICATOR FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("5. Merging macro indicators...")

macro_cols = ['ISM_Manufacturing_PMI', 'Auto_Production_YoY_Pct', 'Vehicle_Inventory_Days',
              'Raw_Material_Cost_Index', 'New_Vehicle_Orders_Index', 'Industrial_Util_Rate_Pct',
              'Fuel_Price_Index', 'EV_Adoption_Index']

macro_merge = macro[['Date'] + macro_cols].copy()
for col in macro_cols:
    macro_merge[col] = pd.to_numeric(macro_merge[col], errors='coerce')

df = df.merge(macro_merge, on='Date', how='left')

# 1-month lag for macro indicators
macro_lag = macro_merge.copy()
macro_lag['Date_Num'] = macro_lag['Date'].str[:4].astype(int) * 12 + macro_lag['Date'].str[5:7].astype(int) + 1
macro_lag = macro_lag.rename(columns={c: f'{c}_Lag1' for c in macro_cols})
df = df.merge(macro_lag[['Date_Num'] + [f'{c}_Lag1' for c in macro_cols]], on='Date_Num', how='left')

for c in macro_cols:
    df[c] = df[c].ffill().bfill().fillna(0)
    df[f'{c}_Lag1'] = df[f'{c}_Lag1'].ffill().bfill().fillna(0)

# ═══════════════════════════════════════════════════════════════════════
# 6. EVENT FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("6. Generating event features...")

df['Has_Price_Change'] = 0
df['Has_Disruption'] = 0
df['Has_Holiday'] = 0
df['Has_Supply_Constraint'] = 0

for _, evt in events.iterrows():
    start = str(evt.get('Start_Date', ''))[:7]
    end = str(evt.get('End_Date', ''))[:7] if pd.notna(evt.get('End_Date')) and str(evt.get('End_Date')).strip() != '' else start
    evt_type = str(evt.get('Event_Type', '')).lower()
    
    if len(start) == 7 and len(end) == 7:
        mask = (df['Date'] >= start) & (df['Date'] <= end)
        scope_type = str(evt.get('Scope_Type', '')).lower()
        scope_val = str(evt.get('Scope_Value', ''))
        
        if scope_type == 'region' and scope_val != 'All':
            mask = mask & (df['Region'] == scope_val)
        elif scope_type == 'product_family':
            mask = mask & (df['Product_Family'] == scope_val)
        
        if 'price' in evt_type:
            df.loc[mask, 'Has_Price_Change'] = 1
        elif 'disruption' in evt_type:
            df.loc[mask, 'Has_Disruption'] = 1
        elif 'holiday' in evt_type:
            df.loc[mask, 'Has_Holiday'] = 1
        elif 'supply' in evt_type or 'constraint' in evt_type:
            df.loc[mask, 'Has_Supply_Constraint'] = 1

# ═══════════════════════════════════════════════════════════════════════
# 7. LIFECYCLE FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("7. Generating lifecycle features...")

def parse_num(dt_str):
    if pd.isna(dt_str) or len(str(dt_str)) < 7:
        return np.nan
    s = str(dt_str)
    try:
        return int(s[:4]) * 12 + int(s[5:7])
    except:
        return np.nan

df['Launch_Date_Num'] = df['Launch_Date'].apply(parse_num)
df['EOL_Date_Num'] = df['EOL_Date'].apply(parse_num)

df['Months_Since_Launch'] = (df['Date_Num'] - df['Launch_Date_Num']).fillna(999)
df['Months_To_EOL'] = (df['EOL_Date_Num'] - df['Date_Num']).fillna(999)

df['Is_New_Launch'] = ((df['Months_Since_Launch'] >= 0) & (df['Months_Since_Launch'] <= 6)).astype(int)
df['Is_Near_EOL'] = ((df['Months_To_EOL'] >= 0) & (df['Months_To_EOL'] <= 6)).astype(int)
df['Is_Discontinued'] = (df['Lifecycle_Status'] == 'Discontinued').astype(int)

# ═══════════════════════════════════════════════════════════════════════
# 8. EV TRANSITION IMPACT (Expert feature)
# ═══════════════════════════════════════════════════════════════════════
log("8. Generating EV transition features...")

df['EV_Impact'] = 0.0
ev_idx = df['EV_Adoption_Index'].fillna(0)
df.loc[df['Powertrain_Application'] == 'EV', 'EV_Impact'] = ev_idx / 100.0
df.loc[df['Powertrain_Application'] == 'ICE', 'EV_Impact'] = -ev_idx / 100.0

# ═══════════════════════════════════════════════════════════════════════
# 9. ENCODING CATEGORICAL FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("9. Encoding categoricals...")

cat_cols = ['Product_Family', 'Business_Unit', 'Region', 'Sales_Channel',
            'Customer_Name', 'Demand_Class', 'Powertrain_Application', 'Size_Class']

for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)
        le = LabelEncoder()
        df[f'{col}_Encoded'] = le.fit_transform(df[col])

df.drop(columns=['Launch_Date_Num', 'EOL_Date_Num'], inplace=True, errors='ignore')

# ═══════════════════════════════════════════════════════════════════════
# 10. SPLIT AND SAVE FEATURES
# ═══════════════════════════════════════════════════════════════════════
log("10. Saving train and test feature matrices...")

train_out = df[df['is_train'] == 1].drop(columns=['is_train', 'test_row_id']).reset_index(drop=True)

# Ensure test_features matches the exact row order of test.csv!
test_out = df[df['is_train'] == 0].sort_values('test_row_id').reset_index(drop=True)
test_out = test_out.drop(columns=['is_train', 'test_row_id', 'Units_Sold', 'Revenue_USD', 'Unit_Price_USD', 'Unconstrained_Demand'], errors='ignore')

train_out.to_csv(FEATURE_DIR / 'train_features.csv', index=False)
test_out.to_csv(FEATURE_DIR / 'test_features.csv', index=False)
intermittency_df.to_csv(FEATURE_DIR / 'intermittency_classes.csv', index=False)

log(f"  train_features saved: {train_out.shape[0]:,} rows x {train_out.shape[1]} columns")
log(f"  test_features saved:  {test_out.shape[0]:,} rows x {test_out.shape[1]} columns")
log("Feature engineering complete!")
