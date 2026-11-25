"""
Shared utilities: canonical label maps, normalisers and date parsing.
"""
import re
import calendar
from difflib import SequenceMatcher

import numpy as np

CLEAN_REGIONS = ['North America', 'Europe', 'Asia Pacific', 'India', 'Latin America', 'Middle East & Africa']
CLEAN_CHANNELS = ['Direct OEM', 'Tier-1 Distributor', 'Aftermarket', 'Export']

REGION_MAP = {
    'north america': 'North America', 'n. america': 'North America', 'na': 'North America',
    'europe': 'Europe', 'eu': 'Europe',
    'asia pacific': 'Asia Pacific', 'asia-pacific': 'Asia Pacific', 'apac': 'Asia Pacific',
    'india': 'India',
    'latin america': 'Latin America', 'latin-america': 'Latin America', 'latam': 'Latin America',
    'middle east & africa': 'Middle East & Africa', 'middle east and africa': 'Middle East & Africa',
    'me&a': 'Middle East & Africa', 'mea': 'Middle East & Africa',
}

CHANNEL_MAP = {
    'direct oem': 'Direct OEM', 'oem direct': 'Direct OEM', 'direct-oem': 'Direct OEM',
    'tier-1 distributor': 'Tier-1 Distributor', 'tier 1 distributor': 'Tier-1 Distributor',
    't1 distributor': 'Tier-1 Distributor',
    'aftermarket': 'Aftermarket', 'after-market': 'Aftermarket',
    'export': 'Export',
}

CANONICAL_CUSTOMERS = [
    'Apex Tier One', 'Atlas Mideast Components', 'Crestline Manufacturing', 'Falcon Gulf Motors',
    'Helios Auto Parts', 'Kestrel Mobility', 'Meridian Assembly', 'Nexon Motor Group',
    'Orion Driveline', 'Quantum Drive Systems', 'Solaris Automotive', 'Vantara Auto',
]

FAMILY_PREFIX_MAP = {
    'BSS': 'Brake & Safety Systems',
    'CSS': 'Chassis & Structural',
    'ELA': 'Electrical Assemblies',
    'HYD': 'Hydraulic Systems',
    'PTC': 'Powertrain Components',
    'PMP': 'Precision Machined Parts',
    'THM': 'Thermal Management',
}

_MONTHS = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if v}


def _key(val):
    return re.sub(r'\s+', ' ', str(val)).strip().lower()


def standardize_region(val):
    if not isinstance(val, str) or not val.strip():
        return None
    return REGION_MAP.get(_key(val))


def standardize_channel(val):
    if not isinstance(val, str) or not val.strip():
        return None
    return CHANNEL_MAP.get(_key(val))


_CUST_LOWER = {c.lower(): c for c in CANONICAL_CUSTOMERS}


def standardize_customer(val):
    """Case/whitespace/'Inc.' normalisation, then fuzzy match for typos (score >= 0.8)."""
    if not isinstance(val, str) or not val.strip():
        return None
    k = re.sub(r'[\s,]+inc\.?$', '', _key(val)).strip()
    if k in _CUST_LOWER:
        return _CUST_LOWER[k]
    best, score = None, 0.0
    for low, canon in _CUST_LOWER.items():
        s = SequenceMatcher(None, k, low).ratio()
        if s > score:
            best, score = canon, s
    return best if score >= 0.8 else None


def standardize_sku(val):
    """' pmp_23 ' / 'PMP023' / 'CSS-31' -> 'PMP-023'."""
    if not isinstance(val, str):
        return None
    s = val.strip().upper().replace('_', '-').replace(' ', '')
    m = re.match(r'^([A-Z]{3})-?(\d{1,3})$', s)
    return f"{m.group(1)}-{int(m.group(2)):03d}" if m else None


def parse_date_to_yyyymm(val):
    """Parse YYYY-MM, YYYY-MM-DD, YYYY/MM, MM/YYYY, MM-YYYY, Mon-YYYY, Mon YYYY -> 'YYYY-MM'."""
    if not isinstance(val, str) or not val.strip():
        return None
    v = val.strip()
    for pat, yi, mi in [(r'^(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?$', 1, 2),
                        (r'^(\d{1,2})[-/](\d{4})$', 2, 1),
                        (r'^(\d{1,2})/\d{1,2}/(\d{4})$', 2, 1)]:
        m = re.match(pat, v)
        if m:
            y, mo = int(m.group(yi)), int(m.group(mi))
            return f"{y:04d}-{mo:02d}" if 1 <= mo <= 12 else None
    m = re.match(r'^([A-Za-z]{3,9})[- /](\d{4})$', v)
    if m and m.group(1)[:3].lower() in _MONTHS:
        return f"{m.group(2)}-{_MONTHS[m.group(1)[:3].lower()]:02d}"
    m = re.match(r'^(\d{4})[- /]([A-Za-z]{3,9})$', v)
    if m and m.group(2)[:3].lower() in _MONTHS:
        return f"{m.group(1)}-{_MONTHS[m.group(2)[:3].lower()]:02d}"
    return None


def infer_family_from_sku(sku_id):
    if not isinstance(sku_id, str):
        return None
    return FAMILY_PREFIX_MAP.get(sku_id.split('-')[0].upper())


def ym_to_int(s):
    """'YYYY-MM' -> months since year 0 (vectorised over pandas Series)."""
    return s.str[:4].astype(int) * 12 + s.str[5:7].astype(int) - 1


def int_to_ym(i):
    i = np.asarray(i)
    return np.char.add(np.char.add((i // 12).astype(str), '-'), np.char.zfill((i % 12 + 1).astype(str), 2))


def wape(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y).sum()
    return float(np.abs(y - p).sum() / d) if d else 0.0


def bias(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    d = np.abs(y).sum()
    return float((p - y).sum() / d) if d else 0.0
