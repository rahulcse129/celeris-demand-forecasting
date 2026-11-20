"""
Shared utility functions for the MKS Hackathon project.
"""
import re
from difflib import SequenceMatcher

# ── Region Mapping ──────────────────────────────────────────────────────
REGION_MAP = {
    'north america': 'North America',
    'n. america': 'North America',
    'na': 'North America',
    'europe': 'Europe',
    'eu': 'Europe',
    'asia pacific': 'Asia Pacific',
    'asia-pacific': 'Asia Pacific',
    'apac': 'Asia Pacific',
    'india': 'India',
    'latin america': 'Latin America',
    'latin-america': 'Latin America',
    'latam': 'Latin America',
    'middle east & africa': 'Middle East & Africa',
    'middle east and africa': 'Middle East & Africa',
    'me&a': 'Middle East & Africa',
    'mea': 'Middle East & Africa',
}

CLEAN_REGIONS = ['North America', 'Europe', 'Asia Pacific', 'India', 'Latin America', 'Middle East & Africa']

# ── Sales Channel Mapping ──────────────────────────────────────────────
CHANNEL_MAP = {
    'direct oem': 'Direct OEM',
    'oem direct': 'Direct OEM',
    'direct-oem': 'Direct OEM',
    'tier-1 distributor': 'Tier-1 Distributor',
    'tier 1 distributor': 'Tier-1 Distributor',
    't1 distributor': 'Tier-1 Distributor',
    'aftermarket': 'Aftermarket',
    'after-market': 'Aftermarket',
    'export': 'Export',
}

CLEAN_CHANNELS = ['Direct OEM', 'Tier-1 Distributor', 'Aftermarket', 'Export']

# ── Canonical Customer Names ───────────────────────────────────────────
CANONICAL_CUSTOMERS = [
    'Apex Tier One',
    'Atlas Mideast Components',
    'Crestline Manufacturing',
    'Falcon Gulf Motors',
    'Helios Auto Parts',
    'Kestrel Mobility',
    'Meridian Assembly',
    'Nexon Motor Group',
    'Orion Driveline',
    'Quantum Drive Systems',
    'Solaris Automotive',
    'Vantara Auto',
]

# ── Product Family Mapping ─────────────────────────────────────────────
FAMILY_PREFIX_MAP = {
    'BSS': 'Brake & Safety Systems',
    'CSS': 'Chassis & Structural',
    'ELA': 'Electrical Assemblies',
    'HYD': 'Hydraulic Systems',
    'PTC': 'Powertrain Components',
    'PMP': 'Precision Machined Parts',
    'THM': 'Thermal Management',
}


def standardize_region(val):
    """Map any region variant to its canonical form."""
    if not isinstance(val, str) or val.strip() == '':
        return None
    return REGION_MAP.get(val.strip().lower(), val.strip())


def standardize_channel(val):
    """Map any sales-channel variant to its canonical form."""
    if not isinstance(val, str) or val.strip() == '':
        return None
    return CHANNEL_MAP.get(val.strip().lower(), val.strip())


def standardize_customer(val):
    """
    Map a (possibly misspelled) customer name to its canonical form 
    using fuzzy matching.
    """
    if not isinstance(val, str) or val.strip() == '':
        return None
    cleaned = val.strip()
    # Quick exact match (case-insensitive, ignore Inc. suffix)
    cleaned_lower = re.sub(r'\s+inc\.?$', '', cleaned, flags=re.IGNORECASE).strip()
    cleaned_lower = re.sub(r'\s+', ' ', cleaned_lower)
    for canon in CANONICAL_CUSTOMERS:
        if cleaned_lower.lower() == canon.lower():
            return canon
    # Fuzzy match
    best_score = 0
    best_match = cleaned
    for canon in CANONICAL_CUSTOMERS:
        score = SequenceMatcher(None, cleaned_lower.lower(), canon.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = canon
    if best_score >= 0.65:
        return best_match
    return cleaned  # Fallback


def parse_date_to_yyyymm(val):
    """
    Parse various date formats to YYYY-MM string.
    Handles: YYYY-MM, MM-YYYY, Mon-YYYY, YYYY-MM-DD, etc.
    """
    import calendar
    if not isinstance(val, str) or val.strip() == '':
        return None
    val = val.strip()
    
    # Already YYYY-MM
    m = re.match(r'^(\d{4})-(\d{1,2})$', val)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    
    # YYYY-MM-DD
    m = re.match(r'^(\d{4})-(\d{1,2})-\d{1,2}$', val)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    
    # MM-YYYY (with dash)
    m = re.match(r'^(\d{1,2})-(\d{4})$', val)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    
    # MM/YYYY (with slash)
    m = re.match(r'^(\d{1,2})/(\d{4})$', val)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    
    # YYYY/MM (with slash)
    m = re.match(r'^(\d{4})/(\d{1,2})$', val)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    
    # YYYY/MM/DD
    m = re.match(r'^(\d{4})/(\d{1,2})/\d{1,2}$', val)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}"
    
    # MM/DD/YYYY
    m = re.match(r'^(\d{1,2})/\d{1,2}/(\d{4})$', val)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    
    # Mon-YYYY (e.g., Sep-2026)
    month_abbrs = {v.lower(): k for k, v in enumerate(calendar.month_abbr) if v}
    m = re.match(r'^([A-Za-z]{3,9})[- /](\d{4})$', val)
    if m:
        mon_str = m.group(1).lower()
        # Try abbreviation first
        mon = month_abbrs.get(mon_str[:3])
        if mon:
            return f"{m.group(2)}-{mon:02d}"
    
    # YYYY-Mon
    m = re.match(r'^(\d{4})[- /]([A-Za-z]{3,9})$', val)
    if m:
        mon_str = m.group(2).lower()
        mon = month_abbrs.get(mon_str[:3])
        if mon:
            return f"{m.group(1)}-{mon:02d}"
    
    return val  # Return as-is if unrecognized


def infer_family_from_sku(sku_id):
    """Infer Product_Family from the SKU_ID prefix."""
    if not isinstance(sku_id, str):
        return None
    prefix = sku_id.split('-')[0].upper()
    return FAMILY_PREFIX_MAP.get(prefix)
