"""
Normalization module.

Maps messy, alias-riddled column names to a standard schema,
then cleans and enriches field values (vendor, amount, date).

Standard fields produced:
  vendor            → raw vendor string
  vendor_normalized → lower-cased, suffix-stripped, whitespace-collapsed
  description       → raw text
  amount            → raw amount string
  amount_numeric    → float or None
  date              → raw date string
  date_parsed       → datetime or None
  end_date          → raw end date string (if present)
  department        → raw department string
  contract_id       → raw contract ID (if present)
  record_id         → stable integer row index (0-based)
"""
import re
import pandas as pd
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Field alias tables
# ---------------------------------------------------------------------------

FIELD_ALIASES: dict[str, list[str]] = {
    "vendor": [
        "vendor", "vendor_name", "supplier", "contractor", "company",
        "firm", "awardee", "prime_contractor", "awarded_to", "recipient",
    ],
    "description": [
        "description", "desc", "project_description", "scope",
        "work_description", "contract_description", "title",
        "project_title", "service_description", "deliverable",
        "statement_of_work", "sow", "purpose", "subject",
    ],
    "amount": [
        "amount", "contract_amount", "award_amount", "total_amount",
        "value", "cost", "price", "obligation", "total_value",
        "contract_value", "funded_amount", "base_amount", "total_obligated",
    ],
    "date": [
        "date", "award_date", "start_date", "contract_date",
        "effective_date", "period_of_performance_start",
        "signed_date", "execution_date", "base_date",
    ],
    "department": [
        "department", "dept", "agency", "office", "ministry",
        "division", "bureau", "awarding_agency", "contracting_agency",
        "funding_agency",
    ],
    "contract_id": [
        "contract_id", "contract_number", "id", "contract_no",
        "award_id", "reference", "piid", "order_number", "solicitation_id",
    ],
    "end_date": [
        "end_date", "period_of_performance_end", "expiry_date",
        "completion_date", "contract_end", "pop_end",
    ],
}

# Corporate-suffix noise to strip during vendor normalization
_CORP_SUFFIXES = [
    r"\bllc\b", r"\binc\.?\b", r"\bcorp\.?\b", r"\bltd\.?\b",
    r"\bco\.?\b", r"\bcompany\b", r"\bgroup\b", r"\bassociates\b",
    r"\bservices\b", r"\bsolutions\b", r"\benterprises\b",
    r"\bglobal\b", r"\bnational\b", r"\binternational\b", r"\bfederal\b",
    r"\bconsulting\b", r"\bpartners\b", r"\bsystems\b", r"\btechnologies\b",
    r"\btechnology\b",
]

_DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
    "%b %d, %Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
    "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%Y%m%d", "%m/%d/%y",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a raw DataFrame (column names may be anything) and return a
    normalised DataFrame with standard column names and cleaned values.
    """
    df = df.copy()

    # --- Step 1: lowercase all column names ---
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # --- Step 2: map aliases to standard names ---
    rename_map: dict[str, str] = {}
    claimed: set[str] = set()
    for col in df.columns:
        for standard, aliases in FIELD_ALIASES.items():
            if col in aliases and standard not in claimed:
                rename_map[col] = standard
                claimed.add(standard)
                break
    df.rename(columns=rename_map, inplace=True)

    # --- Step 3: ensure required columns exist ---
    if "vendor" not in df.columns:
        df["vendor"] = None
    if "description" not in df.columns:
        # Promote raw_text if it's the only text column
        text_cols = [c for c in df.columns if "text" in c or "raw" in c]
        df["description"] = df[text_cols[0]] if text_cols else ""
    if "amount" not in df.columns:
        df["amount"] = None
    if "date" not in df.columns:
        df["date"] = None

    # --- Step 4: clean and enrich values ---
    df["vendor_normalized"] = df["vendor"].apply(_normalize_vendor)
    df["amount_numeric"]    = df["amount"].apply(_parse_amount)
    df["date_parsed"]       = df["date"].apply(_parse_date)
    df["description"]       = df["description"].fillna("").astype(str).str.strip()

    # --- Step 5: stable record index ---
    df = df.reset_index(drop=True)
    df["record_id"] = df.index

    return df


# ---------------------------------------------------------------------------
# Value cleaners (also imported by scorer.py for consistency)
# ---------------------------------------------------------------------------

def normalize_vendor(name) -> str:
    return _normalize_vendor(name)


def _normalize_vendor(name) -> str:
    if pd.isna(name) or str(name).strip() in ("", "None", "N/A", "n/a"):
        return "unknown"
    name = str(name).lower().strip()
    for suffix in _CORP_SUFFIXES:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name if name else "unknown"


def _parse_amount(value) -> Optional[float]:
    if pd.isna(value) or str(value).strip() in ("", "None", "N/A", "n/a", "-"):
        return None
    s = re.sub(r"[$£€¥₹,\s]", "", str(value).strip())
    # Handle shorthand: 125K, 2.5M, 1.1B
    multipliers = {"k": 1e3, "m": 1e6, "b": 1e9}
    if s and s[-1].lower() in multipliers:
        try:
            return float(s[:-1]) * multipliers[s[-1].lower()]
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return None


def _parse_date(value) -> Optional[datetime]:
    if pd.isna(value) or str(value).strip() in ("", "None", "N/A", "n/a"):
        return None
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
