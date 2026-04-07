"""
Normalization module.
Maps messy column names to standard fields and cleans values.
"""
import re
import pandas as pd
from datetime import datetime
from typing import Optional


# Map standard field names to their common aliases
FIELD_ALIASES: dict[str, list[str]] = {
    "vendor": [
        "vendor", "vendor_name", "supplier", "contractor", "company",
        "firm", "awardee", "prime_contractor",
    ],
    "description": [
        "description", "desc", "project_description", "scope",
        "work_description", "contract_description", "title",
        "project_title", "service_description", "deliverable",
        "statement_of_work", "sow",
    ],
    "amount": [
        "amount", "contract_amount", "award_amount", "total_amount",
        "value", "cost", "price", "obligation", "total_value",
        "contract_value", "funded_amount",
    ],
    "date": [
        "date", "award_date", "start_date", "contract_date",
        "effective_date", "period_of_performance_start",
        "signed_date", "execution_date",
    ],
    "department": [
        "department", "dept", "agency", "office", "ministry",
        "division", "bureau", "awarding_agency",
    ],
    "contract_id": [
        "contract_id", "contract_number", "id", "contract_no",
        "award_id", "reference", "piid",
    ],
    "end_date": [
        "end_date", "period_of_performance_end", "expiry_date",
        "completion_date", "contract_end",
    ],
}

CORPORATE_SUFFIXES = [
    r"\bllc\b", r"\binc\.?\b", r"\bcorp\.?\b", r"\bltd\.?\b",
    r"\bco\.?\b", r"\bcompany\b", r"\bgroup\b", r"\bassociates\b",
    r"\bservices\b", r"\bsolutions\b", r"\benterprises\b",
    r"\bglobal\b", r"\bnational\b", r"\binternational\b", r"\bfederal\b",
]

DATE_FORMATS = [
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y",
    "%b %d, %Y", "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
    "%Y-%m-%dT%H:%M:%S", "%d %B %Y", "%Y%m%d",
]


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remap column names to standard fields, then clean and enrich values.
    Adds *_normalized / *_parsed / *_numeric columns for downstream use.
    """
    df = df.copy()

    # --- Column renaming ---
    lower_cols = {c: c.lower().strip() for c in df.columns}
    df.rename(columns=lower_cols, inplace=True)

    rename_map: dict[str, str] = {}
    used_standards: set[str] = set()
    for col in df.columns:
        for standard, aliases in FIELD_ALIASES.items():
            if col in aliases and standard not in used_standards:
                rename_map[col] = standard
                used_standards.add(standard)
                break

    df.rename(columns=rename_map, inplace=True)

    # --- Vendor ---
    if "vendor" not in df.columns:
        df["vendor"] = "Unknown"
    df["vendor_normalized"] = df["vendor"].apply(_normalize_vendor)

    # --- Amount ---
    if "amount" not in df.columns:
        df["amount"] = None
    df["amount_numeric"] = df["amount"].apply(_parse_amount)

    # --- Date ---
    if "date" not in df.columns:
        df["date"] = None
    df["date_parsed"] = df["date"].apply(_parse_date)

    # --- Description ---
    if "description" not in df.columns:
        # Promote raw_text if available
        text_cols = [c for c in df.columns if "text" in c or "raw" in c]
        df["description"] = df[text_cols[0]] if text_cols else ""
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    # --- Stable record ID ---
    df["record_id"] = range(len(df))

    return df


# ---------------------------------------------------------------------------
# Value cleaners
# ---------------------------------------------------------------------------

def _normalize_vendor(name) -> str:
    if pd.isna(name) or str(name).strip() in ("", "Unknown"):
        return "unknown"
    name = str(name).lower().strip()
    for suffix in CORPORATE_SUFFIXES:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _parse_amount(value) -> Optional[float]:
    if pd.isna(value) or str(value).strip() in ("", "None"):
        return None
    s = re.sub(r"[$£€,\s]", "", str(value).strip())
    # Handle shorthand suffixes  (e.g. "125K", "2.5M")
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
    if pd.isna(value) or str(value).strip() in ("", "None"):
        return None
    s = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
