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
        # Generic
        "vendor", "vendor_name", "supplier", "contractor", "company",
        "firm", "awardee", "prime_contractor", "awarded_to", "recipient",
        # USAspending.gov / FPDS
        "recipient_name", "recipient_legal_business_name",
        "prime_award_recipient_name", "awardee_name",
        "legal_business_name", "doing_business_as_name",
        # UK / AU procurement
        "supplier_name", "contractor_name", "organisation_name",
        "entity_name", "payee_name", "beneficiary",
        # CanadaBuys (after stripping French suffix)
        "vendorname", "suppliername", "vendeur",
    ],
    "description": [
        # Generic
        "description", "desc", "project_description", "scope",
        "work_description", "contract_description", "title",
        "project_title", "service_description", "deliverable",
        "statement_of_work", "sow", "purpose", "subject",
        # USAspending.gov / FPDS
        "award_description", "description_of_contract_requirement",
        "description_of_requirement", "product_or_service_description",
        "short_description", "award_type_description",
        # UK / AU
        "description_of_purchase", "description_of_services",
        "procurement_description", "goods_services_description",
        "nature_of_contract",
    ],
    "amount": [
        # Generic
        "amount", "contract_amount", "award_amount", "total_amount",
        "value", "cost", "price", "obligation", "total_value",
        "contract_value", "funded_amount", "base_amount", "total_obligated",
        # USAspending.gov / FPDS
        "total_obligated_amount", "base_and_exercised_options_value",
        "base_and_all_options_value", "current_total_value_of_award",
        "potential_total_value_of_award", "federal_action_obligation",
        "dollars_obligated", "cumulative_transaction_obligation",
        # UK / AU
        "value_gbp", "value_aud", "contract_value_aud", "total_contract_value",
        "approved_amount", "committed_amount",
        # CanadaBuys (after stripping French suffix)
        "contractvalue", "originalvalue",
    ],
    "date": [
        # Generic
        "date", "award_date", "start_date", "contract_date",
        "effective_date", "period_of_performance_start",
        "signed_date", "execution_date", "base_date",
        # USAspending.gov / FPDS
        "action_date", "award_action_date", "last_modified_date",
        "period_of_performance_start_date", "ordering_period_end_date",
        # UK / AU
        "contract_start_date", "commencement_date", "published_date",
        "notification_date", "publish_date",
        # CanadaBuys (after stripping French suffix)
        "contractawarddate", "contractperiodstart",
    ],
    "department": [
        "department", "dept", "agency", "office", "ministry",
        "division", "bureau", "awarding_agency", "contracting_agency",
        "funding_agency",
        # USAspending.gov / FPDS
        "awarding_agency_name", "awarding_sub_agency_name",
        "funding_agency_name", "funding_sub_agency_name",
        "awarding_office_name", "contracting_office_name",
        # UK / AU
        "organisation", "entity", "buyer_name", "purchasing_entity",
        "procuring_entity", "government_department",
        # CanadaBuys (after stripping French suffix)
        "departmentname", "owneracronym",
    ],
    "contract_id": [
        "contract_id", "contract_number", "id", "contract_no",
        "award_id", "reference", "piid", "order_number", "solicitation_id",
        # USAspending.gov / FPDS
        "award_id_piid", "generated_unique_award_id",
        "modification_number", "transaction_unique_key",
        # UK / AU
        "contract_reference", "tender_reference", "notice_identifier",
        # CanadaBuys
        "referencenumber", "solicitationnumber",
    ],
    "end_date": [
        "end_date", "period_of_performance_end", "expiry_date",
        "completion_date", "contract_end", "pop_end",
        # USAspending.gov / FPDS
        "period_of_performance_current_end_date",
        "period_of_performance_potential_end_date",
        # UK / AU
        "contract_end_date", "expiration_date",
        # CanadaBuys
        "contractperiodend",
    ],
    # CanadaBuys-specific: amendment number (tracked separately for business logic)
    "amendment_number": [
        "amendmentnumber", "amendment_number", "amendment_no",
        "modification_number",
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

    Handles bilingual CanadaBuys column names (e.g. vendorName-nomFournisseur)
    by stripping the French suffix before alias matching.
    """
    df = df.copy()

    # --- Step 1: normalise column names ---
    # For bilingual hyphenated names (CanadaBuys format), take only the English part.
    # e.g. "vendorName-nomFournisseur" → "vendorname"
    original_cols = list(df.columns)
    def _norm_col(c: str) -> str:
        c = str(c).strip()
        if "-" in c:
            c = c.split("-")[0]   # Keep English half only
        return c.lower().replace(" ", "_")
    df.columns = [_norm_col(c) for c in df.columns]

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

    # Store mapping for diagnostics — use _norm_col so bilingual names map correctly
    col_mapping = {orig: rename_map.get(_norm_col(orig), "(unmapped)")
                   for orig in original_cols}
    df.attrs["column_mapping"] = col_mapping
    df.attrs["unmapped_cols"]  = [orig for orig, std in col_mapping.items()
                                   if std == "(unmapped)"]

    # --- Step 3: ensure required columns exist with catch-all fallbacks ---

    # Vendor fallback: find any high-cardinality string column
    if "vendor" not in df.columns:
        fallback = _find_entity_column(df)
        df["vendor"] = df[fallback] if fallback else None

    # Description fallback: find the column with the longest average text
    if "description" not in df.columns:
        text_cols = [c for c in df.columns if "text" in c or "raw" in c]
        if text_cols:
            df["description"] = df[text_cols[0]]
        else:
            best = _find_text_column(df, exclude={"vendor"})
            df["description"] = df[best] if best else ""

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


def _find_text_column(df: pd.DataFrame, exclude: set[str]) -> str | None:
    """Return the non-excluded string column with the highest mean text length."""
    best_col, best_len = None, 0.0
    for col in df.columns:
        if col in exclude or col.startswith("_"):
            continue
        try:
            mean_len = df[col].fillna("").astype(str).str.len().mean()
            if mean_len > best_len and mean_len > 20:
                best_len, best_col = mean_len, col
        except Exception:
            continue
    return best_col


def _find_entity_column(df: pd.DataFrame) -> str | None:
    """
    Return the string column that looks most like vendor/entity names:
    medium-cardinality (not unique per row, not a constant), reasonable string lengths.
    """
    n = len(df)
    best_col, best_score = None, -1.0
    for col in df.columns:
        if col.startswith("_"):
            continue
        try:
            series = df[col].fillna("").astype(str)
            n_unique = series.nunique()
            mean_len = series.str.len().mean()
            # Ideal: cardinality 2–80% of n, mean length 5–60 chars
            if n_unique < 2 or n_unique > n * 0.85:
                continue
            if mean_len < 3 or mean_len > 100:
                continue
            score = (n_unique / n) * min(mean_len / 30, 1.0)
            if score > best_score:
                best_score, best_col = score, col
        except Exception:
            continue
    return best_col


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
