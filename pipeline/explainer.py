"""
Explanation generation module.

Converts numeric similarity scores into structured, analyst-grade findings.
Each finding contains:
  - cleaned evidence records (Contract A / Contract B)
  - plain-language signals explaining why the pair was flagged
  - a per-dimension score breakdown
  - a tiered analyst recommendation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

# Fields shown in the evidence cards shown to the analyst
DISPLAY_FIELDS = [
    "contract_id", "vendor", "description",
    "amount", "date", "end_date", "department",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanations(
    scored_df: pd.DataFrame,
    records_df: pd.DataFrame,
    threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Build a list of finding dicts for every pair in scored_df.

    All pairs are returned; the caller uses `above_threshold` to filter
    the display.  This allows the UI to let users interactively lower the
    threshold without re-running the scoring stage.
    """
    if scored_df.empty:
        return []

    results: List[Dict[str, Any]] = []

    for rank, (_, row) in enumerate(scored_df.iterrows()):
        idx_a = int(row["idx_a"])
        idx_b = int(row["idx_b"])
        risk  = float(row["risk_score"])

        if idx_a >= len(records_df) or idx_b >= len(records_df):
            continue

        rec_a = records_df.iloc[idx_a].to_dict()
        rec_b = records_df.iloc[idx_b].to_dict()

        breakdown = {
            "description": float(row["desc_similarity"]),
            "vendor":      float(row["vendor_similarity"]),
            "date":        float(row["date_proximity"]),
            "amount":      float(row["amount_similarity"]),
        }

        results.append({
            "rank":             rank + 1,
            "pair_id":          f"PAIR-{idx_a:04d}-{idx_b:04d}",
            "risk_score":       risk,
            "above_threshold":  risk >= threshold,
            "vendor_a":         str(rec_a.get("vendor", "Unknown")),
            "vendor_b":         str(rec_b.get("vendor", "Unknown")),
            "record_a":         _clean_record(rec_a),
            "record_b":         _clean_record(rec_b),
            "signals":          _build_signals(breakdown, rec_a, rec_b),
            "score_breakdown":  breakdown,
            "recommendation":   _build_recommendation(risk, breakdown),
        })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_record(rec: dict) -> dict:
    """Return only display-worthy fields as strings."""
    out: dict = {}
    for field in DISPLAY_FIELDS:
        val = rec.get(field)
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        s = str(val).strip()
        if s and s.lower() not in ("none", "nan", "nat", ""):
            out[field] = s
    return out


def _build_signals(
    scores: Dict[str, float],
    rec_a: dict,
    rec_b: dict,
) -> List[str]:
    signals: List[str] = []

    d = scores["description"]
    if d >= 0.80:
        signals.append(
            f"**Very high description similarity ({d:.0%})** — "
            "contract scopes appear nearly identical in language and structure."
        )
    elif d >= 0.50:
        signals.append(
            f"**Significant description overlap ({d:.0%})** — "
            "similar deliverables or scope-of-work detected across both contracts."
        )
    elif d >= 0.25:
        signals.append(f"Moderate description similarity ({d:.0%}).")

    v = scores["vendor"]
    va = rec_a.get("vendor", "")
    vb = rec_b.get("vendor", "")
    if v >= 0.90:
        signals.append(
            f"**Same or near-identical vendor** — '{va}' and '{vb}' "
            "appear to be the same legal entity operating under different names."
        )
    elif v >= 0.50:
        signals.append(
            f"**Similar vendor names ({v:.0%} match)** — '{va}' vs '{vb}'. "
            "Possible same entity, related subsidiary, or name variation."
        )

    dt = scores["date"]
    if dt >= 0.90:
        signals.append(
            "**Awards on the same or adjacent dates** — "
            "contracts may represent simultaneous or split awards from the same procurement."
        )
    elif dt >= 0.55:
        signals.append(
            f"**Close award dates ({dt:.0%} proximity)** — "
            "contracts awarded within a few months of each other."
        )

    a = scores["amount"]
    amt_a = rec_a.get("amount", "N/A")
    amt_b = rec_b.get("amount", "N/A")
    if a >= 0.95:
        signals.append(
            f"**Nearly identical contract values** — {amt_a} vs {amt_b}. "
            "May represent the same budget line obligated twice."
        )
    elif a >= 0.80:
        signals.append(
            f"**Very similar contract amounts ({a:.0%} ratio)** — "
            f"{amt_a} vs {amt_b}."
        )

    # Department match signal
    dept_a = str(rec_a.get("department", "")).strip().lower()
    dept_b = str(rec_b.get("department", "")).strip().lower()
    if dept_a and dept_b and dept_a == dept_b and dept_a not in ("", "none", "nan"):
        signals.append(
            f"**Same awarding department** — both contracts issued by '{rec_a.get('department', '')}'."
        )

    if not signals:
        signals.append(
            "Multiple low-level signals combined to exceed the configured risk threshold."
        )

    return signals


def _build_recommendation(risk: float, scores: Dict[str, float]) -> str:
    if risk >= 0.85:
        return (
            "URGENT: High confidence of duplication or scope overlap. "
            "Recommend immediate escalation to senior analyst. "
            "Consider suspending further payments pending review."
        )
    if risk >= 0.70:
        return (
            "Review recommended. Cross-reference contract scopes, deliverables, and invoices "
            "with the respective contracting officers before renewal or extension."
        )
    if risk >= 0.50:
        return (
            "Flag for secondary review. Verify that these contracts cover distinct deliverables "
            "and do not draw from the same funding line or budget authority."
        )
    return (
        "Below standard threshold. Monitor for future patterns. "
        "No immediate action required."
    )
