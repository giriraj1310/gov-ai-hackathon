"""
Explanation generation module.
Turns numeric similarity scores into human-readable analyst-grade findings.
"""
import pandas as pd
from typing import Any, Dict, List


# Fields shown in the evidence cards
DISPLAY_FIELDS = [
    "contract_id", "vendor", "description", "amount",
    "date", "end_date", "department",
]


def generate_explanations(
    scored_df: pd.DataFrame,
    records_df: pd.DataFrame,
    threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Build a rich finding dict for every pair in scored_df.
    Pairs below `threshold` are included but marked as below threshold,
    so the UI can still render them if the user lowers the slider.
    """
    if scored_df.empty:
        return []

    results = []

    for _, row in scored_df.iterrows():
        idx_a = int(row["idx_a"])
        idx_b = int(row["idx_b"])
        risk = float(row["risk_score"])

        rec_a = records_df.iloc[idx_a].to_dict()
        rec_b = records_df.iloc[idx_b].to_dict()

        score_breakdown = {
            "description": float(row["desc_similarity"]),
            "vendor":      float(row["vendor_similarity"]),
            "date":        float(row["date_proximity"]),
            "amount":      float(row["amount_similarity"]),
        }

        signals = _build_signals(score_breakdown, rec_a, rec_b)
        recommendation = _build_recommendation(risk, score_breakdown)

        results.append({
            "pair_id": f"PAIR-{idx_a:03d}-{idx_b:03d}",
            "risk_score": risk,
            "above_threshold": risk >= threshold,
            "vendor_a": rec_a.get("vendor", "Unknown"),
            "vendor_b": rec_b.get("vendor", "Unknown"),
            "record_a": _clean_record(rec_a),
            "record_b": _clean_record(rec_b),
            "signals": signals,
            "score_breakdown": score_breakdown,
            "recommendation": recommendation,
        })

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_record(rec: dict) -> dict:
    """Return only display-worthy fields, converted to strings."""
    out = {}
    for field in DISPLAY_FIELDS:
        val = rec.get(field)
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            s = str(val).strip()
            if s and s.lower() not in ("none", "nan", ""):
                out[field] = s
    return out


def _build_signals(scores: Dict[str, float], rec_a: dict, rec_b: dict) -> List[str]:
    signals: List[str] = []

    d = scores["description"]
    if d >= 0.80:
        signals.append(
            f"**Very high description similarity ({d:.0%})** — contract scopes appear nearly identical."
        )
    elif d >= 0.50:
        signals.append(
            f"**Significant description overlap ({d:.0%})** — similar deliverables or scope of work detected."
        )
    elif d >= 0.25:
        signals.append(f"Moderate description similarity ({d:.0%}).")

    v = scores["vendor"]
    vendor_a = rec_a.get("vendor", "")
    vendor_b = rec_b.get("vendor", "")
    if v >= 0.90:
        signals.append(
            f"**Same or near-identical vendor** — '{vendor_a}' and '{vendor_b}' appear to be the same entity."
        )
    elif v >= 0.50:
        signals.append(
            f"**Similar vendor names ({v:.0%} match)** — possible same entity operating under different registrations."
        )

    dt = scores["date"]
    if dt >= 0.90:
        signals.append(
            "**Awards on the same or adjacent dates** — contracts may represent simultaneous or split awards."
        )
    elif dt >= 0.55:
        signals.append(
            f"**Close award dates ({dt:.0%} proximity)** — contracts awarded within a few months of each other."
        )

    a = scores["amount"]
    amt_a = rec_a.get("amount", "N/A")
    amt_b = rec_b.get("amount", "N/A")
    if a >= 0.95:
        signals.append(
            f"**Nearly identical contract values** — {amt_a} vs {amt_b}. May represent the same budget line."
        )
    elif a >= 0.80:
        signals.append(
            f"**Very similar contract amounts ({a:.0%} ratio)** — {amt_a} vs {amt_b}."
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
            "Escalate to senior analyst. Consider contract suspension pending review."
        )
    if risk >= 0.70:
        return (
            "Review recommended. Cross-reference contract scopes, deliverables, and "
            "invoices with the respective contracting officers before renewal or extension."
        )
    if risk >= 0.50:
        return (
            "Flag for secondary review. Verify that contracts cover distinct deliverables "
            "and do not draw from the same funding line."
        )
    return (
        "Below standard threshold. Monitor for future patterns. "
        "No immediate action required."
    )
