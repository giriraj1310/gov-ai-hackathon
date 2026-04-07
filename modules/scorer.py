"""
Risk scoring module.
Applies configurable weights to per-dimension similarity scores
and produces a single normalised risk_score per pair.
"""
import pandas as pd
from typing import Dict


DEFAULT_WEIGHTS: Dict[str, float] = {
    "description": 0.40,
    "vendor": 0.30,
    "date": 0.15,
    "amount": 0.15,
}


def score_pairs(
    pairs_df: pd.DataFrame,
    records_df: pd.DataFrame,
    weights: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute a weighted risk_score for every pair and sort descending.

    Parameters
    ----------
    pairs_df   : Output of compute_similarities()
    records_df : Normalised records DataFrame
    weights    : Dict with keys description/vendor/date/amount.
                 Values are normalised internally so they need not sum to 1.
    """
    if pairs_df.empty:
        return pairs_df.copy()

    if weights is None:
        weights = DEFAULT_WEIGHTS

    w = weights.copy()
    total = sum(w.values()) or 1.0  # guard against zero-sum config

    df = pairs_df.copy()

    df["risk_score"] = (
        df["desc_similarity"]   * w.get("description", 0.40)
        + df["vendor_similarity"] * w.get("vendor", 0.30)
        + df["date_proximity"]    * w.get("date", 0.15)
        + df["amount_similarity"] * w.get("amount", 0.15)
    ) / total

    # Clip to [0, 1] in case of floating-point overshoot
    df["risk_score"] = df["risk_score"].clip(0.0, 1.0)

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)
