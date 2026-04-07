"""
Similarity computation module.
Computes pairwise scores for description, vendor, date, and amount.
Uses TF-IDF + cosine similarity for text; simple heuristics for structured fields.
"""
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional


def compute_similarities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame of all record pairs with per-dimension similarity scores.
    """
    if len(df) < 2:
        return pd.DataFrame()

    n = len(df)
    desc_matrix = _text_similarity_matrix(df["description"].fillna("").tolist())

    pairs = []
    for i, j in combinations(range(n), 2):
        pairs.append({
            "idx_a": i,
            "idx_b": j,
            "desc_similarity": float(desc_matrix[i, j]),
            "vendor_similarity": _vendor_similarity(
                df.iloc[i].get("vendor_normalized", ""),
                df.iloc[j].get("vendor_normalized", ""),
            ),
            "date_proximity": _date_proximity(
                df.iloc[i].get("date_parsed"),
                df.iloc[j].get("date_parsed"),
            ),
            "amount_similarity": _amount_similarity(
                df.iloc[i].get("amount_numeric"),
                df.iloc[j].get("amount_numeric"),
            ),
        })

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def _text_similarity_matrix(texts: list[str]) -> np.ndarray:
    """TF-IDF cosine similarity with bigrams."""
    n = len(texts)
    fallback = np.zeros((n, n))

    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return fallback

    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=10_000,
            sublinear_tf=True,
        )
        mat = vec.fit_transform(texts)
        return cosine_similarity(mat)
    except Exception:
        return fallback


def _vendor_similarity(a: str, b: str) -> float:
    """Jaccard token similarity + containment bonus."""
    if not a or not b or "unknown" in (a, b):
        return 0.0
    if a == b:
        return 1.0

    tok_a = set(a.split())
    tok_b = set(b.split())
    if not tok_a or not tok_b:
        return 0.0

    jaccard = len(tok_a & tok_b) / len(tok_a | tok_b)
    # Substring containment bonus (e.g. "techsecure" ⊆ "techsecure solutions")
    contains = 0.3 if (a in b or b in a) else 0.0

    return min(1.0, jaccard + contains)


def _date_proximity(d1: Optional[object], d2: Optional[object]) -> float:
    """
    Decaying proximity score.  Same day → 1.0, >5 years apart → ~0.
    """
    if d1 is None or d2 is None:
        return 0.0

    delta = abs((d1 - d2).days)

    if delta == 0:
        return 1.0
    if delta <= 30:
        return 0.90
    if delta <= 90:
        return 0.75
    if delta <= 180:
        return 0.55
    if delta <= 365:
        return 0.35
    # Gradual decay beyond one year
    return max(0.0, 1.0 - delta / 1825)  # 5-year window


def _amount_similarity(a: Optional[float], b: Optional[float]) -> float:
    """Ratio of the smaller to the larger amount."""
    if a is None or b is None:
        return 0.0
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return float(min(a, b) / max(a, b))
