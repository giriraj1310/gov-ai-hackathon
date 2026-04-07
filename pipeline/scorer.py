"""
Parallel similarity scoring module.

Scores every candidate pair across four dimensions:
  - description  : TF-IDF cosine similarity (pre-computed once, L2-normalised)
  - vendor       : token Jaccard + substring containment bonus
  - date         : decaying proximity over a 5-year window
  - amount       : min/max ratio

The TF-IDF matrix is built once and L2-normalised so that cosine similarity
reduces to a sparse dot product — no full matrix is materialised.

Candidate pairs are split into batches and scored in parallel using
ThreadPoolExecutor.  Batching amortises thread-dispatch overhead and keeps
each task large enough to benefit from numpy's GIL-releasing BLAS calls.
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

from pipeline import CandidatePair

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "description": 0.40,
    "vendor":      0.30,
    "date":        0.15,
    "amount":      0.15,
}

BATCH_SIZE  = 2_000                              # pairs per thread task
MAX_WORKERS = min(4, os.cpu_count() or 1)       # CPU-bound; don't over-subscribe

ProgressCb = Optional[Callable[[int, int], None]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidates(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    weights: Optional[Dict[str, float]] = None,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    progress_cb: ProgressCb = None,
) -> pd.DataFrame:
    """
    Score all candidate pairs and return a DataFrame sorted by risk_score desc.

    Parameters
    ----------
    df          : Normalised records DataFrame
    candidates  : Set of (i, j) pairs from the blocking stage
    weights     : Scoring dimension weights (normalised internally)
    batch_size  : Pairs per thread task
    max_workers : Thread pool size
    progress_cb : Called with (batches_done, total_batches) after each batch
    """
    if not candidates or df.empty:
        return pd.DataFrame()

    w = _normalise_weights(weights or DEFAULT_WEIGHTS)

    # Build TF-IDF matrix once — L2-normalised so cosine = dot product
    tfidf_mat = _build_tfidf(df["description"].fillna("").tolist())

    # Split into batches
    pair_list = sorted(candidates)                # Sort for reproducibility
    batches   = [pair_list[i:i + batch_size] for i in range(0, len(pair_list), batch_size)]
    total_b   = len(batches)

    all_results: List[Dict] = []
    completed  = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_batch, batch, df, tfidf_mat, w): batch_idx
            for batch_idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            try:
                all_results.extend(future.result())
            except Exception:
                pass   # Individual batch failures are non-fatal
            completed += 1
            if progress_cb:
                progress_cb(completed, total_b)

    if not all_results:
        return pd.DataFrame()

    out = pd.DataFrame(all_results)
    out["risk_score"] = out["risk_score"].clip(0.0, 1.0)
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = sum(w.values())
    if total == 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in w.items()}


def _build_tfidf(texts: List[str]) -> Optional[sp.csr_matrix]:
    """
    Fit TF-IDF on all descriptions and return an L2-normalised sparse matrix.
    L2-normalisation means cosine(i, j) == mat[i] · mat[j] (dot product).
    Returns None if not enough non-empty texts.

    Uses conservative max_features to limit memory on large datasets.
    """
    non_empty = [t for t in texts if t.strip()]
    if len(non_empty) < 2:
        return None
    # Scale features with dataset size to bound memory
    n = len(texts)
    max_feat = min(5_000, max(500, n // 2))
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,          # Ignore terms appearing in only 1 doc (noise reduction)
            max_features=max_feat,
            sublinear_tf=True,
            dtype="float64",   # float32 not supported by sklearn TfidfVectorizer
        )
        mat = vec.fit_transform(texts)
        return sk_normalize(mat, norm="l2", copy=False)
    except Exception:
        return None


def _score_batch(
    pairs: List[CandidatePair],
    df: pd.DataFrame,
    tfidf_mat: Optional[sp.csr_matrix],
    weights: Dict[str, float],
) -> List[Dict]:
    """Score a single batch of candidate pairs. Runs inside a thread."""
    results: List[Dict] = []

    for i, j in pairs:
        if i >= len(df) or j >= len(df):
            continue

        rec_a = df.iloc[i]
        rec_b = df.iloc[j]

        d_sim  = _cosine_similarity(i, j, tfidf_mat)
        v_sim  = _vendor_similarity(
            rec_a.get("vendor_normalized", ""),
            rec_b.get("vendor_normalized", ""),
        )
        dt_prx = _date_proximity(rec_a.get("date_parsed"), rec_b.get("date_parsed"))
        a_sim  = _amount_similarity(rec_a.get("amount_numeric"), rec_b.get("amount_numeric"))

        risk = (
            d_sim  * weights.get("description", 0.40)
            + v_sim  * weights.get("vendor",      0.30)
            + dt_prx * weights.get("date",        0.15)
            + a_sim  * weights.get("amount",      0.15)
        )

        results.append({
            "idx_a":             i,
            "idx_b":             j,
            "desc_similarity":   d_sim,
            "vendor_similarity": v_sim,
            "date_proximity":    dt_prx,
            "amount_similarity": a_sim,
            "risk_score":        float(risk),
        })

    return results


# ---------------------------------------------------------------------------
# Per-dimension similarity functions
# ---------------------------------------------------------------------------

def _cosine_similarity(i: int, j: int, mat: Optional[sp.csr_matrix]) -> float:
    """
    Cosine similarity via sparse dot product (L2-normalised matrix).
    GIL is released during the BLAS dot-product step, allowing true parallelism.
    """
    if mat is None or i >= mat.shape[0] or j >= mat.shape[0]:
        return 0.0
    try:
        # Sparse dot product: shape (1, 1)
        result = (mat[i] * mat[j].T).toarray()
        return float(np.clip(result[0, 0], 0.0, 1.0))
    except Exception:
        return 0.0


def _vendor_similarity(a: str, b: str) -> float:
    """Token Jaccard + substring containment bonus."""
    if not a or not b:
        return 0.0
    if "unknown" in (a, b):
        return 0.0
    if a == b:
        return 1.0
    tok_a = set(a.split())
    tok_b = set(b.split())
    if not tok_a or not tok_b:
        return 0.0
    jaccard  = len(tok_a & tok_b) / len(tok_a | tok_b)
    contains = 0.30 if (a in b or b in a) else 0.0
    return float(min(1.0, jaccard + contains))


def _date_proximity(d1, d2) -> float:
    """Decaying score.  Same day → 1.0; > 5 years → ~0."""
    if d1 is None or d2 is None:
        return 0.0
    delta = abs((d1 - d2).days)
    if delta == 0:    return 1.00
    if delta <= 30:   return 0.90
    if delta <= 90:   return 0.75
    if delta <= 180:  return 0.55
    if delta <= 365:  return 0.35
    return float(max(0.0, 1.0 - delta / 1825))   # 5-year linear decay


def _amount_similarity(a, b) -> float:
    """Min/max ratio — 1.0 if equal, 0.0 if one is zero or missing."""
    if a is None or b is None:
        return 0.0
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return 0.0
    if a == 0 and b == 0:
        return 1.0
    if a == 0 or b == 0:
        return 0.0
    return float(min(a, b) / max(a, b))
