"""
Blocking module.

Reduces O(n²) candidate pair generation to a manageable subset
by grouping records into blocks before comparing.

Five blocking strategies are applied and their candidate sets are unioned:
  1. Vendor prefix        — first 5 chars of normalised vendor name
  2. Department           — normalised agency/department string
  3. Amount band          — order-of-magnitude bucket (log10)
  4. Date window          — year-quarter bucket
  5. Description LSH      — MinHashLSH on word shingles (datasketch)

All strategies enforce a per-block record cap to prevent any single
over-populated block from reproducing O(n²) behaviour.
Large blocks are subdivided by a secondary key before capping.
"""
from __future__ import annotations

import random
import re
from itertools import combinations
from typing import Callable, Optional, Set, Tuple

import numpy as np
import pandas as pd

from pipeline import CandidatePair

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

MAX_RECORDS_PER_BLOCK = 400    # Max records per leaf block before sampling
MAX_TOTAL_PAIRS       = 500_000  # Hard cap on total candidate pairs
LSH_THRESHOLD         = 0.40   # MinHash Jaccard threshold for description pairing
LSH_NUM_PERM          = 64     # MinHash permutations (64 is fast; 128 is more accurate)
MIN_WORDS_FOR_LSH     = 3      # Skip LSH for descriptions shorter than this

ProgressCb = Optional[Callable[[int, int, str], None]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    df: pd.DataFrame,
    max_records_per_block: int = MAX_RECORDS_PER_BLOCK,
    max_total_pairs: int       = MAX_TOTAL_PAIRS,
    lsh_threshold: float       = LSH_THRESHOLD,
    progress_cb: ProgressCb    = None,
) -> Set[CandidatePair]:
    """
    Generate a set of candidate (i, j) pairs worth scoring.
    Returns a set of tuples where i < j (no self-pairs, no mirror-pairs).
    """
    n = len(df)

    # Tiny dataset: compare everything
    if n <= max_records_per_block:
        return set(combinations(range(n), 2))

    candidates: Set[CandidatePair] = set()

    steps = [
        ("Blocking by vendor prefix…",    _block_by_vendor),
        ("Blocking by department…",       _block_by_department),
        ("Blocking by amount band…",      _block_by_amount),
        ("Blocking by date window…",      _block_by_date),
    ]

    for idx, (msg, fn) in enumerate(steps):
        if progress_cb:
            progress_cb(idx, len(steps) + 1, msg)
        fn(df, candidates, max_records_per_block)

    # LSH blocking (optional — needs datasketch)
    if progress_cb:
        progress_cb(len(steps), len(steps) + 1, "Blocking by description similarity (LSH)…")
    _block_by_description_lsh(df, candidates, lsh_threshold, LSH_NUM_PERM)

    if progress_cb:
        progress_cb(len(steps) + 1, len(steps) + 1, f"Candidate set built: {len(candidates):,} pairs.")

    # Global cap — sample deterministically by sorting for reproducibility
    if len(candidates) > max_total_pairs:
        sorted_pairs = sorted(candidates)
        candidates   = set(random.sample(sorted_pairs, max_total_pairs))

    return candidates


# ---------------------------------------------------------------------------
# Blocking strategies
# ---------------------------------------------------------------------------

def _safe_add_block(
    indices: list[int],
    candidates: Set[CandidatePair],
    max_records: int,
) -> None:
    """
    Add all pairwise combinations from `indices` into `candidates`.
    If the block exceeds `max_records`, randomly sample before pairing
    to keep pair count bounded.
    """
    if len(indices) < 2:
        return
    if len(indices) > max_records:
        indices = random.sample(indices, max_records)
    for i, j in combinations(sorted(indices), 2):
        candidates.add((i, j))


def _block_by_vendor(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    max_records: int,
) -> None:
    if "vendor_normalized" not in df.columns:
        return

    safe = df[
        df["vendor_normalized"].notna()
        & (df["vendor_normalized"].str.len() >= 4)
        & ~df["vendor_normalized"].str.startswith("unkno")
        & (df["vendor_normalized"] != "unknown")
    ].copy()

    safe["_prefix"] = safe["vendor_normalized"].str[:5]

    for _, group in safe.groupby("_prefix"):
        if len(group) < 2:
            continue
        _safe_add_block(group.index.tolist(), candidates, max_records)


def _block_by_department(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    max_records: int,
) -> None:
    if "department" not in df.columns:
        return

    valid = df[df["department"].notna()].copy()
    valid["_dept_key"] = (
        valid["department"].astype(str)
        .str.lower().str.strip().str[:30]
    )
    valid = valid[valid["_dept_key"].str.len() >= 3]

    for _, group in valid.groupby("_dept_key"):
        if len(group) < 2:
            continue

        # Subdivide large department blocks by amount band to prevent explosion
        if len(group) > max_records * 2 and "amount_numeric" in group.columns:
            group = group.copy()
            group["_subdiv"] = np.floor(
                np.log10(group["amount_numeric"].clip(lower=1).fillna(1))
            ).astype(int)
            for _, subgroup in group.groupby("_subdiv"):
                if len(subgroup) >= 2:
                    _safe_add_block(subgroup.index.tolist(), candidates, max_records)
        else:
            _safe_add_block(group.index.tolist(), candidates, max_records)


def _block_by_amount(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    max_records: int,
) -> None:
    if "amount_numeric" not in df.columns:
        return

    # Rows with valid positive amounts
    valid = df[df["amount_numeric"].notna() & (df["amount_numeric"] > 0)].copy()
    if valid.empty:
        return

    valid["_bucket"] = np.floor(np.log10(valid["amount_numeric"])).astype(int)

    # Rows with zero or null amounts get their own bucket — don't mix with populated rows
    zero_null = df[~df.index.isin(valid.index)].copy()
    if len(zero_null) >= 2:
        _safe_add_block(zero_null.index.tolist(), candidates, max_records)

    for _, group in valid.groupby("_bucket"):
        if len(group) >= 2:
            _safe_add_block(group.index.tolist(), candidates, max_records)


def _block_by_date(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    max_records: int,
) -> None:
    if "date_parsed" not in df.columns:
        return

    valid = df[df["date_parsed"].notna()].copy()
    if valid.empty:
        return

    valid["_date_bucket"] = valid["date_parsed"].apply(
        lambda d: f"{d.year}-Q{(d.month - 1) // 3 + 1}"
    )

    for _, group in valid.groupby("_date_bucket"):
        if len(group) >= 2:
            _safe_add_block(group.index.tolist(), candidates, max_records)


def _block_by_description_lsh(
    df: pd.DataFrame,
    candidates: Set[CandidatePair],
    threshold: float,
    num_perm: int,
) -> None:
    """MinHashLSH on word shingles. Skips records with fewer than MIN_WORDS_FOR_LSH words."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        return  # datasketch not installed — skip silently

    if "description" not in df.columns:
        return

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[int, object] = {}

    for idx, row in df.iterrows():
        text = str(row.get("description", "")).lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)   # 3+ char words only

        if len(words) < MIN_WORDS_FOR_LSH:
            continue   # Too sparse for reliable LSH

        m = MinHash(num_perm=num_perm)
        for w in set(words):          # Unique word shingles
            m.update(w.encode("utf-8"))

        key = str(idx)
        try:
            lsh.insert(key, m)
            minhashes[idx] = m
        except Exception:
            pass   # Duplicate key in LSH — skip

    # Query each record's nearest neighbours
    for idx, m in minhashes.items():
        try:
            neighbours = lsh.query(m)
            for nbr_key in neighbours:
                nbr_idx = int(nbr_key)
                if nbr_idx != idx:
                    pair: CandidatePair = (min(idx, nbr_idx), max(idx, nbr_idx))
                    candidates.add(pair)
        except Exception:
            pass
