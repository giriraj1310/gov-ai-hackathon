"""
Deduplication module.

Removes exact and near-exact duplicate records before analysis so that
structural duplicates don't flood the scoring stage with trivial pairs.

Two passes:
  1. Exact contract_id duplicates (keep first occurrence)
  2. Near-exact structural duplicates: same vendor_normalized + amount_numeric + date_parsed

Returns (clean_df, removed_df) with record_id reassigned on clean_df.
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove exact and near-exact duplicate records.

    Returns
    -------
    clean_df : de-duplicated DataFrame (record_id reassigned)
    removed_df : rows that were removed (for reporting)
    """
    removed_parts: list[pd.DataFrame] = []

    # --- Pass 1: exact contract_id duplicates ---
    if "contract_id" in df.columns:
        # Only consider rows where contract_id is populated
        has_id = df["contract_id"].notna() & (df["contract_id"].astype(str).str.strip() != "")
        id_dup_mask = has_id & df.duplicated(subset=["contract_id"], keep="first")
        if id_dup_mask.any():
            removed_parts.append(df[id_dup_mask].copy())
            df = df[~id_dup_mask].copy()

    # --- Pass 2: near-exact structural duplicates ---
    # Match on vendor_normalized + amount_numeric + date_parsed
    struct_cols = [
        c for c in ["vendor_normalized", "amount_numeric", "date_parsed"]
        if c in df.columns
    ]
    if len(struct_cols) >= 2:
        # Only deduplicate rows where all struct_cols are populated
        all_populated = df[struct_cols].notna().all(axis=1)
        candidates = df[all_populated]
        dup_mask_candidates = candidates.duplicated(subset=struct_cols, keep="first")
        dup_indices = candidates[dup_mask_candidates].index
        if len(dup_indices) > 0:
            removed_parts.append(df.loc[dup_indices].copy())
            df = df.drop(index=dup_indices)

    # --- Reassign stable record IDs ---
    df = df.reset_index(drop=True)
    df["record_id"] = df.index

    removed_df = pd.concat(removed_parts, ignore_index=True) if removed_parts else pd.DataFrame()

    return df, removed_df
