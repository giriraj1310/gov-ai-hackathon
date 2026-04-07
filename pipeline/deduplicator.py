"""
Deduplication module.

Handles two distinct scenarios:

1. AMENDMENTS (CanadaBuys and similar systems)
   The same contract gets amended over time — same referenceNumber /
   contract_id, different amendment_number. These are NOT suspicious;
   they are normal contract modifications.
   Action: collapse each amendment chain to its latest amendment,
   carrying forward the most current values. Store the amendment history
   for reference in the UI.

2. EXACT STRUCTURAL DUPLICATES
   Rows where vendor_normalized + amount_numeric + date_parsed are
   identical — these are export artifacts (same transaction exported twice).
   Action: remove, keep first occurrence.

Returns (clean_df, audit) where audit is a dict with counts and examples.
"""
from __future__ import annotations

import pandas as pd
from typing import Dict, Any, Tuple


def deduplicate(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Remove duplicates and collapse amendment chains.

    Returns
    -------
    clean_df : de-duplicated, amendment-collapsed DataFrame
    audit    : dict with counts and sample rows for UI display
    """
    audit: Dict[str, Any] = {
        "amendments_collapsed": 0,
        "amendment_groups":     0,
        "structural_dups":      0,
        "original_count":       len(df),
    }

    # --- Pass 1: Collapse amendment chains ---
    # If contract_id and amendment_number both exist, group by contract_id
    # and keep only the latest amendment per contract.
    if "contract_id" in df.columns and "amendment_number" in df.columns:
        df, n_collapsed, n_groups = _collapse_amendments(df)
        audit["amendments_collapsed"] = n_collapsed
        audit["amendment_groups"]     = n_groups

    # --- Pass 2: Exact structural duplicates ---
    struct_cols = [c for c in ["vendor_normalized", "amount_numeric", "date_parsed"]
                   if c in df.columns]
    if len(struct_cols) >= 2:
        # Only deduplicate rows where all struct_cols are non-null
        populated = df[struct_cols].notna().all(axis=1)
        dup_mask  = populated & df.duplicated(subset=struct_cols, keep="first")
        audit["structural_dups"] = int(dup_mask.sum())
        if dup_mask.any():
            df = df[~dup_mask].copy()

    df = df.reset_index(drop=True)
    df["record_id"] = df.index
    audit["final_count"] = len(df)

    return df, audit


def _collapse_amendments(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Within each contract_id group, keep only the row with the highest
    amendment_number (most recent state of the contract).
    Returns (collapsed_df, rows_removed, n_groups_collapsed).
    """
    # Normalise amendment_number to numeric for sorting
    df = df.copy()
    df["_amend_num"] = pd.to_numeric(df["amendment_number"], errors="coerce").fillna(0)

    # Only process groups where at least two amendments exist
    id_col = "contract_id"
    has_id = df[id_col].notna() & (df[id_col].astype(str).str.strip() != "")
    df_with_id   = df[has_id]
    df_without_id = df[~has_id]

    group_sizes = df_with_id.groupby(id_col)[id_col].transform("count")
    multi_amend  = df_with_id[group_sizes > 1]
    single       = df_with_id[group_sizes == 1]

    n_groups = 0
    n_removed = 0

    if len(multi_amend) > 0:
        # Keep only the row with the max amendment_number per contract_id
        latest_idx = (
            multi_amend.groupby(id_col)["_amend_num"]
            .idxmax()
            .values
        )
        kept     = multi_amend.loc[latest_idx]
        n_groups = multi_amend[id_col].nunique()
        n_removed = len(multi_amend) - len(kept)
        df = pd.concat([kept, single, df_without_id], ignore_index=True)

    df = df.drop(columns=["_amend_num"], errors="ignore")
    return df, n_removed, n_groups
