"""
Data quality assessment module.

Runs a pre-analysis pass over the normalised DataFrame to surface
data quality issues before the user commits to a long-running analysis.

Returns a DataQualityReport dataclass with fields, scores, warnings,
and blocking-readiness indicators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldReport:
    name: str
    completeness: float          # 0.0 – 1.0 fraction of non-empty rows
    unique_ratio: float          # unique values / populated rows
    sample_values: List[str]
    issues: List[str]


@dataclass
class DataQualityReport:
    total_records: int
    field_reports: List[FieldReport]
    duplicate_count: int
    vocabulary_sparsity: float   # fraction of descriptions under 5 words
    overall_score: float         # 0.0 – 1.0 composite
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    blocking_notes: List[str]    # Notes about blocking strategy suitability


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

KEY_FIELDS = [
    ("vendor",      0.30),
    ("description", 0.35),
    ("amount",      0.20),
    ("date",        0.15),
]


def assess_quality(df: pd.DataFrame) -> DataQualityReport:
    """Assess the normalised DataFrame and return a DataQualityReport."""
    total = max(len(df), 1)
    field_reports: List[FieldReport] = []
    critical: List[str] = []
    warnings: List[str] = []
    recommendations: List[str] = []
    blocking_notes: List[str] = []

    for std_field, _ in KEY_FIELDS:
        report = _assess_field(df, std_field, total)
        field_reports.append(report)

        if std_field in ("vendor", "description"):
            if report.completeness < 0.50:
                critical.append(
                    f"'{std_field}' is only {report.completeness:.0%} populated — "
                    "similarity detection will be significantly impaired."
                )
            elif report.completeness < 0.80:
                warnings.append(
                    f"'{std_field}' is {report.completeness:.0%} complete — "
                    "some records will not participate in similarity scoring."
                )
        else:
            if report.completeness < 0.50:
                warnings.append(
                    f"'{std_field}' is only {report.completeness:.0%} populated — "
                    "this scoring dimension will have limited effect."
                )

        for issue in report.issues:
            if "low completeness" in issue.lower():
                pass  # already reported above
            else:
                warnings.append(f"[{std_field}] {issue}")

    # --- Exact / structural duplicates ---
    dup_cols = [c for c in ("vendor_normalized", "amount_numeric", "date_parsed") if c in df.columns]
    dup_count = int(df.duplicated(subset=dup_cols).sum()) if dup_cols else 0
    if dup_count > 0:
        warnings.append(
            f"{dup_count:,} near-exact duplicate rows detected "
            "(same vendor + amount + date). These will be removed before analysis."
        )
    if dup_count > total * 0.10:
        recommendations.append(
            "More than 10% of records appear to be exact duplicates. "
            "Review the data export process — the source system may be double-reporting."
        )

    # --- Vocabulary sparsity (affects MinHashLSH reliability) ---
    if "description" in df.columns:
        word_counts = df["description"].fillna("").apply(lambda t: len(t.split()))
        sparse_frac = float((word_counts < 5).sum() / total)
    else:
        sparse_frac = 1.0

    if sparse_frac > 0.50:
        blocking_notes.append(
            f"{sparse_frac:.0%} of descriptions contain fewer than 5 words. "
            "Text-based blocking (LSH) will be less reliable — "
            "structural blocking (vendor, department, date) will carry more weight."
        )
    elif sparse_frac > 0.20:
        blocking_notes.append(
            f"{sparse_frac:.0%} of descriptions are very short (<5 words). "
            "LSH similarity may produce some false positives for these records."
        )

    # --- Department block concentration check ---
    if "department" in df.columns:
        dept_counts = df["department"].value_counts(normalize=True)
        largest_dept_frac = float(dept_counts.iloc[0]) if len(dept_counts) > 0 else 0
        largest_dept = str(dept_counts.index[0]) if len(dept_counts) > 0 else "unknown"
        if largest_dept_frac > 0.40:
            blocking_notes.append(
                f"'{largest_dept}' accounts for {largest_dept_frac:.0%} of all records. "
                "This department block is large and will be subdivided automatically "
                "to prevent O(n²) pair explosion."
            )

    # --- Recommendations ---
    desc_report = next((r for r in field_reports if r.name == "description"), None)
    if desc_report and desc_report.completeness < 0.70:
        recommendations.append(
            "Enriching records with contract description or scope-of-work text "
            "will significantly improve overlap detection accuracy."
        )

    # --- Overall score ---
    weight_map = dict(KEY_FIELDS)
    overall = sum(
        next((r.completeness for r in field_reports if r.name == fname), 0.0) * w
        for fname, w in weight_map.items()
    )

    return DataQualityReport(
        total_records=len(df),
        field_reports=field_reports,
        duplicate_count=dup_count,
        vocabulary_sparsity=sparse_frac,
        overall_score=float(np.clip(overall, 0.0, 1.0)),
        critical_issues=critical,
        warnings=warnings,
        recommendations=recommendations,
        blocking_notes=blocking_notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assess_field(df: pd.DataFrame, field: str, total: int) -> FieldReport:
    issues: List[str] = []

    if field not in df.columns:
        return FieldReport(
            name=field,
            completeness=0.0,
            unique_ratio=0.0,
            sample_values=[],
            issues=[f"Field not found in dataset."],
        )

    series = df[field].astype(str).replace({"None": "", "nan": "", "NaN": "", "N/A": ""})
    populated = series.str.strip().replace("", pd.NA).dropna()
    completeness = len(populated) / total if total > 0 else 0.0
    unique_ratio  = populated.nunique() / len(populated) if len(populated) > 0 else 0.0
    sample = populated.head(3).tolist()

    if completeness < 0.50:
        issues.append(f"Low completeness ({completeness:.0%}).")

    if field == "vendor":
        if unique_ratio > 0.98 and len(populated) > 100:
            issues.append(
                "Nearly every vendor name is unique — possible encoding issue "
                "or the dataset may use free-text vendor entries."
            )
        # Check for "unknown" dominance
        unknown_frac = (df["vendor_normalized"] == "unknown").sum() / total if "vendor_normalized" in df.columns else 0
        if unknown_frac > 0.30:
            issues.append(
                f"{unknown_frac:.0%} of vendors normalised to 'unknown'. "
                "Vendor-based blocking will be limited."
            )

    if field == "amount" and "amount_numeric" in df.columns:
        parseable = df["amount_numeric"].notna().sum() / total
        if parseable < completeness - 0.10:
            issues.append(
                f"Only {parseable:.0%} of amounts parsed as numbers "
                "(expected ~{completeness:.0%}). Check currency formatting."
            )

    return FieldReport(
        name=field,
        completeness=completeness,
        unique_ratio=unique_ratio,
        sample_values=sample,
        issues=issues,
    )
