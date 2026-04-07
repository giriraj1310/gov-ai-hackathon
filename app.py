"""
GovSpend Overlap Analyzer — v2
An analyst copilot for government procurement oversight.

Pipeline stages:
  1. Ingest      — streaming chunked file reading
  2. Normalise   — field alias mapping + value cleaning
  3. Quality     — pre-analysis data quality report
  4. Deduplicate — remove exact / near-exact duplicates
  5. Block       — multi-strategy candidate pair generation
  6. Score       — parallel weighted similarity scoring
  7. Explain     — signal detection + analyst recommendations
"""

import hashlib
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.ingestor    import ingest_file
from pipeline.normalizer  import normalize_records
from pipeline.quality     import assess_quality
from pipeline.deduplicator import deduplicate
from pipeline.blocker     import build_candidate_pairs
from pipeline.scorer      import score_candidates, DEFAULT_WEIGHTS
from pipeline.explainer   import generate_explanations
from pipeline.exporter    import to_json, to_csv, to_html_report

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GovSpend Analyzer",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="stExpander"] { border-left: 4px solid #1565C0; margin-bottom: 8px; }
.stProgress > div > div { background-color: #1565C0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏛️ GovSpend Overlap Analyzer")
st.markdown(
    "**An analyst copilot for government procurement oversight.** "
    "Upload contract or spending data to surface potential duplications, "
    "overlaps, and anomalies — each finding backed by transparent, auditable evidence."
)
st.caption(
    "Decision-support tool only. All findings require human analyst review. "
    "No automated procurement decisions are made."
)
st.divider()


# ---------------------------------------------------------------------------
# Sidebar — simplified, no technical sliders
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**What should the system focus on?**")
    focus = st.radio(
        "Focus",
        ["Balanced (recommended)", "Prioritise vendor matches", "Prioritise description overlap"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**How sensitive should flagging be?**")
    sensitivity = st.radio(
        "Sensitivity",
        ["High — catch everything (more false positives)",
         "Medium — balanced (recommended)",
         "Low — only clear-cut cases"],
        index=1,
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "The system automatically configures itself based on your dataset size. "
        "No technical tuning required."
    )
    st.caption("Every finding shows exactly why it was flagged.")

# Translate plain-language choices to weights + threshold
if focus == "Prioritise vendor matches":
    weights = {"description": 0.25, "vendor": 0.50, "date": 0.15, "amount": 0.10}
elif focus == "Prioritise description overlap":
    weights = {"description": 0.55, "vendor": 0.20, "date": 0.15, "amount": 0.10}
else:
    weights = {"description": 0.40, "vendor": 0.30, "date": 0.15, "amount": 0.15}

if "High" in sensitivity:
    threshold = 0.35
elif "Low" in sensitivity:
    threshold = 0.65
else:
    threshold = 0.50

# These are auto-configured based on dataset size (set after upload when n is known)
lsh_threshold = 0.60   # Conservative — only obvious text matches
max_pairs     = 200_000


# ---------------------------------------------------------------------------
# Step 1 — Upload
# ---------------------------------------------------------------------------
st.header("Step 1 — Upload")

col_up, col_tip = st.columns([2, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=["csv", "pdf", "html", "htm", "json", "txt"],
        accept_multiple_files=True,
    )
with col_tip:
    st.markdown("""
    **Supported formats:** CSV · PDF · HTML · JSON · TXT

    **Auto-detected fields**
    `vendor · description · amount · date · department · contract_id`

    Try `data/sample_contracts.csv` to see a full demo.
    """)

if not uploaded_files:
    st.stop()


# ---------------------------------------------------------------------------
# File fingerprint — invalidate cache when files change
# ---------------------------------------------------------------------------
file_buffers: dict[str, bytes] = {}
for f in uploaded_files:
    file_buffers[f.name] = f.read()
    f.seek(0)

current_sig = hashlib.md5(
    "|".join(
        f"{name}:{hashlib.md5(buf).hexdigest()}"
        for name, buf in sorted(file_buffers.items())
    ).encode()
).hexdigest()

if st.session_state.get("_file_sig") != current_sig:
    for key in ("df_norm", "quality_report", "df_clean", "removed_df",
                "candidates", "scored_df", "results", "dataset_summary"):
        st.session_state.pop(key, None)
    st.session_state["_file_sig"] = current_sig


# ---------------------------------------------------------------------------
# Stage 1–2: Ingest + Normalise (cached per file fingerprint)
# ---------------------------------------------------------------------------
if "df_norm" not in st.session_state:
    all_records: list = []
    all_logs: list    = []

    ingest_status = st.status("Reading files…", expanded=True)
    prog_bar      = st.progress(0.0)

    def ingest_cb(done, total, msg):
        prog_bar.progress(min(1.0, done / max(total, 1)))
        ingest_status.write(msg)

    for f in uploaded_files:
        # Feed bytes back to ingestor so file pointer is fresh
        import io as _io
        fake_file = _io.BytesIO(file_buffers[f.name])
        fake_file.name = f.name
        records, logs = ingest_file(fake_file, progress_cb=ingest_cb)
        all_records.extend(records)
        all_logs.extend(logs)

    prog_bar.progress(1.0)
    ingest_status.update(
        label=f"Extraction complete — {len(all_records):,} raw records.",
        state="complete",
        expanded=False,
    )

    if not all_records:
        st.error("No records extracted. Check file format.")
        st.stop()

    with st.spinner("Normalising fields…"):
        df_norm = normalize_records(pd.DataFrame(all_records))

    st.session_state["df_norm"]    = df_norm
    st.session_state["ingest_log"] = all_logs
else:
    df_norm = st.session_state["df_norm"]


# ---------------------------------------------------------------------------
# Step 2 — Data Quality Report
# ---------------------------------------------------------------------------
st.header("Step 2 — Data Quality")

if "quality_report" not in st.session_state:
    with st.spinner("Assessing data quality…"):
        qr = assess_quality(df_norm)
    st.session_state["quality_report"] = qr
else:
    qr = st.session_state["quality_report"]

# Summary metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records",      f"{qr.total_records:,}")
m2.metric("Quality Score",      f"{qr.overall_score:.0%}")
m3.metric("Duplicates Found",   f"{qr.duplicate_count:,}")
m4.metric("Unique Vendors",     df_norm["vendor_normalized"].nunique())
m5.metric("Short Descriptions", f"{qr.vocabulary_sparsity:.0%}",
          help="Fraction of descriptions with fewer than 5 words.")

# Field completeness bar chart
field_names  = [r.name for r in qr.field_reports]
completeness = [r.completeness for r in qr.field_reports]
fig_q = go.Figure(go.Bar(
    x=field_names, y=completeness,
    marker_color=["#1565C0" if c >= 0.80 else "#f57c00" if c >= 0.50 else "#d32f2f"
                  for c in completeness],
    text=[f"{c:.0%}" for c in completeness],
    textposition="outside",
))
fig_q.update_layout(
    title="Field Completeness",
    yaxis=dict(range=[0, 1.15], tickformat=".0%"),
    height=260, margin=dict(t=40, b=0, l=0, r=0),
)
st.plotly_chart(fig_q, use_container_width=True)

# Column mapping diagnostics — always show so user can verify detection
col_mapping = df_norm.attrs.get("column_mapping", {})
unmapped    = df_norm.attrs.get("unmapped_cols", [])
if col_mapping:
    mapped_rows   = [(orig, std) for orig, std in col_mapping.items() if std != "(unmapped)"]
    unmapped_rows = [(orig, "(not used)") for orig in unmapped]

    with st.expander(
        f"Column detection — {len(mapped_rows)} field(s) mapped, {len(unmapped_rows)} unmapped",
        expanded=(len(mapped_rows) == 0),   # expand if nothing was mapped (problem indicator)
    ):
        if mapped_rows:
            st.markdown("**Mapped fields**")
            st.dataframe(
                pd.DataFrame(mapped_rows, columns=["Your column", "Standard field"]),
                hide_index=True, use_container_width=True,
            )
        if unmapped_rows:
            st.markdown("**Unmapped columns** (not used in analysis)")
            st.dataframe(
                pd.DataFrame(unmapped_rows, columns=["Your column", "Status"]),
                hide_index=True, use_container_width=True,
            )
        if len(mapped_rows) == 0:
            st.warning(
                "None of your column names matched known aliases. "
                "The system will attempt catch-all detection and fall back to "
                "random-sample blocking. For best results, rename key columns to: "
                "`vendor_name`, `description`, `award_amount`, `award_date`, `department`."
            )

# Issues / warnings / blocking notes
if qr.critical_issues:
    for issue in qr.critical_issues:
        st.error(f"**Critical:** {issue}")
if qr.warnings:
    with st.expander(f"{len(qr.warnings)} warning(s)", expanded=False):
        for w in qr.warnings:
            st.warning(w)
if qr.blocking_notes:
    with st.expander("Blocking strategy notes", expanded=False):
        for note in qr.blocking_notes:
            st.info(note)
if qr.recommendations:
    with st.expander("Recommendations for better results", expanded=False):
        for rec in qr.recommendations:
            st.markdown(f"- {rec}")

with st.expander("View normalised records", expanded=False):
    display_cols = [c for c in df_norm.columns
                    if not c.endswith(("_parsed", "_numeric", "_normalized"))]
    st.dataframe(df_norm[display_cols], use_container_width=True)

with st.expander("Extraction log", expanded=False):
    st.dataframe(pd.DataFrame(st.session_state.get("ingest_log", [])),
                 use_container_width=True)

if len(df_norm) < 2:
    st.warning("At least 2 records are needed to run analysis.")
    st.stop()

st.divider()


# ---------------------------------------------------------------------------
# Step 3 — Run Analysis
# ---------------------------------------------------------------------------
st.header("Step 3 — Run Analysis")

# --- Pre-run size guidance ---
n_records = len(df_norm)
if n_records > 50_000:
    est = "10–30 minutes"
    mem_warn = True
elif n_records > 10_000:
    est = "2–5 minutes"
    mem_warn = False
elif n_records > 1_000:
    est = "15–60 seconds"
    mem_warn = False
else:
    est = "a few seconds"
    mem_warn = False

st.info(
    f"**{n_records:,} records loaded.** "
    f"Estimated analysis time: **{est}**. "
    + ("LSH text-similarity blocking is automatically disabled for large datasets "
       "to protect memory — structural blocking (vendor, department, date, amount) will be used instead."
       if n_records > 3_000 else "")
)
if mem_warn:
    st.warning(
        "This is a very large dataset. Analysis may take a while and use significant memory. "
        "Consider uploading a filtered subset (e.g. one department or one year) for a faster demo."
    )

st.markdown("**What will happen when you click Run:**")
st.markdown(
    "1. Remove exact structural duplicates from the dataset  \n"
    "2. Group records into candidate pairs using vendor, department, date, and amount similarity  \n"
    "3. Score every candidate pair across all four dimensions  \n"
    "4. Rank and explain the highest-risk cases  \n"
    "5. Show you findings with evidence — nothing is flagged without a reason"
)

run_clicked = st.button("▶  Run Overlap Analysis", type="primary")

if run_clicked:
    # Clear previous results so re-runs with new settings work correctly
    for key in ("df_clean", "removed_df", "candidates", "scored_df", "results", "dataset_summary"):
        st.session_state.pop(key, None)

    with st.status("Running pipeline…", expanded=True) as pipeline_status:

        # --- Deduplication ---
        st.write("Step 1/4 — Checking for structural duplicates…")
        df_clean, removed_df = deduplicate(df_norm)
        if len(removed_df) > 0:
            st.write(
                f"Found {len(removed_df):,} rows with identical vendor + amount + date. "
                f"These look like duplicate export rows (same transaction reported twice), "
                f"not separate contracts. They've been set aside. "
                f"{len(df_clean):,} unique records remain for analysis."
            )
        else:
            st.write(f"No structural duplicates found. All {len(df_clean):,} records are distinct.")

        # --- Blocking ---
        st.write("Step 2/4 — Identifying candidate pairs to compare…")
        block_prog = st.progress(0.0, text="Building candidate pairs…")

        def block_cb(done, total, msg):
            block_prog.progress(min(1.0, done / max(total, 1)), text=msg)

        candidates = build_candidate_pairs(
            df_clean,
            lsh_threshold=lsh_threshold,
            max_total_pairs=max_pairs,
            progress_cb=block_cb,
        )
        block_prog.progress(1.0, text=f"Candidate pairs: {len(candidates):,}")
        st.write(
            f"Identified **{len(candidates):,}** record pairs worth comparing "
            f"(out of {len(df_clean):,} × {len(df_clean):,} = "
            f"{len(df_clean)**2:,} possible combinations). "
            f"Blocking reduced comparisons by "
            f"**{max(0, 100 - len(candidates)*100//(len(df_clean)**2 or 1))}%**."
        )

        if not candidates:
            st.error("No candidate pairs generated — the dataset may have too few populated fields.")
            pipeline_status.update(label="Analysis failed.", state="error")
            st.stop()

        # --- Scoring ---
        st.write("Step 3/4 — Scoring each candidate pair…")
        score_prog = st.progress(0.0, text="Scoring pairs…")

        def score_cb(done, total):
            pct = done / max(total, 1)
            score_prog.progress(
                min(1.0, pct),
                text=f"Scoring batches: {done}/{total} complete ({pct:.0%})",
            )

        scored_df = score_candidates(
            df_clean,
            candidates,
            weights=weights,
            progress_cb=score_cb,
        )
        score_prog.progress(1.0, text="Scoring complete.")
        st.write(f"Scored {len(scored_df):,} pairs across 4 dimensions.")

        # --- Explanations ---
        st.write("Step 4/4 — Ranking findings and writing explanations…")
        results = generate_explanations(scored_df, df_clean, threshold)

        pipeline_status.update(
            label="Analysis complete.", state="complete", expanded=False
        )

    # Cache results
    st.session_state["df_clean"]        = df_clean
    st.session_state["removed_df"]      = removed_df
    st.session_state["candidates"]      = candidates
    st.session_state["scored_df"]       = scored_df
    st.session_state["results"]         = results
    st.session_state["dataset_summary"] = {
        "total_records":  len(df_norm),
        "pairs_analysed": len(scored_df),
    }

if "results" not in st.session_state:
    st.info("Click **Run Overlap Analysis** to start.")
    st.stop()


# ---------------------------------------------------------------------------
# Step 4 — Findings
# ---------------------------------------------------------------------------
results:   list         = st.session_state["results"]
scored_df: pd.DataFrame = st.session_state["scored_df"]
summary:   dict         = st.session_state.get("dataset_summary", {})

st.header("Step 4 — Findings")

flagged     = [r for r in results if r["above_threshold"]]
total_pairs = len(results)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pairs Scored",        f"{total_pairs:,}")
c2.metric("Cases Flagged",       len(flagged),
          help=f"Risk score ≥ {threshold:.0%}")
c3.metric("High Risk  (≥75%)",   sum(1 for r in flagged if r["risk_score"] >= 0.75))
c4.metric("Medium Risk (50–74%)", sum(1 for r in flagged if 0.50 <= r["risk_score"] < 0.75))

# Risk distribution chart
if results:
    scores = [r["risk_score"] for r in results]
    fig_d  = go.Figure()
    fig_d.add_trace(go.Histogram(
        x=scores, nbinsx=40,
        marker_color=[
            "#c62828" if s >= 0.75 else "#e65100" if s >= 0.50 else "#90caf9"
            for s in scores
        ],
    ))
    fig_d.add_vline(
        x=threshold, line_dash="dash", line_color="#333",
        annotation_text=f"Threshold ({threshold:.0%})",
        annotation_position="top right",
    )
    fig_d.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Number of Pairs",
        height=240,
        margin=dict(t=40, b=0, l=0, r=0),
        showlegend=False,
    )
    st.plotly_chart(fig_d, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Flagged case cards
# ---------------------------------------------------------------------------
if not flagged:
    st.success(
        f"No pairs exceeded the {threshold:.0%} risk threshold. "
        "Try lowering the threshold or reviewing data quality."
    )
else:
    st.markdown(
        f"**{len(flagged)} case(s) flagged for analyst review**, ranked by risk score."
    )

    for i, case in enumerate(flagged):
        risk  = case["risk_score"]
        badge = ("🔴 HIGH RISK"   if risk >= 0.75 else
                 "🟡 MEDIUM RISK" if risk >= 0.50 else
                 "🟢 LOW RISK")

        vendor_label = (
            case["vendor_a"] if case["vendor_a"] == case["vendor_b"]
            else f"{case['vendor_a']}  ↔  {case['vendor_b']}"
        )

        with st.expander(
            f"{badge}  |  {risk:.0%}  |  {vendor_label}  —  {case['pair_id']}",
            expanded=(i == 0),
        ):
            ea, eb = st.columns(2)
            with ea:
                st.markdown("**Contract A**")
                st.json(case["record_a"], expanded=True)
            with eb:
                st.markdown("**Contract B**")
                st.json(case["record_b"], expanded=True)

            st.markdown("---")
            st.markdown("**Why flagged**")
            for signal in case["signals"]:
                st.markdown(f"- {signal}")

            st.markdown("**Score breakdown**")
            bd   = case["score_breakdown"]
            scols = st.columns(4)
            for col, (dim, val) in zip(scols, bd.items()):
                col.metric(dim.capitalize(), f"{val:.0%}")

            st.progress(min(1.0, risk), text=f"Overall risk: {risk:.0%}")
            st.markdown("---")

            if risk >= 0.75:
                st.error(f"**Recommendation:** {case['recommendation']}")
            elif risk >= 0.50:
                st.warning(f"**Recommendation:** {case['recommendation']}")
            else:
                st.info(f"**Recommendation:** {case['recommendation']}")

# ---------------------------------------------------------------------------
# Step 5 — Export
# ---------------------------------------------------------------------------
if flagged:
    st.divider()
    st.header("Step 5 — Export")

    ex1, ex2, ex3 = st.columns(3)

    with ex1:
        st.download_button(
            "⬇️ JSON (audit trail)",
            data=to_json(flagged),
            file_name="flagged_cases.json",
            mime="application/json",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            "⬇️ CSV (Excel-ready)",
            data=to_csv(flagged),
            file_name="flagged_cases.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with ex3:
        st.download_button(
            "⬇️ HTML Report (shareable)",
            data=to_html_report(flagged, summary),
            file_name="govspend_report.html",
            mime="text/html",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "GovSpend Analyzer v2 · Hackathon Prototype · "
    "Outputs are indicative only and require qualified analyst review."
)
