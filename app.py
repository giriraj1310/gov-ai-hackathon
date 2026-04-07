"""
GovSpend Overlap Analyzer
An analyst copilot for government procurement oversight.
"""
import hashlib
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline.ingestor    import ingest_file
from pipeline.normalizer  import normalize_records
from pipeline.deduplicator import deduplicate
from pipeline.blocker     import build_candidate_pairs
from pipeline.scorer      import score_candidates
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
div[data-testid="stExpander"] { border-left: 3px solid #1565C0; margin-bottom: 6px; }
.finding-high   { border-left: 4px solid #c62828 !important; }
.finding-medium { border-left: 4px solid #e65100 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏛️ GovSpend Overlap Analyzer")
st.markdown(
    "Upload government procurement or spending data. "
    "The system finds contracts that may overlap, duplicate, or conflict — "
    "and explains exactly why each case was flagged."
)
st.caption("Every finding is backed by evidence. Nothing is flagged without a reason. "
           "This tool supports analyst review — it does not make decisions.")
st.divider()


# ---------------------------------------------------------------------------
# Sidebar — plain-language controls only
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    st.markdown("**What should the system look for?**")
    focus = st.radio(
        "focus",
        options=[
            "Balanced — vendor + description + date + value",
            "Vendor-led — flag same or similar suppliers",
            "Description-led — flag similar scope of work",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**How many results do you want to see?**")
    top_n = st.select_slider(
        "top_n",
        options=[10, 20, 50, 100, 250],
        value=50,
        label_visibility="collapsed",
        help="Shows the highest-scoring cases first. Start with 50.",
    )

    st.divider()
    st.markdown("**Minimum confidence to show a finding**")
    sensitivity_label = st.select_slider(
        "sensitivity",
        options=["Show more (lower bar)", "Balanced", "Show only clear cases"],
        value="Balanced",
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "The system automatically handles large files, removes duplicate export rows, "
        "and collapses contract amendments before analysis."
    )

# Translate settings to internal parameters
if "Vendor-led" in focus:
    weights = {"description": 0.20, "vendor": 0.55, "date": 0.15, "amount": 0.10}
elif "Description-led" in focus:
    weights = {"description": 0.60, "vendor": 0.15, "date": 0.15, "amount": 0.10}
else:
    weights = {"description": 0.40, "vendor": 0.30, "date": 0.15, "amount": 0.15}

threshold = {"Show more (lower bar)": 0.55, "Balanced": 0.65, "Show only clear cases": 0.78}[sensitivity_label]

lsh_threshold = 0.65
max_pairs     = 200_000


# ---------------------------------------------------------------------------
# Step 1 — Upload
# ---------------------------------------------------------------------------
st.header("Step 1 — Upload your file")

col_up, col_help = st.columns([2, 1])
with col_up:
    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=["csv", "pdf", "html", "htm", "json", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
with col_help:
    st.markdown("""
    **Accepted formats**
    CSV · PDF · HTML · JSON · TXT

    **Automatically detected fields**
    Supplier/vendor · Description · Value · Date · Department

    Works with CanadaBuys, USASpending, UK Contracts Finder,
    and most government procurement exports.
    """)

if not uploaded_files:
    st.info("Upload a file above to get started. You can use the sample in `data/sample_contracts.csv`.")
    st.stop()


# ---------------------------------------------------------------------------
# File fingerprint — invalidate cached stages when files change
# ---------------------------------------------------------------------------
file_buffers: dict[str, bytes] = {}
for f in uploaded_files:
    file_buffers[f.name] = f.read()
    f.seek(0)

current_sig = hashlib.md5(
    "|".join(f"{n}:{hashlib.md5(b).hexdigest()}"
             for n, b in sorted(file_buffers.items())).encode()
).hexdigest()

if st.session_state.get("_file_sig") != current_sig:
    for k in ("df_norm", "dedup_audit", "df_clean", "candidates",
              "scored_df", "results", "ingest_log"):
        st.session_state.pop(k, None)
    st.session_state["_file_sig"] = current_sig


# ---------------------------------------------------------------------------
# Ingest + Normalise (cached per file fingerprint)
# ---------------------------------------------------------------------------
if "df_norm" not in st.session_state:
    all_records, all_logs = [], []

    status = st.status("Reading your file…", expanded=True)
    bar    = st.progress(0.0)

    def _ingest_cb(done, total, msg):
        bar.progress(min(1.0, done / max(total, 1)))
        status.write(msg)

    for f in uploaded_files:
        fake = io.BytesIO(file_buffers[f.name])
        fake.name = f.name
        recs, logs = ingest_file(fake, progress_cb=_ingest_cb)
        all_records.extend(recs)
        all_logs.extend(logs)

    bar.progress(1.0)
    status.update(label=f"File read — {len(all_records):,} rows extracted.", state="complete", expanded=False)

    if not all_records:
        st.error("No records could be read from your file. Please check the format.")
        st.stop()

    with st.spinner("Organising fields…"):
        df_norm = normalize_records(pd.DataFrame(all_records))

    st.session_state["df_norm"]    = df_norm
    st.session_state["ingest_log"] = all_logs
else:
    df_norm = st.session_state["df_norm"]


# ---------------------------------------------------------------------------
# Data summary — shown inline, not as a separate step
# ---------------------------------------------------------------------------
col_mapping = df_norm.attrs.get("column_mapping", {})
mapped      = {o: s for o, s in col_mapping.items() if s != "(unmapped)"}
unmapped    = df_norm.attrs.get("unmapped_cols", [])

n_records = len(df_norm)
n_vendors = df_norm["vendor_normalized"].nunique()
has_amounts = df_norm["amount_numeric"].notna().sum()
has_dates   = df_norm["date_parsed"].notna().sum()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Records loaded",   f"{n_records:,}")
m2.metric("Unique suppliers", f"{n_vendors:,}")
m3.metric("With values",      f"{has_amounts:,}")
m4.metric("With dates",       f"{has_dates:,}")

# Show field detection status — immediately visible, plain language
if mapped:
    detected = ", ".join(f"**{v}** (`{k}`)" for k, v in list(mapped.items())[:6])
    st.success(f"Detected fields: {detected}")
else:
    st.warning(
        "Could not automatically identify key fields (supplier, value, date) in your file. "
        "Analysis will run using all available text. "
        "For best results, ensure columns are labelled: "
        "`vendor_name`, `description`, `award_amount`, `award_date`, `department`."
    )

with st.expander("See all detected columns", expanded=False):
    if mapped:
        st.dataframe(pd.DataFrame(
            [{"Your column": k, "Used as": v} for k, v in mapped.items()]
        ), hide_index=True, use_container_width=True)
    if unmapped:
        st.caption(f"{len(unmapped)} column(s) not used: {', '.join(unmapped[:10])}")

with st.expander("Preview the data", expanded=False):
    show_cols = [c for c in df_norm.columns if not c.endswith(("_parsed","_numeric","_normalized","record_id"))]
    st.dataframe(df_norm[show_cols].head(100), use_container_width=True)

st.divider()


# ---------------------------------------------------------------------------
# Step 2 — Run Analysis
# ---------------------------------------------------------------------------
st.header("Step 2 — Find overlaps")

# Estimate and guide the user
if n_records > 50_000:
    est_time = "15–30 minutes"
elif n_records > 10_000:
    est_time = "2–5 minutes"
elif n_records > 1_000:
    est_time = "15–60 seconds"
else:
    est_time = "a few seconds"

st.markdown(
    f"The system will compare your **{n_records:,} contracts** to find cases where "
    f"the same supplier may have received overlapping or duplicate awards. "
    f"Estimated time: **{est_time}**."
)

if n_records > 3_000:
    st.info(
        "For large datasets, the system uses fast structural matching (supplier name, "
        "department, value range, date window) instead of full text comparison. "
        "This keeps analysis fast and memory-efficient."
    )

run_clicked = st.button("▶  Run Analysis", type="primary", use_container_width=False)

if run_clicked:
    for k in ("df_clean", "dedup_audit", "candidates", "scored_df", "results"):
        st.session_state.pop(k, None)

    with st.status("Analysing contracts…", expanded=True) as pipeline_status:

        # Step 1: deduplicate
        st.write("Cleaning the dataset…")
        df_clean, audit = deduplicate(df_norm)

        msgs = []
        if audit["amendments_collapsed"] > 0:
            msgs.append(
                f"Found **{audit['amendment_groups']:,}** contracts with amendments "
                f"({audit['amendments_collapsed']:,} older versions removed — keeping latest values only)."
            )
        if audit["structural_dups"] > 0:
            msgs.append(
                f"Found **{audit['structural_dups']:,}** rows that appear to be the same transaction "
                f"exported more than once — removed."
            )
        if msgs:
            for m in msgs:
                st.write(m)
        else:
            st.write(f"Dataset is clean — all {len(df_clean):,} records are distinct.")

        # Step 2: block
        st.write(f"Identifying which of the {len(df_clean):,} contracts to compare…")
        block_prog = st.progress(0.0)

        def _block_cb(done, total, msg):
            block_prog.progress(min(1.0, done / max(total, 1)), text=msg)

        candidates = build_candidate_pairs(
            df_clean,
            lsh_threshold=lsh_threshold,
            max_total_pairs=max_pairs,
            progress_cb=_block_cb,
        )
        block_prog.progress(1.0)
        total_possible = len(df_clean) * (len(df_clean) - 1) // 2
        pct_reduction  = max(0, 100 - len(candidates) * 100 // max(total_possible, 1))
        st.write(
            f"Selected **{len(candidates):,}** contract pairs to examine "
            f"(filtered down from {total_possible:,} possible combinations — "
            f"{pct_reduction}% reduction)."
        )

        if not candidates:
            st.error("Could not identify any contract pairs to compare. Check that key fields (supplier, date, department) were detected correctly above.")
            pipeline_status.update(label="Analysis could not complete.", state="error")
            st.stop()

        # Step 3: score
        st.write("Scoring each pair…")
        score_prog = st.progress(0.0)

        def _score_cb(done, total):
            score_prog.progress(min(1.0, done / max(total, 1)),
                                text=f"Scoring: {done}/{total} batches ({done*100//max(total,1)}%)")

        scored_df = score_candidates(df_clean, candidates, weights=weights, progress_cb=_score_cb)
        score_prog.progress(1.0)

        # Step 4: explain
        st.write("Preparing findings…")
        results = generate_explanations(scored_df, df_clean, threshold)

        pipeline_status.update(label="Analysis complete.", state="complete", expanded=False)

    st.session_state.update({
        "df_clean":   df_clean,
        "dedup_audit": audit,
        "candidates": candidates,
        "scored_df":  scored_df,
        "results":    results,
    })

if "results" not in st.session_state:
    st.info("Click **Run Analysis** to start.")
    st.stop()


# ---------------------------------------------------------------------------
# Step 3 — Findings
# ---------------------------------------------------------------------------
results:   list         = st.session_state["results"]
scored_df: pd.DataFrame = st.session_state["scored_df"]
df_clean:  pd.DataFrame = st.session_state["df_clean"]
audit:     dict         = st.session_state.get("dedup_audit", {})

st.divider()
st.header("Step 3 — Review findings")

flagged = [r for r in results if r["above_threshold"]]
high    = [r for r in flagged if r["risk_score"] >= 0.80]
medium  = [r for r in flagged if 0.65 <= r["risk_score"] < 0.80]

# Summary sentence — plain English
if not flagged:
    st.success(
        f"We reviewed {len(results):,} contract pairs and found no cases that clearly "
        f"stand out as overlapping or duplicated at the current sensitivity level. "
        f"Try 'Show more' in the sidebar settings to widen the search."
    )
    st.stop()

st.markdown(
    f"We reviewed **{len(scored_df):,}** contract pairs and identified "
    f"**{len(flagged):,}** cases worth investigating."
)

c1, c2, c3 = st.columns(3)
c1.metric("Needs urgent attention", len(high),
          help="Risk score ≥ 80% — strong overlap signals across multiple dimensions")
c2.metric("Recommend review",       len(medium),
          help="Risk score 65–79% — notable similarity in one or more dimensions")
c3.metric("Showing top",            min(top_n, len(flagged)),
          help="Ranked by risk score, highest first")

# Risk distribution
scores = [r["risk_score"] for r in results]
fig = go.Figure(go.Histogram(
    x=scores, nbinsx=30,
    marker_color=["#c62828" if s >= 0.80 else "#e65100" if s >= 0.65 else "#90caf9"
                  for s in scores],
))
fig.add_vline(x=threshold, line_dash="dash", line_color="#555",
              annotation_text="Current threshold", annotation_position="top right")
fig.update_layout(
    title="How contract pairs scored (higher = more suspicious)",
    xaxis_title="Suspicion score",
    yaxis_title="Number of pairs",
    height=220, margin=dict(t=40, b=0, l=0, r=0), showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Show top N findings
display = flagged[:top_n]

for i, case in enumerate(display):
    risk  = case["risk_score"]
    badge = "🔴 Urgent" if risk >= 0.80 else "🟡 Review"

    # Build a plain-English title
    va = case["vendor_a"]
    vb = case["vendor_b"]
    vendor_str = va if va == vb else f"{va}  ↔  {vb}"
    title = f"{badge}  |  {risk:.0%} match  |  {vendor_str}"

    with st.expander(title, expanded=(i == 0)):

        # Evidence — side by side, no JSON, just a clean table
        rec_a = case["record_a"]
        rec_b = case["record_b"]

        ea, eb = st.columns(2)
        with ea:
            st.markdown("**Contract A**")
            for field, label in [("contract_id","Reference"), ("vendor","Supplier"),
                                  ("description","Description"), ("amount","Value"),
                                  ("date","Award date"), ("department","Department")]:
                val = rec_a.get(field)
                if val:
                    st.markdown(f"**{label}:** {val}")
        with eb:
            st.markdown("**Contract B**")
            for field, label in [("contract_id","Reference"), ("vendor","Supplier"),
                                  ("description","Description"), ("amount","Value"),
                                  ("date","Award date"), ("department","Department")]:
                val = rec_b.get(field)
                if val:
                    st.markdown(f"**{label}:** {val}")

        st.markdown("---")
        st.markdown("**Why this was flagged**")
        for signal in case["signals"]:
            st.markdown(f"- {signal}")

        # Score breakdown — plain labels, no percentages unless expanded
        bd = case["score_breakdown"]
        with st.expander("See score details", expanded=False):
            score_rows = [
                {"Signal": "Description similarity",  "Score": f"{bd['description']:.0%}"},
                {"Signal": "Supplier name match",     "Score": f"{bd['vendor']:.0%}"},
                {"Signal": "Award date proximity",    "Score": f"{bd['date']:.0%}"},
                {"Signal": "Contract value similarity","Score": f"{bd['amount']:.0%}"},
                {"Signal": "Overall suspicion score", "Score": f"{risk:.0%}"},
            ]
            st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)

        st.markdown("---")
        rec_txt = case["recommendation"]
        if risk >= 0.80:
            st.error(f"**Recommended action:** {rec_txt}")
        else:
            st.warning(f"**Recommended action:** {rec_txt}")

if len(flagged) > top_n:
    st.info(
        f"Showing top {top_n} of {len(flagged):,} flagged cases. "
        f"Increase 'How many results' in the sidebar to see more, "
        f"or download the full list below."
    )


# ---------------------------------------------------------------------------
# Step 4 — Export
# ---------------------------------------------------------------------------
st.divider()
st.header("Step 4 — Download findings")
st.markdown("Export the findings for further review, audit documentation, or sharing with your team.")

summary = {"total_records": n_records, "pairs_analysed": len(scored_df)}

ex1, ex2, ex3 = st.columns(3)
with ex1:
    st.download_button(
        "⬇️ Excel / CSV",
        data=to_csv(flagged),
        file_name="govspend_findings.csv",
        mime="text/csv",
        use_container_width=True,
        help="Opens in Excel. One row per flagged case.",
    )
with ex2:
    st.download_button(
        "⬇️ Shareable Report (HTML)",
        data=to_html_report(flagged, summary),
        file_name="govspend_report.html",
        mime="text/html",
        use_container_width=True,
        help="A formatted report you can email or print.",
    )
with ex3:
    st.download_button(
        "⬇️ Raw Data (JSON)",
        data=to_json(flagged),
        file_name="govspend_findings.json",
        mime="application/json",
        use_container_width=True,
        help="Full data for audit systems or further processing.",
    )

st.markdown("---")
st.caption(
    "GovSpend Analyzer · All findings are indicative only and require qualified analyst review. "
    "This tool does not make procurement decisions."
)
