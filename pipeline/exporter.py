"""
Export module.

Converts flagged findings into three formats:
  - JSON  : machine-readable, audit-trail friendly
  - CSV   : flat table, opens in Excel
  - HTML  : standalone report for sharing with non-technical reviewers
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_json(flagged: List[Dict[str, Any]]) -> str:
    """Serialise findings to pretty-printed JSON."""
    return json.dumps(flagged, indent=2, default=str)


def to_csv(flagged: List[Dict[str, Any]]) -> str:
    """Flatten findings into a CSV table, one row per flagged pair."""
    rows = []
    for case in flagged:
        rows.append({
            "rank":               case.get("rank", ""),
            "pair_id":            case["pair_id"],
            "risk_score":         f"{case['risk_score']:.2%}",
            "vendor_a":           case["vendor_a"],
            "vendor_b":           case["vendor_b"],
            "dept_a":             case["record_a"].get("department", ""),
            "dept_b":             case["record_b"].get("department", ""),
            "amount_a":           case["record_a"].get("amount", ""),
            "amount_b":           case["record_b"].get("amount", ""),
            "date_a":             case["record_a"].get("date", ""),
            "date_b":             case["record_b"].get("date", ""),
            "desc_similarity":    f"{case['score_breakdown']['description']:.2%}",
            "vendor_similarity":  f"{case['score_breakdown']['vendor']:.2%}",
            "date_proximity":     f"{case['score_breakdown']['date']:.2%}",
            "amount_similarity":  f"{case['score_breakdown']['amount']:.2%}",
            "primary_signal":     case["signals"][0].replace("**", "") if case["signals"] else "",
            "recommendation":     case["recommendation"],
        })
    return pd.DataFrame(rows).to_csv(index=False)


def to_html_report(
    flagged: List[Dict[str, Any]],
    dataset_summary: Dict[str, Any],
) -> str:
    """Generate a standalone HTML report suitable for emailing to reviewers."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows_html = ""
    for case in flagged:
        risk = case["risk_score"]
        color = "#c62828" if risk >= 0.75 else "#e65100" if risk >= 0.50 else "#2e7d32"
        signals_html = "".join(
            f"<li>{s.replace('**', '<strong>').replace('**', '</strong>')}</li>"
            for s in case["signals"]
        )
        rows_html += f"""
        <tr>
          <td style="text-align:center"><strong>{case.get('rank', '')}</strong></td>
          <td style="text-align:center;color:{color};font-weight:700">{risk:.0%}</td>
          <td>{case['vendor_a']}</td>
          <td>{case['vendor_b']}</td>
          <td style="font-size:12px"><ul style="margin:0;padding-left:16px">{signals_html}</ul></td>
          <td style="font-size:12px;font-style:italic">{case['recommendation']}</td>
        </tr>
        """

    high   = sum(1 for c in flagged if c["risk_score"] >= 0.75)
    medium = sum(1 for c in flagged if 0.50 <= c["risk_score"] < 0.75)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GovSpend Overlap Report — {now}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body    {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
               padding: 40px; color: #1a1a1a; max-width: 1200px; margin: 0 auto; }}
    h1      {{ color: #0d47a1; margin-bottom: 4px; }}
    h2      {{ color: #333; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 32px; }}
    .meta   {{ background: #f5f7fa; border: 1px solid #e0e0e0; border-radius: 8px;
               padding: 16px 24px; display: flex; gap: 32px; flex-wrap: wrap; margin: 16px 0 24px; }}
    .meta-item {{ text-align: center; }}
    .meta-item .val {{ font-size: 28px; font-weight: 700; color: #0d47a1; }}
    .meta-item .lbl {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: .05em; }}
    table   {{ border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }}
    th      {{ background: #0d47a1; color: white; padding: 10px 14px; text-align: left;
               position: sticky; top: 0; }}
    td      {{ padding: 10px 14px; border-bottom: 1px solid #eeeeee; vertical-align: top; }}
    tr:hover {{ background: #fafafa; }}
    .badge-high   {{ background:#ffebee; color:#c62828; padding:2px 8px; border-radius:12px; font-weight:700; }}
    .badge-medium {{ background:#fff3e0; color:#e65100; padding:2px 8px; border-radius:12px; font-weight:700; }}
    .disclaimer {{ font-size: 11px; color: #999; margin-top: 40px;
                   border-top: 1px solid #e0e0e0; padding-top: 16px; }}
  </style>
</head>
<body>
  <h1>🏛️ GovSpend Overlap Analyzer — Findings Report</h1>
  <p style="color:#555">Generated {now} &nbsp;·&nbsp; Decision-support tool — all findings require analyst review.</p>

  <div class="meta">
    <div class="meta-item">
      <div class="val">{dataset_summary.get('total_records', 'N/A'):,}</div>
      <div class="lbl">Total Records</div>
    </div>
    <div class="meta-item">
      <div class="val">{dataset_summary.get('pairs_analysed', 'N/A'):,}</div>
      <div class="lbl">Pairs Analysed</div>
    </div>
    <div class="meta-item">
      <div class="val">{len(flagged)}</div>
      <div class="lbl">Cases Flagged</div>
    </div>
    <div class="meta-item">
      <div class="val" style="color:#c62828">{high}</div>
      <div class="lbl">High Risk (≥75%)</div>
    </div>
    <div class="meta-item">
      <div class="val" style="color:#e65100">{medium}</div>
      <div class="lbl">Medium Risk (50–74%)</div>
    </div>
  </div>

  <h2>Flagged Cases</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Risk Score</th>
        <th>Vendor A</th>
        <th>Vendor B</th>
        <th>Signals</th>
        <th>Recommendation</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div class="disclaimer">
    This report was generated automatically by GovSpend Overlap Analyzer, a decision-support tool.
    All findings are indicative only and require qualified analyst review before any action is taken.
    No automated procurement decisions are made by this system.
  </div>
</body>
</html>"""
