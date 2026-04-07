# GovSpend Overlap Analyzer

An analyst copilot for government procurement oversight.
Surfaces potential duplicate contracts, vendor overlaps, and spending anomalies — each finding backed by transparent evidence.

---

## What it does

1. **Ingests** CSV, PDF, HTML, JSON, or TXT contract/spending data
2. **Normalizes** messy fields (vendor names, amounts, dates) into a standard schema
3. **Scores** every pair of contracts across four dimensions:
   - Description similarity (TF-IDF cosine)
   - Vendor match (token Jaccard + containment)
   - Date proximity (decaying window)
   - Amount similarity (min/max ratio)
4. **Ranks** pairs by a configurable weighted risk score
5. **Explains** each flagged case with plain-language signals and an analyst recommendation

The system does **not** make decisions. It reduces noise so analysts can focus on the cases most worth investigating.

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

### 3. Upload data

- Use `data/sample_contracts.csv` to see the full demo flow
- Or upload your own CSV with columns like: `vendor_name`, `description`, `award_date`, `contract_amount`, `department`

---

## Project structure

```
gov-ai-hackathon/
├── app.py                  # Streamlit UI
├── requirements.txt
├── modules/
│   ├── extractor.py        # CSV / PDF / HTML / JSON / TXT parsing
│   ├── normalizer.py       # Column mapping + value cleaning
│   ├── similarity.py       # Pairwise similarity computation
│   ├── scorer.py           # Weighted risk scoring
│   └── explainer.py        # Signal detection + recommendation text
└── data/
    └── sample_contracts.csv
```

---

## Configuring scoring weights

Weights can be adjusted live in the sidebar or by editing the defaults in `modules/scorer.py`:

```python
DEFAULT_WEIGHTS = {
    "description": 0.40,   # semantic overlap of contract scope
    "vendor":      0.30,   # same or similar entity
    "date":        0.15,   # close award timing
    "amount":      0.15,   # similar contract value
}
```

Weights are normalised internally, so they don't need to sum to exactly 1.

---

## Supported input formats

| Format | Extraction method | Notes |
|--------|------------------|-------|
| CSV    | pandas           | Recommended for structured data |
| PDF    | pdfplumber       | Works best when PDF contains tables |
| HTML   | BeautifulSoup    | Extracts `<table>` elements |
| JSON   | stdlib json      | Expects array of objects |
| TXT    | raw read         | Treated as unstructured text |

---

## Sample dataset

`data/sample_contracts.csv` contains 15 synthetic contracts with intentional patterns:

- **GOV-2024-001 / 002**: Same vendor (TechSecure), same department, similar cybersecurity scope → **HIGH**
- **GOV-2024-003 / 004**: Same vendor (Excellence Training), same department, overlapping leadership training → **HIGH**
- **GOV-2024-005 / 006**: Same vendor (NextGen IT), same department, near-identical IT helpdesk scope → **HIGH**
- **GOV-2024-013 / 014**: Same vendor (Apex Consulting), same department, similar policy analysis work → **HIGH**
- **GOV-2024-010 / 015**: Same vendor, same department, related analytics work → **MEDIUM**
- Remaining contracts: distinct vendors, departments, and purposes → **LOW/NONE**

---

## Design principles

- **Explainability first** — every result shows exactly why it was flagged
- **Human in the loop** — recommendations suggest review, never action
- **Transparent scoring** — simple weighted math, no black-box models
- **Graceful degradation** — works with partial or messy data
