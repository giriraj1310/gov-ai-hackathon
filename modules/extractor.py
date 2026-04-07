"""
Data extraction module.
Handles CSV, PDF, HTML, JSON, and TXT inputs.
Returns a list of raw record dicts and a log message.
"""
import io
import json
import pandas as pd
from typing import List, Tuple, Dict


def extract_data(file) -> Tuple[List[Dict], str]:
    """
    Dispatch extraction based on file extension.
    Returns (records, log_message).
    """
    name = file.name.lower()

    if name.endswith(".csv"):
        return _extract_csv(file)
    elif name.endswith(".pdf"):
        return _extract_pdf(file)
    elif name.endswith((".html", ".htm")):
        return _extract_html(file)
    elif name.endswith(".json"):
        return _extract_json(file)
    elif name.endswith(".txt"):
        return _extract_txt(file)
    else:
        return [], f"Unsupported file type: {name}"


# ---------------------------------------------------------------------------
# Format-specific handlers
# ---------------------------------------------------------------------------

def _extract_csv(file) -> Tuple[List[Dict], str]:
    try:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        records = df.where(pd.notna(df), None).to_dict(orient="records")
        return records, f"Extracted {len(records)} rows."
    except Exception as e:
        return [], f"CSV parse error: {e}"


def _extract_pdf(file) -> Tuple[List[Dict], str]:
    try:
        import pdfplumber
    except ImportError:
        return [], "pdfplumber not installed. Run: pip install pdfplumber"

    try:
        records = []
        raw_bytes = file.read()

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                # Prefer structured tables
                for table in page.extract_tables() or []:
                    if not table or len(table) < 2:
                        continue
                    headers = [
                        str(h).strip().lower().replace(" ", "_") if h else f"col_{i}"
                        for i, h in enumerate(table[0])
                    ]
                    for row in table[1:]:
                        record = {
                            headers[i]: str(cell).strip() if cell else ""
                            for i, cell in enumerate(row)
                            if i < len(headers)
                        }
                        records.append(record)

                # Fall back to raw text per page if no tables found
                if not records:
                    text = page.extract_text() or ""
                    if text.strip():
                        records.append({"raw_text": text, "source": "pdf_page"})

        return records, f"Extracted {len(records)} record(s)/block(s) from PDF."
    except Exception as e:
        return [], f"PDF parse error: {e}"


def _extract_html(file) -> Tuple[List[Dict], str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [], "beautifulsoup4 not installed. Run: pip install beautifulsoup4"

    try:
        content = file.read().decode("utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        records = []

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            headers = [
                cell.get_text(strip=True).lower().replace(" ", "_")
                for cell in rows[0].find_all(["th", "td"])
            ]
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
                if len(cells) == len(headers):
                    records.append(dict(zip(headers, cells)))

        if records:
            return records, f"Extracted {len(records)} rows from HTML table(s)."

        # No tables — fall back to raw text
        text = soup.get_text(separator="\n", strip=True)[:3000]
        return [{"raw_text": text, "source": "html_text"}], "No HTML tables found; extracted raw text."

    except Exception as e:
        return [], f"HTML parse error: {e}"


def _extract_json(file) -> Tuple[List[Dict], str]:
    try:
        data = json.load(file)
        if isinstance(data, list):
            return data, f"Extracted {len(data)} records."
        if isinstance(data, dict):
            for key in ("records", "contracts", "data", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key], f"Extracted {len(data[key])} records from '{key}'."
            return [data], "Wrapped single JSON object as one record."
        return [], "Unrecognized JSON structure."
    except Exception as e:
        return [], f"JSON parse error: {e}"


def _extract_txt(file) -> Tuple[List[Dict], str]:
    content = file.read().decode("utf-8", errors="ignore")
    return [{"raw_text": content[:5000], "source": "txt_file"}], "Extracted as raw text block."
