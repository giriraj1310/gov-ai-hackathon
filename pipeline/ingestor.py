"""
Streaming ingestion module.

Reads files in chunks to avoid peak-memory doubling on large datasets.
All formats assemble into a flat List[Dict] of raw string records,
which downstream modules can then normalize.

Progress callbacks receive (chunks_done: int, chunks_total: int, message: str).
"""
import io
import json
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple

CHUNK_SIZE = 5_000   # rows per CSV chunk

ProgressCb = Optional[Callable[[int, int, str], None]]
IngestResult = Tuple[List[Dict], List[Dict]]   # (records, log_entries)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(file, progress_cb: ProgressCb = None) -> IngestResult:
    """
    Read one uploaded file and return (all_records, log).

    Internally reads in chunks / pages to bound peak memory, then assembles
    the full list before returning.  Downstream modules receive one flat list.
    """
    name = file.name.lower()

    if name.endswith(".csv"):
        return _ingest_csv(file, progress_cb)
    elif name.endswith(".pdf"):
        return _ingest_pdf(file, progress_cb)
    elif name.endswith((".html", ".htm")):
        return _ingest_html(file, progress_cb)
    elif name.endswith(".json"):
        return _ingest_json(file, progress_cb)
    elif name.endswith(".txt"):
        return _ingest_txt(file, progress_cb)
    else:
        return [], [{"file": file.name, "status": "error", "notes": f"Unsupported file type: {name}"}]


# ---------------------------------------------------------------------------
# Format handlers
# ---------------------------------------------------------------------------

def _ingest_csv(file, progress_cb: ProgressCb) -> IngestResult:
    raw = file.read()
    file_size = len(raw)
    log: List[Dict] = []

    # Estimate total chunks for progress reporting
    sample = raw[:20_000].decode("utf-8", errors="ignore")
    rows_in_sample = sample.count("\n")
    bytes_per_row = max(1, len(sample) / max(1, rows_in_sample))
    estimated_rows = int(file_size / bytes_per_row)
    estimated_chunks = max(1, estimated_rows // CHUNK_SIZE)

    all_records: List[Dict] = []
    chunks_done = 0

    try:
        reader = pd.read_csv(
            io.BytesIO(raw),
            chunksize=CHUNK_SIZE,
            dtype=str,           # Keep everything as string; normalizer handles types
            on_bad_lines="skip",
        )

        for chunk in reader:
            chunk.columns = [c.strip().lower().replace(" ", "_") for c in chunk.columns]
            records = chunk.where(pd.notna(chunk), None).to_dict(orient="records")
            all_records.extend(records)
            chunks_done += 1

            if progress_cb:
                progress_cb(
                    chunks_done,
                    estimated_chunks,
                    f"Reading CSV — chunk {chunks_done} ({len(all_records):,} rows so far)…",
                )

        log.append({
            "file": file.name,
            "status": "ok",
            "records_extracted": len(all_records),
            "notes": f"Streamed {chunks_done} chunk(s) of ~{CHUNK_SIZE:,} rows each.",
        })

    except Exception as e:
        log.append({"file": file.name, "status": "error", "notes": f"CSV parse error: {e}"})

    return all_records, log


def _ingest_pdf(file, progress_cb: ProgressCb) -> IngestResult:
    try:
        import pdfplumber
    except ImportError:
        return [], [{"file": file.name, "status": "error",
                     "notes": "pdfplumber not installed. Run: pip install pdfplumber"}]

    raw = file.read()
    all_records: List[Dict] = []
    log: List[Dict] = []

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            total_pages = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages):
                if progress_cb:
                    progress_cb(
                        page_num + 1,
                        total_pages,
                        f"Extracting PDF page {page_num + 1}/{total_pages}…",
                    )

                page_records: List[Dict] = []

                # Prefer structured table extraction
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
                        record["_source_page"] = page_num + 1
                        page_records.append(record)

                # Fall back to raw text
                if not page_records:
                    text = page.extract_text() or ""
                    if text.strip():
                        page_records.append({
                            "raw_text": text,
                            "_source_page": page_num + 1,
                        })

                all_records.extend(page_records)

        log.append({
            "file": file.name,
            "status": "ok",
            "records_extracted": len(all_records),
            "notes": f"Processed {total_pages} page(s).",
        })

    except Exception as e:
        log.append({"file": file.name, "status": "error", "notes": f"PDF parse error: {e}"})

    return all_records, log


def _ingest_html(file, progress_cb: ProgressCb) -> IngestResult:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [], [{"file": file.name, "status": "error",
                     "notes": "beautifulsoup4 not installed. Run: pip install beautifulsoup4"}]

    content = file.read().decode("utf-8", errors="ignore")
    soup = BeautifulSoup(content, "html.parser")
    tables = soup.find_all("table")
    all_records: List[Dict] = []
    log: List[Dict] = []

    for t_idx, table in enumerate(tables):
        if progress_cb:
            progress_cb(t_idx + 1, max(1, len(tables)),
                        f"Parsing HTML table {t_idx + 1}/{len(tables)}…")

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
                all_records.append(dict(zip(headers, cells)))

    if not all_records:
        text = soup.get_text(separator="\n", strip=True)[:5_000]
        all_records = [{"raw_text": text}]
        notes = "No HTML tables found; extracted raw text."
    else:
        notes = f"Extracted from {len(tables)} table(s)."

    log.append({
        "file": file.name,
        "status": "ok",
        "records_extracted": len(all_records),
        "notes": notes,
    })

    return all_records, log


def _ingest_json(file, progress_cb: ProgressCb) -> IngestResult:
    log: List[Dict] = []
    try:
        data = json.load(file)
    except Exception as e:
        return [], [{"file": file.name, "status": "error", "notes": f"JSON parse error: {e}"}]

    if isinstance(data, dict):
        for key in ("records", "contracts", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = [data]

    if not isinstance(data, list):
        return [], [{"file": file.name, "status": "error", "notes": "Unrecognised JSON structure."}]

    if progress_cb:
        progress_cb(1, 1, f"Loaded {len(data):,} JSON records.")

    log.append({
        "file": file.name,
        "status": "ok",
        "records_extracted": len(data),
        "notes": "JSON loaded.",
    })
    return data, log


def _ingest_txt(file, progress_cb: ProgressCb) -> IngestResult:
    content = file.read().decode("utf-8", errors="ignore")
    if progress_cb:
        progress_cb(1, 1, "Loaded text file.")
    return (
        [{"raw_text": content[:10_000]}],
        [{"file": file.name, "status": "ok", "records_extracted": 1,
          "notes": "Loaded as raw text block."}],
    )
