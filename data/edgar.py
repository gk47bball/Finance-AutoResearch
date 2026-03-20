"""SEC EDGAR API client for fetching 10-K, 10-Q, and 8-K filings."""

import os
import re
import time
import requests
from data.cache import get_cache

EDGAR_BASE = "https://efts.sec.gov/LATEST"
EDGAR_FILINGS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

# Rate limit: max 10 requests/second
_last_request_time = 0


def _get_headers() -> dict:
    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT", "FinAutoResearch research@example.com")
    return {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}


def _rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < 0.1:
        time.sleep(0.1 - elapsed)
    _last_request_time = time.time()


def _get_cik(ticker: str) -> str | None:
    """Get CIK number for a ticker symbol."""
    cache = get_cache()
    key = f"edgar:cik:{ticker}"
    cached = cache.get(key, ttl_hours=168)
    if cached:
        return cached

    _rate_limit()
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_get_headers(),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        for entry in data.values():
            if entry.get("ticker", "").upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                cache.put(key, cik)
                return cik
    except Exception:
        pass
    return None


def get_recent_filings(ticker: str, form_type: str = "10-K", count: int = 1) -> list[dict]:
    """Get recent filings metadata for a ticker."""
    cache = get_cache()
    key = f"edgar:filings:{ticker}:{form_type}:{count}"
    cached = cache.get(key, ttl_hours=168)
    if cached:
        return cached

    cik = _get_cik(ticker)
    if not cik:
        return []

    _rate_limit()
    try:
        url = f"{EDGAR_FILINGS}/CIK{cik}.json"
        resp = requests.get(url, headers=_get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()

        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        results = []
        for i, form in enumerate(forms):
            if form == form_type and len(results) < count:
                results.append({
                    "form": form,
                    "accession": accessions[i].replace("-", ""),
                    "accession_raw": accessions[i],
                    "date": dates[i],
                    "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                    "cik": cik,
                })

        cache.put(key, results)
        return results
    except Exception:
        return []


def get_filing_text(ticker: str, form_type: str = "10-K") -> str:
    """Download and extract text from the most recent filing of given type."""
    cache = get_cache()
    key = f"edgar:text:{ticker}:{form_type}"
    cached = cache.get(key, ttl_hours=168)
    if cached:
        return cached

    filings = get_recent_filings(ticker, form_type, count=1)
    if not filings:
        return ""

    filing = filings[0]
    cik = filing["cik"].lstrip("0")
    accession = filing["accession"]
    primary_doc = filing["primary_doc"]

    if not primary_doc:
        return ""

    _rate_limit()
    try:
        url = f"{EDGAR_ARCHIVES}/{cik}/{accession}/{primary_doc}"
        resp = requests.get(url, headers=_get_headers(), timeout=30)
        resp.raise_for_status()
        text = resp.text

        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Truncate to ~60K chars for LLM context
        text = text[:60000]

        cache.put(key, text)
        return text
    except Exception:
        return ""


def extract_section(text: str, section_name: str) -> str:
    """Extract a named section from filing text (e.g., 'Risk Factors', 'MD&A')."""
    if not text:
        return ""

    patterns = {
        "risk_factors": [
            r"(?i)item\s+1a[\.\s]*risk\s+factors",
            r"(?i)risk\s+factors",
        ],
        "mdna": [
            r"(?i)item\s+7[\.\s]*management.s\s+discussion",
            r"(?i)management.s\s+discussion\s+and\s+analysis",
        ],
        "business": [
            r"(?i)item\s+1[\.\s]*business",
        ],
    }

    section_patterns = patterns.get(section_name, [])
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            start = match.start()
            # Find next "Item" heading as end boundary
            next_item = re.search(r"(?i)item\s+\d", text[match.end() + 10:])
            if next_item:
                end = match.end() + 10 + next_item.start()
            else:
                end = min(start + 15000, len(text))
            section_text = text[start:end].strip()
            # Truncate to ~10K chars
            return section_text[:10000]

    return ""
