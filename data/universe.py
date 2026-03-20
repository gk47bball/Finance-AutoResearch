"""Universe construction — fetches S&P 500 tickers."""

import requests
import pandas as pd
from data.cache import get_cache


def get_sp500_tickers() -> list[str]:
    cache = get_cache()
    cached = cache.get("universe:sp500", ttl_hours=168)
    if cached is not None:
        return cached

    tickers = []

    # Method 1: Wikipedia via requests with proper headers
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "FinAutoResearch/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        tickers = sorted(set(tickers))
    except Exception:
        pass

    # Method 2: Fallback to a hardcoded core list if Wikipedia fails
    if not tickers:
        tickers = _fallback_sp500()

    cache.put("universe:sp500", tickers)
    return tickers


def _fallback_sp500() -> list[str]:
    """Hardcoded top ~100 S&P 500 stocks as fallback."""
    return sorted([
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AFL",
        "AIG", "AMAT", "AMD", "AMGN", "AMZN", "ANET", "AVGO", "AXP", "BA", "BAC",
        "BDX", "BK", "BKNG", "BLK", "BMY", "BSX", "C", "CAT", "CB", "CCI",
        "CDNS", "CEG", "CI", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO",
        "CTAS", "CVS", "CVX", "D", "DE", "DHR", "DIS", "DUK", "ECL", "EL",
        "EMR", "EOG", "EW", "EXC", "F", "FDX", "FI", "GD", "GE", "GILD",
        "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM", "ICE", "INTC", "INTU",
        "ISRG", "ITW", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
        "MCD", "MDLZ", "MDT", "MET", "META", "MMC", "MMM", "MO", "MRK", "MS",
        "MSFT", "NEE", "NFLX", "NKE", "NOC", "NOW", "NVDA", "ORCL", "PEP", "PFE",
        "PG", "PGR", "PM", "PNC", "PYPL", "QCOM", "REGN", "RTX", "SBUX", "SCHW",
        "SHW", "SLB", "SNPS", "SO", "SPG", "SYK", "T", "TGT", "TMO", "TMUS",
        "TRV", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA",
        "WFC", "WMT", "XOM", "ZTS",
    ])


def build_universe(universe_config: dict) -> list[str]:
    source = universe_config.get("source", "sp500")
    if source == "sp500":
        tickers = get_sp500_tickers()
    elif source == "custom":
        tickers = universe_config.get("tickers", [])
    else:
        tickers = get_sp500_tickers()

    exclude = set(universe_config.get("exclude_tickers", []))
    tickers = [t for t in tickers if t not in exclude]
    return tickers
