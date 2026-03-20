"""Macroeconomic data from FRED."""

import os
import pandas as pd
from data.cache import get_cache

FRED_SERIES = {
    "gdp_growth": "A191RL1Q225SBEA",       # Real GDP growth rate
    "cpi_yoy": "CPIAUCSL",                  # CPI (compute YoY)
    "unemployment": "UNRATE",                # Unemployment rate
    "fed_funds": "FEDFUNDS",                 # Federal Funds rate
    "treasury_10y": "DGS10",                 # 10-Year Treasury
    "treasury_2y": "DGS2",                   # 2-Year Treasury
    "treasury_3m": "DTB3",                   # 3-Month T-Bill
    "vix": "VIXCLS",                         # VIX
    "baa_spread": "BAA10Y",                  # BAA corporate - 10Y spread
}


def _get_fred_client():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
        return Fred(api_key=api_key)
    except ImportError:
        return None


def get_fred_series(series_id: str, start: str = None, end: str = None) -> pd.Series:
    cache = get_cache()
    key = f"fred:{series_id}:{start}:{end}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    fred = _get_fred_client()
    if fred is None:
        return pd.Series(dtype=float)

    try:
        data = fred.get_series(series_id, observation_start=start, observation_end=end)
        cache.put(key, data)
        return data
    except Exception:
        return pd.Series(dtype=float)


def get_macro_snapshot() -> dict:
    """Get current values for key macro indicators."""
    cache = get_cache()
    cached = cache.get("macro:snapshot", ttl_hours=6)
    if cached is not None:
        return cached

    snapshot = {}
    for name, series_id in FRED_SERIES.items():
        series = get_fred_series(series_id)
        if not series.empty:
            snapshot[name] = {
                "value": float(series.dropna().iloc[-1]),
                "date": str(series.dropna().index[-1].date()),
            }
        else:
            snapshot[name] = {"value": None, "date": None}

    # Compute yield curve spread
    t10 = snapshot.get("treasury_10y", {}).get("value")
    t2 = snapshot.get("treasury_2y", {}).get("value")
    if t10 is not None and t2 is not None:
        snapshot["yield_curve_spread"] = {
            "value": t10 - t2,
            "date": snapshot["treasury_10y"]["date"],
        }

    cache.put("macro:snapshot", snapshot)
    return snapshot


def get_risk_free_rate() -> float:
    """Get current 3-month T-bill rate as annualized decimal."""
    snapshot = get_macro_snapshot()
    rate = snapshot.get("treasury_3m", {}).get("value")
    if rate is not None:
        return rate / 100.0  # Convert from percent
    return 0.04  # Default fallback
