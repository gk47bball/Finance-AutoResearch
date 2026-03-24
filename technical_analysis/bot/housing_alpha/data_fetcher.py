"""
Housing Data Fetcher
=====================
Pulls housing-related data from FRED (API) and Zillow Research (CSV downloads).
Caches locally to avoid re-fetching slow monthly data.

FRED series sourced from:
  https://fred.stlouisfed.org/categories/97  (Housing)
  https://fred.stlouisfed.org/categories/46  (Interest Rates)

Zillow data from:
  https://www.zillow.com/research/data/
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

CACHE_DIR = Path(__file__).parent / "state" / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# FRED Housing Series
# ---------------------------------------------------------------------------

FRED_HOUSING_SERIES = {
    # Leading indicators (permits → starts → completions → sales)
    "housing_permits":          "PERMIT",       # New private housing units authorized (monthly, ~3 month lead)
    "housing_starts":           "HOUST",        # Housing starts (monthly, ~2 month lead)
    "housing_completions":      "COMPUTSA",     # Housing completions (monthly)
    "new_home_sales":           "HSN1F",        # New residential sales (monthly, seasonally adj)
    "existing_home_sales":      "EXHOSLUSM495S",# Existing home sales (monthly)

    # Inventory and supply
    "months_supply":            "MSACSR",       # Monthly supply of new houses (months)
    "housing_inventory":        "ACTLISCOUUS",  # Active listing count (monthly, Realtor.com via FRED)

    # Prices
    "case_shiller_national":    "CSUSHPINSA",   # S&P/Case-Shiller US National Home Price Index
    "median_home_price":        "MSPUS",        # Median sales price of houses sold (quarterly)
    "fhfa_hpi":                 "USSTHPI",      # FHFA House Price Index (quarterly)

    # Mortgage rates and credit
    "mortgage_30y":             "MORTGAGE30US",  # 30-Year fixed mortgage rate (weekly)
    "mortgage_15y":             "MORTGAGE15US",  # 15-Year fixed mortgage rate (weekly)
    # "mortgage_apps" removed — MBASSMWO not publicly available via FRED API

    # Construction and materials
    "construction_spending":    "TLRESCONS",     # Total residential construction spending (monthly)
    "lumber_ppi":               "WPU0811",       # PPI: Lumber (monthly)

    # Affordability
    "median_hh_income":         "MEHOINUSA672N", # Median household income (annual)
    "cpi_shelter":              "CUSR0000SAH1",  # CPI: Shelter component (monthly)
    "homeownership_rate":       "RHORUSQ156N",   # Homeownership rate (quarterly)

    # Macro rates (critical for housing)
    "fed_funds":                "FEDFUNDS",      # Federal funds rate
    "treasury_10y":             "DGS10",         # 10-Year Treasury yield
    "treasury_2y":              "DGS2",          # 2-Year Treasury yield
}

# Core series that must be available for the engine to work
CORE_SERIES = [
    "housing_starts", "housing_permits", "new_home_sales",
    "existing_home_sales", "mortgage_30y", "case_shiller_national",
    "months_supply", "treasury_10y",
]


def _fred_api_key() -> Optional[str]:
    return os.environ.get("FRED_API_KEY")


def fetch_fred_series(
    series_id: str,
    start: str = "2000-01-01",
    end: Optional[str] = None,
) -> pd.Series:
    """Fetch a single FRED series. Returns a datetime-indexed pd.Series."""
    api_key = _fred_api_key()
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set in .env. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    cache_file = CACHE_DIR / f"fred_{series_id}.parquet"

    # Use cache if fresh (less than 12 hours old)
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            df = pd.read_parquet(cache_file)
            return df["value"]

    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # Fall back to cache if API fails
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df["value"]
        raise RuntimeError(f"FRED API failed for {series_id}: {e}")

    observations = data.get("observations", [])
    if not observations:
        return pd.Series(dtype=float, name=series_id)

    records = []
    for obs in observations:
        val = obs.get("value", ".")
        if val == ".":
            continue
        try:
            records.append({
                "date": pd.Timestamp(obs["date"]),
                "value": float(val),
            })
        except (ValueError, KeyError):
            continue

    if not records:
        return pd.Series(dtype=float, name=series_id)

    df = pd.DataFrame(records).set_index("date")
    df.index.name = "date"

    # Cache to disk
    try:
        df.to_parquet(cache_file)
    except Exception:
        pass

    return df["value"].rename(series_id)


def fetch_all_fred_housing(start: str = "2000-01-01") -> dict[str, pd.Series]:
    """Fetch all FRED housing series. Returns dict of name → pd.Series."""
    results = {}
    for name, series_id in FRED_HOUSING_SERIES.items():
        try:
            s = fetch_fred_series(series_id, start=start)
            if not s.empty:
                results[name] = s
        except Exception as e:
            print(f"  [housing] FRED {name} ({series_id}) failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Zillow Research Data (CSV downloads)
# ---------------------------------------------------------------------------

ZILLOW_URLS = {
    # Zillow Home Value Index — national, monthly
    "zhvi_national": "https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv",
    # Zillow Observed Rent Index — national, monthly
    "zori_national": "https://files.zillowstatic.com/research/public_csvs/zori/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv",
    # Inventory — for sale count
    "inventory": "https://files.zillowstatic.com/research/public_csvs/invt_fs/Metro_invt_fs_uc_sfrcondo_sm_month.csv",
    # New listings count
    "new_listings": "https://files.zillowstatic.com/research/public_csvs/new_listings/Metro_new_listings_uc_sfrcondo_sm_month.csv",
    # Days to pending — URL changed, may not be available
    # "days_to_pending": removed — Zillow deprecated this CSV endpoint
}


def fetch_zillow_series(name: str) -> Optional[pd.Series]:
    """
    Fetch a Zillow Research CSV and extract the national (United States) row.
    Returns a datetime-indexed pd.Series of monthly values.
    """
    url = ZILLOW_URLS.get(name)
    if not url:
        return None

    cache_file = CACHE_DIR / f"zillow_{name}.parquet"

    # Use cache if fresh (less than 24 hours)
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            df = pd.read_parquet(cache_file)
            return df["value"]

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
            return df["value"]
        print(f"  [housing] Zillow {name} download failed: {e}")
        return None

    try:
        from io import StringIO
        df_raw = pd.read_csv(StringIO(resp.text))
    except Exception as e:
        print(f"  [housing] Zillow {name} parse failed: {e}")
        return None

    # Find the US national row
    us_row = None
    for _, row in df_raw.iterrows():
        region = str(row.get("RegionName", ""))
        if region.lower() in ("united states", "us"):
            us_row = row
            break

    if us_row is None:
        print(f"  [housing] Zillow {name}: no 'United States' row found")
        return None

    # Date columns are like "2020-01-31"
    date_cols = [c for c in df_raw.columns if c[:4].isdigit()]
    records = []
    for col in date_cols:
        val = us_row.get(col)
        if pd.notna(val):
            try:
                records.append({"date": pd.Timestamp(col), "value": float(val)})
            except (ValueError, TypeError):
                continue

    if not records:
        return None

    result_df = pd.DataFrame(records).set_index("date").sort_index()
    result_df.index.name = "date"

    try:
        result_df.to_parquet(cache_file)
    except Exception:
        pass

    return result_df["value"].rename(name)


def fetch_all_zillow() -> dict[str, pd.Series]:
    """Fetch all Zillow series. Returns dict of name → pd.Series."""
    results = {}
    for name in ZILLOW_URLS:
        try:
            s = fetch_zillow_series(name)
            if s is not None and not s.empty:
                results[name] = s
        except Exception as e:
            print(f"  [housing] Zillow {name} failed: {e}")
    return results


# ---------------------------------------------------------------------------
# Unified data builder
# ---------------------------------------------------------------------------

def build_housing_dataset(start: str = "2000-01-01") -> pd.DataFrame:
    """
    Build a unified monthly housing dataset from FRED + Zillow.
    Returns a DataFrame with datetime index and one column per indicator.
    All series resampled to monthly frequency (end-of-month).
    """
    all_series = {}

    # FRED
    fred_data = fetch_all_fred_housing(start=start)
    all_series.update(fred_data)

    # Zillow
    zillow_data = fetch_all_zillow()
    all_series.update(zillow_data)

    if not all_series:
        raise RuntimeError("No housing data available. Check FRED_API_KEY.")

    # Resample everything to monthly (end-of-month) and forward-fill
    monthly_dfs = []
    for name, series in all_series.items():
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        # Resample to monthly — take last observation in each month
        s_monthly = s.resample("ME").last()
        s_monthly.name = name
        monthly_dfs.append(s_monthly)

    # Combine into single DataFrame
    df = pd.concat(monthly_dfs, axis=1).sort_index()

    # Forward-fill (housing data has staggered release dates)
    df = df.ffill()

    print(f"  [housing] Dataset: {len(df)} months, {len(df.columns)} indicators, "
          f"{df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

    return df


def get_housing_snapshot() -> dict:
    """Get current values and month-over-month changes for key indicators."""
    try:
        df = build_housing_dataset(start="2023-01-01")
    except Exception as e:
        return {"error": str(e)}

    snapshot = {}
    for col in df.columns:
        series = df[col].dropna()
        if len(series) < 2:
            continue
        current = float(series.iloc[-1])
        prev = float(series.iloc[-2])
        mom_pct = (current / prev - 1) * 100 if prev != 0 else 0
        yoy_pct = None
        if len(series) >= 13:
            year_ago = float(series.iloc[-13])
            yoy_pct = (current / year_ago - 1) * 100 if year_ago != 0 else None

        snapshot[col] = {
            "value": round(current, 2),
            "date": str(series.index[-1].date()),
            "mom_pct": round(mom_pct, 2),
            "yoy_pct": round(yoy_pct, 2) if yoy_pct is not None else None,
        }

    return snapshot
