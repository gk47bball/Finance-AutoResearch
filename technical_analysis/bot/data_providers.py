"""
Data Providers
==============
Unified data access layer for the JK trading bot.

EODHD (basic subscription — EOD daily only):
  - get_eod(ticker, start, end) — daily OHLCV, all world exchanges
  - Automatically retries yfinance on any EODHD failure

yfinance (free, intraday-capable):
  - get_intraday(ticker, start, end, interval) — 5m/1h bars
  - get_eod_yf(ticker, period) — daily bars for longer lookbacks

API budget: EODHD basic = 100,000 requests/day. We track usage in memory.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

EODHD_API_KEY = os.environ.get("EODHD_API_KEY", "")
EODHD_BASE = "https://eodhd.com/api"

# Simple in-memory usage counter (resets each process start)
_eodhd_calls_today = 0


# ---------------------------------------------------------------------------
# EODHD — End-of-Day Data
# ---------------------------------------------------------------------------

def get_eod_eodhd(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from EODHD for a US ticker.
    ticker: bare symbol, e.g. "SPY" (we append .US automatically)
    start/end: "YYYY-MM-DD" strings
    Returns: DataFrame with DatetimeIndex and columns [open, high, low, close, volume]
    """
    global _eodhd_calls_today
    if not EODHD_API_KEY:
        raise RuntimeError("EODHD_API_KEY not set in .env")

    symbol = f"{ticker}.US" if "." not in ticker else ticker
    url = f"{EODHD_BASE}/eod/{symbol}"
    params = {
        "api_token": EODHD_API_KEY,
        "fmt": "json",
        "from": start,
        "to": end,
        "period": "d",
        "order": "a",
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    _eodhd_calls_today += 1

    data = resp.json()
    if not data:
        raise ValueError(f"EODHD returned empty data for {ticker} ({start} - {end})")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.rename(columns={"adjusted_close": "adj_close"})
    # Keep standard columns
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols].astype(float)

    # Use adjusted close if available
    if "adj_close" in pd.DataFrame(data).columns:
        adj_df = pd.DataFrame(data)
        adj_df["date"] = pd.to_datetime(adj_df["date"])
        adj_df = adj_df.set_index("date")
        df["close"] = adj_df["adjusted_close"].astype(float)

    return df


def get_eod_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: fetch daily data via yfinance."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end,
                     interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].astype(float)


def get_eod(ticker: str, start: str = None, end: str = None,
            period_days: int = 252) -> pd.DataFrame:
    """
    Get daily EOD data. Tries EODHD first, falls back to yfinance.
    If start/end not given, uses period_days ending today.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.today() - timedelta(days=period_days * 1.5)).strftime("%Y-%m-%d")

    if EODHD_API_KEY:
        try:
            df = get_eod_eodhd(ticker, start, end)
            if len(df) >= 5:
                return df
        except Exception as e:
            print(f"  EODHD EOD failed for {ticker}: {e} — falling back to yfinance")

    return get_eod_yfinance(ticker, start, end)


# ---------------------------------------------------------------------------
# yfinance — Intraday Data
# (EODHD intraday requires a paid upgrade; yfinance is used for all intraday)
# ---------------------------------------------------------------------------

def get_intraday(ticker: str, start: str, end: str,
                 interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday OHLCV via yfinance.
    interval: "1m" (7d max), "5m"/"15m"/"30m" (60d max), "1h" (730d max)
    Returns: DataFrame with tz-aware index in America/New_York
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end,
                     interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # Normalize timezone to ET
    if df.index.tz is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    return df[cols].astype(float)


def get_intraday_with_warmup(ticker: str, target_date: str,
                              interval: str = "5m",
                              warmup_trading_days: int = 15) -> pd.DataFrame:
    """
    Fetch intraday bars for target_date plus warmup_trading_days before it.
    The warmup ensures RSI, EMA, and other indicators are fully initialized
    when we reach the target day — solving the "no trades" problem with
    single-day data fetches.

    Returns: Full DataFrame (warmup + target day) with ET timezone index.
    """
    target = pd.Timestamp(target_date)
    # Fetch ~3x trading days worth of calendar days to cover weekends/holidays
    cal_days_back = int(warmup_trading_days * 1.6)
    start = (target - timedelta(days=cal_days_back)).strftime("%Y-%m-%d")
    end = (target + timedelta(days=1)).strftime("%Y-%m-%d")

    return get_intraday(ticker, start=start, end=end, interval=interval)


def eodhd_usage() -> dict:
    """Return current session EODHD call count."""
    return {"calls_this_session": _eodhd_calls_today, "daily_limit": 100_000}
