"""Price data via yfinance with caching."""

import yfinance as yf
import pandas as pd
from data.cache import get_cache


def get_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    cache = get_cache()
    key = f"prices:{ticker}:{start}:{end}"
    cached = cache.get(key, ttl_hours=1)
    if cached is not None:
        return cached

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return df

    # Flatten MultiIndex columns if present (yfinance returns MultiIndex for single ticker too)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cache.put(key, df)
    return df


def get_returns(ticker: str, start: str, end: str) -> pd.Series:
    prices = get_prices(ticker, start, end)
    if prices.empty:
        return pd.Series(dtype=float)
    return prices["Close"].pct_change().dropna()


def get_bulk_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Fetch prices for many tickers. Uses individual calls with caching."""
    results = {}
    for ticker in tickers:
        df = get_prices(ticker, start, end)
        if not df.empty:
            results[ticker] = df
    return results


def get_benchmark_returns(benchmark: str, start: str, end: str) -> pd.Series:
    return get_returns(benchmark, start, end)
