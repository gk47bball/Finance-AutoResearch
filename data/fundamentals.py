"""Financial statements and ratios via yfinance."""

import yfinance as yf
import pandas as pd
import numpy as np
from data.cache import get_cache


def _get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


def get_info(ticker: str) -> dict:
    cache = get_cache()
    key = f"info:{ticker}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    try:
        info = _get_ticker(ticker).info
        if not info or "symbol" not in info:
            return {}
        cache.put(key, info)
        return info
    except Exception:
        return {}


def get_income_statement(ticker: str, annual: bool = True) -> pd.DataFrame:
    cache = get_cache()
    key = f"income:{ticker}:{'annual' if annual else 'quarterly'}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    t = _get_ticker(ticker)
    df = t.financials if annual else t.quarterly_financials
    if df is not None and not df.empty:
        cache.put(key, df)
    return df if df is not None else pd.DataFrame()


def get_balance_sheet(ticker: str, annual: bool = True) -> pd.DataFrame:
    cache = get_cache()
    key = f"balance:{ticker}:{'annual' if annual else 'quarterly'}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    t = _get_ticker(ticker)
    df = t.balance_sheet if annual else t.quarterly_balance_sheet
    if df is not None and not df.empty:
        cache.put(key, df)
    return df if df is not None else pd.DataFrame()


def get_cash_flow(ticker: str, annual: bool = True) -> pd.DataFrame:
    cache = get_cache()
    key = f"cashflow:{ticker}:{'annual' if annual else 'quarterly'}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    t = _get_ticker(ticker)
    df = t.cashflow if annual else t.quarterly_cashflow
    if df is not None and not df.empty:
        cache.put(key, df)
    return df if df is not None else pd.DataFrame()


def _safe_get(d: dict, key: str, default=None):
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return default
    return v


def get_key_ratios(ticker: str) -> dict:
    """Compute key financial ratios from yfinance data."""
    cache = get_cache()
    key = f"ratios:{ticker}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    info = get_info(ticker)
    if not info:
        return {}

    ratios = {
        "ticker": ticker,
        "sector": _safe_get(info, "sector", "Unknown"),
        "industry": _safe_get(info, "industry", "Unknown"),
        "market_cap": _safe_get(info, "marketCap", 0),
        "pe_ratio": _safe_get(info, "trailingPE"),
        "forward_pe": _safe_get(info, "forwardPE"),
        "pb_ratio": _safe_get(info, "priceToBook"),
        "ps_ratio": _safe_get(info, "priceToSalesTrailing12Months"),
        "ev_to_ebitda": _safe_get(info, "enterpriseToEbitda"),
        "ev_to_revenue": _safe_get(info, "enterpriseToRevenue"),
        "profit_margin": _safe_get(info, "profitMargins"),
        "operating_margin": _safe_get(info, "operatingMargins"),
        "gross_margin": _safe_get(info, "grossMargins"),
        "roe": _safe_get(info, "returnOnEquity"),
        "roa": _safe_get(info, "returnOnAssets"),
        "debt_to_equity": _safe_get(info, "debtToEquity"),
        "current_ratio": _safe_get(info, "currentRatio"),
        "quick_ratio": _safe_get(info, "quickRatio"),
        "revenue_growth": _safe_get(info, "revenueGrowth"),
        "earnings_growth": _safe_get(info, "earningsGrowth"),
        "dividend_yield": _safe_get(info, "dividendYield"),
        "payout_ratio": _safe_get(info, "payoutRatio"),
        "beta": _safe_get(info, "beta"),
        "avg_volume": _safe_get(info, "averageVolume", 0),
        "avg_volume_10d": _safe_get(info, "averageDailyVolume10Day", 0),
        "shares_outstanding": _safe_get(info, "sharesOutstanding", 0),
        "free_cash_flow": _safe_get(info, "freeCashflow"),
        "total_revenue": _safe_get(info, "totalRevenue"),
        "ebitda": _safe_get(info, "ebitda"),
    }

    # Derived ratios
    if ratios["pe_ratio"] and ratios["pe_ratio"] > 0:
        ratios["earnings_yield"] = 1.0 / ratios["pe_ratio"]
    else:
        ratios["earnings_yield"] = None

    if ratios["free_cash_flow"] and ratios["market_cap"] and ratios["market_cap"] > 0:
        ratios["fcf_yield"] = ratios["free_cash_flow"] / ratios["market_cap"]
    else:
        ratios["fcf_yield"] = None

    if ratios["ev_to_ebitda"] and ratios["ev_to_ebitda"] > 0:
        ratios["ev_to_ebitda_inv"] = 1.0 / ratios["ev_to_ebitda"]
    else:
        ratios["ev_to_ebitda_inv"] = None

    if ratios["debt_to_equity"] is not None:
        # Scale from percentage (yfinance returns D/E as percentage like 150 for 1.5x)
        if ratios["debt_to_equity"] > 10:
            ratios["debt_to_equity"] = ratios["debt_to_equity"] / 100.0
        ratios["debt_to_equity_inv"] = 1.0 / (1.0 + ratios["debt_to_equity"])
    else:
        ratios["debt_to_equity_inv"] = None

    cache.put(key, ratios)
    return ratios


def get_historical_financials(ticker: str) -> dict:
    """Get annual financial statements for point-in-time backtesting."""
    cache = get_cache()
    key = f"hist_financials:{ticker}"
    cached = cache.get(key, ttl_hours=24)
    if cached is not None:
        return cached

    income = get_income_statement(ticker, annual=True)
    balance = get_balance_sheet(ticker, annual=True)
    cashflow = get_cash_flow(ticker, annual=True)

    result = {
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }
    cache.put(key, result)
    return result
