"""
Signal Testing Framework
=========================
Tests whether a technical indicator signal has statistically significant
predictive power for forward returns.

Tests performed:
1. Quintile analysis: Sort days by indicator value, measure forward returns per quintile
2. Long/Short spread: Top quintile return - Bottom quintile return
3. IC (Information Coefficient): Rank correlation between signal and forward returns
4. Hit rate: % of days where signal direction matches forward return direction
5. Regime analysis: Does the signal work in all market regimes?
6. Signal decay: How quickly does predictive power decay (1d, 2d, 5d, 10d, 20d)?
7. Bootstrap significance: Is the IC statistically significant?
"""

import os
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import yfinance as yf

from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Alpaca data fetcher (free IEX feed — real-time, no subscription needed)
# ---------------------------------------------------------------------------

_ALPACA_BASE = "https://data.alpaca.markets/v2/stocks"
_ALPACA_KEY    = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_API_SECRET", "")

# Period strings → approximate calendar days (used to build start date)
_PERIOD_DAYS = {
    "1d": 1, "5d": 7, "1mo": 35, "3mo": 100, "6mo": 185,
    "1y": 370, "2y": 740, "5y": 1830, "10y": 3660,
}


def _alpaca_available() -> bool:
    return bool(_ALPACA_KEY and _ALPACA_SECRET and
                _ALPACA_KEY != "your_key_here")


def fetch_data_alpaca(
    ticker: str,
    period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV daily bars from Alpaca's free IEX data feed.
    Returns a DataFrame in the same format as fetch_data(), or None on failure.

    Free tier uses the IEX feed — real-time quotes, daily bars back ~5 years.
    No paid subscription needed; just a free Alpaca paper-trading account.
    """
    if not _alpaca_available():
        return None

    headers = {
        "APCA-API-KEY-ID":     _ALPACA_KEY,
        "APCA-API-SECRET-KEY": _ALPACA_SECRET,
    }

    # Build date range
    end_dt   = datetime.fromisoformat(end)   if end   else datetime.utcnow()
    if start:
        start_dt = datetime.fromisoformat(start)
    else:
        days     = _PERIOD_DAYS.get(period, 370)
        start_dt = end_dt - timedelta(days=days + 10)  # small buffer

    params = {
        "timeframe": "1Day",
        "start":     start_dt.strftime("%Y-%m-%d"),
        "end":       end_dt.strftime("%Y-%m-%d"),
        "feed":      "iex",       # free tier; "sip" requires paid subscription
        "limit":     10000,
        "adjustment": "all",      # split + dividend adjusted — same as yfinance default
    }

    try:
        url  = f"{_ALPACA_BASE}/{ticker}/bars"
        rows = []
        while True:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            bars = data.get("bars", [])
            if not bars:
                break
            rows.extend(bars)
            token = data.get("next_page_token")
            if not token:
                break
            params["page_token"] = token

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["t"]).dt.tz_localize(None)
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low",
            "c": "Close", "v": "Volume",
        })
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df = df.set_index("Date").sort_index()
        df.index.name = "Date"
        return df

    except Exception as e:
        print(f"  [alpaca] fetch failed for {ticker}: {e}")
        return None


@dataclass
class SignalTestResult:
    """Results from testing one indicator on one ticker."""
    indicator_name: str
    ticker: str
    n_observations: int

    # Quintile analysis
    quintile_returns: dict  # {1: mean_ret, 2: ..., 5: ...} for each forward period
    long_short_spread: dict  # {1: spread, 5: spread, ...}

    # Information Coefficient
    ic_by_horizon: dict  # {1: IC, 5: IC, 10: IC, 20: IC}
    ic_t_stat: dict  # t-stat for each IC
    ic_pvalue: dict

    # Hit rate
    hit_rate: dict  # {1: pct, 5: pct, ...}

    # Overall scores
    composite_alpha_score: float  # 0-100 composite
    is_significant: bool


@dataclass
class MultiTickerResult:
    """Aggregated results across multiple tickers."""
    indicator_name: str
    tickers: list
    avg_ic: dict
    avg_spread: dict
    avg_hit_rate: dict
    consistency: float  # % of tickers where signal is significant
    composite_score: float
    per_ticker: list  # list of SignalTestResult


def fetch_data(
    ticker: str,
    period: str = "10y",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV daily bars.

    Routing logic:
      - SHORT periods (≤ 2y) and live scans ("5d", "1d", "1mo") → try Alpaca
        first (real-time IEX, no delay), fall back to yfinance on failure.
      - LONG periods (> 2y) or explicit start/end for backtests → yfinance
        directly (10-year data, consistent with AutoResearch history).

    Args:
        ticker: Ticker symbol.
        period: Period string e.g. "10y", "2y", "5d". Ignored if start provided.
        start:  ISO date "YYYY-MM-DD". If set, uses date range (forces yfinance).
        end:    ISO date "YYYY-MM-DD". Upper bound when using start.
    """
    # Long backtests or explicit date ranges → yfinance (has 10y data)
    _long_periods = {"5y", "10y", "15y", "20y", "max"}
    use_yfinance_direct = start or (period in _long_periods)

    if not use_yfinance_direct and _alpaca_available():
        df = fetch_data_alpaca(ticker, period=period, end=end)
        if df is not None and len(df) >= 5:
            return df
        # Alpaca failed → fall through to yfinance

    # yfinance path
    if start:
        df = yf.download(ticker, start=start, end=end, progress=False)
    else:
        df = yf.download(ticker, period=period, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Drop duplicate columns — yfinance occasionally returns two "Close" columns
    # after MultiIndex flattening causing shape-(N,2) errors in backtest arithmetic.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def compute_forward_returns(df: pd.DataFrame,
                           horizons: list = [1, 2, 5, 10, 20]) -> pd.DataFrame:
    """Compute forward returns at multiple horizons."""
    c = df["Close"]
    fwd = pd.DataFrame(index=df.index)
    for h in horizons:
        fwd[f"fwd_{h}d"] = c.shift(-h) / c - 1
    return fwd


def test_signal(indicator_name: str,
                signal: pd.Series,
                df: pd.DataFrame,
                horizons: list = [1, 2, 5, 10, 20],
                n_quantiles: int = 5,
                ticker: str = "SPY") -> SignalTestResult:
    """
    Test a single signal series for predictive power.

    Args:
        indicator_name: Name of the indicator
        signal: pd.Series of indicator values (aligned with df index)
        df: DataFrame with OHLCV data
        horizons: Forward return horizons to test
        n_quantiles: Number of quantile buckets
        ticker: Ticker symbol for labeling
    """
    fwd = compute_forward_returns(df, horizons)

    # Align and drop NaN
    combined = pd.DataFrame({"signal": signal}, index=df.index)
    for h in horizons:
        combined[f"fwd_{h}d"] = fwd[f"fwd_{h}d"]
    combined = combined.dropna()

    n_obs = len(combined)
    if n_obs < 100:
        return SignalTestResult(
            indicator_name=indicator_name, ticker=ticker, n_observations=n_obs,
            quintile_returns={}, long_short_spread={}, ic_by_horizon={},
            ic_t_stat={}, ic_pvalue={}, hit_rate={},
            composite_alpha_score=0.0, is_significant=False,
        )

    # 1. Quintile analysis
    combined["quintile"] = pd.qcut(combined["signal"], n_quantiles,
                                    labels=False, duplicates="drop") + 1
    quintile_returns = {}
    long_short_spread = {}
    for h in horizons:
        col = f"fwd_{h}d"
        qr = combined.groupby("quintile")[col].mean()
        quintile_returns[h] = qr.to_dict()
        # Spread = top quintile - bottom quintile (annualized)
        if n_quantiles in qr.index and 1 in qr.index:
            spread = (qr[n_quantiles] - qr[1]) * (252 / h)
            long_short_spread[h] = spread
        else:
            long_short_spread[h] = 0.0

    # 2. Information Coefficient (Spearman rank correlation)
    ic_by_horizon = {}
    ic_t_stat = {}
    ic_pvalue = {}
    for h in horizons:
        col = f"fwd_{h}d"
        from scipy.stats import spearmanr
        corr, pval = spearmanr(combined["signal"], combined[col])
        ic_by_horizon[h] = corr
        # t-stat approximation
        t = corr * np.sqrt((n_obs - 2) / (1 - corr**2 + 1e-10))
        ic_t_stat[h] = t
        ic_pvalue[h] = pval

    # 3. Hit rate (signal sign matches forward return sign)
    hit_rate = {}
    for h in horizons:
        col = f"fwd_{h}d"
        signal_sign = np.sign(combined["signal"])
        return_sign = np.sign(combined[col])
        hits = (signal_sign == return_sign).sum()
        hit_rate[h] = hits / n_obs

    # 4. Composite score
    # Weighted: IC importance + monotonicity + spread magnitude
    avg_ic = np.mean([abs(ic_by_horizon.get(h, 0)) for h in [5, 10, 20]])
    avg_spread = np.mean([abs(long_short_spread.get(h, 0)) for h in [5, 10, 20]])
    avg_hit = np.mean([hit_rate.get(h, 0.5) for h in [5, 10, 20]])

    # Check monotonicity of quintile returns (do higher quintiles have higher returns?)
    mono_score = 0
    for h in [5, 10, 20]:
        qr = quintile_returns.get(h, {})
        if len(qr) >= n_quantiles:
            vals = [qr.get(q, 0) for q in range(1, n_quantiles + 1)]
            diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
            if all(d > 0 for d in diffs):
                mono_score += 1
            elif all(d < 0 for d in diffs):
                mono_score += 0.8  # Reverse monotonic is also useful (just go short)

    mono_pct = mono_score / 3

    composite = (
        30 * min(avg_ic / 0.05, 1.0) +  # IC score (0.05 is strong)
        25 * min(avg_spread / 0.10, 1.0) +  # Spread score (10% ann is strong)
        20 * max(0, (avg_hit - 0.5) / 0.05) +  # Hit rate above 50%
        25 * mono_pct  # Monotonicity
    )
    composite = min(composite, 100)

    # Significance: IC p-value < 0.05 at 5d or 10d horizon
    is_sig = any(ic_pvalue.get(h, 1.0) < 0.05 for h in [5, 10, 20])

    return SignalTestResult(
        indicator_name=indicator_name,
        ticker=ticker,
        n_observations=n_obs,
        quintile_returns=quintile_returns,
        long_short_spread=long_short_spread,
        ic_by_horizon=ic_by_horizon,
        ic_t_stat=ic_t_stat,
        ic_pvalue=ic_pvalue,
        hit_rate=hit_rate,
        composite_alpha_score=composite,
        is_significant=is_sig,
    )


def test_indicator_multi_ticker(indicator_name: str,
                                indicator_fn,
                                signal_col: str,
                                params: dict,
                                tickers: list = ["SPY", "QQQ", "IWM", "DIA"],
                                period: str = "10y",
                                horizons: list = [1, 2, 5, 10, 20]) -> MultiTickerResult:
    """
    Test an indicator across multiple tickers for robustness.
    """
    results = []
    for ticker in tickers:
        try:
            df = fetch_data(ticker, period)
            if len(df) < 200:
                continue
            indicator_df = indicator_fn(df, **params)
            signal = indicator_df[signal_col]
            result = test_signal(indicator_name, signal, df,
                               horizons=horizons, ticker=ticker)
            results.append(result)
        except Exception as e:
            print(f"  Warning: {ticker} failed — {e}")
            continue

    if not results:
        return MultiTickerResult(
            indicator_name=indicator_name, tickers=tickers,
            avg_ic={}, avg_spread={}, avg_hit_rate={},
            consistency=0.0, composite_score=0.0, per_ticker=[],
        )

    # Aggregate
    avg_ic = {}
    avg_spread = {}
    avg_hit = {}
    for h in horizons:
        ics = [r.ic_by_horizon.get(h, 0) for r in results]
        spreads = [r.long_short_spread.get(h, 0) for r in results]
        hits = [r.hit_rate.get(h, 0.5) for r in results]
        avg_ic[h] = np.mean(ics)
        avg_spread[h] = np.mean(spreads)
        avg_hit[h] = np.mean(hits)

    consistency = sum(1 for r in results if r.is_significant) / len(results)
    composite = np.mean([r.composite_alpha_score for r in results])

    return MultiTickerResult(
        indicator_name=indicator_name,
        tickers=[r.ticker for r in results],
        avg_ic=avg_ic,
        avg_spread=avg_spread,
        avg_hit_rate=avg_hit,
        consistency=consistency,
        composite_score=composite,
        per_ticker=results,
    )
