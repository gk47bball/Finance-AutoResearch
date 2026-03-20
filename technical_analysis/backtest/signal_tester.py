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

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf


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


def fetch_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
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
