"""
Technical Analysis Backtest Engine
===================================
Takes a TA strategy configuration and runs a full signal-based backtest.

Two modes:
1. Alpha Test: Test each indicator individually for predictive power
2. Strategy Backtest: Combine signals and simulate a trading strategy
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import yfinance as yf

from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
from technical_analysis.backtest.signal_tester import (
    fetch_data, test_signal, test_indicator_multi_ticker,
    SignalTestResult, MultiTickerResult,
)


@dataclass
class TABacktestResult:
    """Results from a TA strategy backtest."""
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float
    total_return: float
    win_rate: float
    n_trades: int
    avg_trade_duration: float
    profit_factor: float
    exposure_pct: float  # % of time in market
    benchmark_sharpe: float
    benchmark_return: float
    alpha: float
    metrics: dict  # full metrics dict


def _normalize_signal(signal: pd.Series, lookback: int = 63) -> pd.Series:
    """Normalize signal to rolling z-score."""
    mean = signal.rolling(lookback, min_periods=20).mean()
    std = signal.rolling(lookback, min_periods=20).std().replace(0, np.nan)
    return (signal - mean) / std


def run_alpha_test(strategy_module, verbose: bool = True) -> dict:
    """
    Test each enabled indicator for standalone predictive power.
    Returns dict of indicator_name → MultiTickerResult.
    """
    indicators = getattr(strategy_module, "INDICATORS", {})
    eval_cfg = getattr(strategy_module, "EVALUATION", {})
    test_tickers = eval_cfg.get("test_tickers", ["SPY"])
    horizons = eval_cfg.get("forward_horizons", [1, 2, 5, 10, 20])

    results = {}
    for name, cfg in indicators.items():
        if name not in INDICATOR_REGISTRY:
            if verbose:
                print(f"  [SKIP] {name}: not in indicator registry")
            continue

        reg = INDICATOR_REGISTRY[name]
        if verbose:
            print(f"\n  Testing: {name} — {reg['description']}")

        params = cfg.get("params", reg.get("params", {}))
        signal_col = cfg.get("signal_col", reg.get("signal_col"))

        result = test_indicator_multi_ticker(
            indicator_name=name,
            indicator_fn=reg["fn"],
            signal_col=signal_col,
            params=params,
            tickers=test_tickers,
            horizons=horizons,
        )
        results[name] = result

        if verbose:
            print(f"    Composite Score: {result.composite_score:.1f}/100")
            print(f"    Consistency: {result.consistency:.0%}")
            for h in [5, 10, 20]:
                ic = result.avg_ic.get(h, 0)
                spread = result.avg_spread.get(h, 0)
                print(f"    {h}d IC={ic:+.4f}  Spread={spread:+.1%}")

    return results


def run_strategy_backtest(strategy_module, verbose: bool = True) -> TABacktestResult:
    """
    Run a combined-signal backtest using the strategy configuration.
    """
    indicators_cfg = getattr(strategy_module, "INDICATORS", {})
    signal_rules = getattr(strategy_module, "SIGNAL_RULES", {})
    trading = getattr(strategy_module, "TRADING", {})
    eval_cfg = getattr(strategy_module, "EVALUATION", {})
    universe = getattr(strategy_module, "UNIVERSE", {})

    ticker = universe.get("tickers", ["SPY"])[0]
    period = universe.get("period", "10y")
    benchmark = eval_cfg.get("benchmark", "SPY")

    if verbose:
        print(f"  Fetching data: {ticker} ({period})...")

    df = fetch_data(ticker, period)
    if len(df) < 200:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} bars")

    # Compute all enabled indicators
    combined_signal = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for name, cfg in indicators_cfg.items():
        if not cfg.get("enabled", False) or cfg.get("weight", 0) == 0:
            continue
        if name not in INDICATOR_REGISTRY:
            continue

        reg = INDICATOR_REGISTRY[name]
        params = cfg.get("params", reg.get("params", {}))
        signal_col = cfg.get("signal_col", reg.get("signal_col"))

        try:
            ind_df = reg["fn"](df, **params)
            signal = ind_df[signal_col]

            # Normalize if configured
            if signal_rules.get("normalize", True):
                lb = signal_rules.get("lookback_for_zscore", 63)
                signal = _normalize_signal(signal, lb)

            weight = cfg["weight"]
            combined_signal += signal.fillna(0) * weight
            total_weight += weight
        except Exception as e:
            if verbose:
                print(f"  Warning: {name} failed — {e}")

    if total_weight > 0:
        combined_signal /= total_weight

    # Flip signal for contrarian/mean-reversion mode
    if signal_rules.get("flip_signal", False):
        combined_signal = -combined_signal

    # Generate positions from combined signal
    long_thresh = trading.get("long_threshold", 0.5)
    short_thresh = trading.get("short_threshold", -0.5)
    allow_short = trading.get("allow_short", False)
    sizing = trading.get("position_sizing", "binary")

    if sizing == "binary":
        position = pd.Series(0.0, index=df.index)
        position[combined_signal > long_thresh] = 1.0
        if allow_short:
            position[combined_signal < short_thresh] = -1.0
    elif sizing == "scaled":
        position = combined_signal.clip(-1, 1)
        if not allow_short:
            position = position.clip(lower=0)
    elif sizing == "always_in":
        position = np.where(combined_signal > 0, 1.0, -1.0 if allow_short else 0.0)
        position = pd.Series(position, index=df.index)

    # Apply minimum holding period
    min_hold = trading.get("holding_period_min", 1)
    if min_hold > 1:
        # Once a position is entered, hold for at least min_hold days
        smoothed = position.copy()
        i = 0
        while i < len(smoothed):
            if smoothed.iloc[i] != 0 and (i == 0 or smoothed.iloc[i-1] == 0):
                # New position — enforce hold
                end = min(i + min_hold, len(smoothed))
                smoothed.iloc[i:end] = smoothed.iloc[i]
                i = end
            else:
                i += 1
        position = smoothed

    # Compute returns
    daily_returns = df["Close"].pct_change()
    strategy_returns = position.shift(1) * daily_returns  # shift to avoid lookahead
    strategy_returns = strategy_returns.dropna()

    # Commission drag
    comm_bps = eval_cfg.get("commission_bps", 5) / 10000
    trades = position.diff().abs()
    strategy_returns -= trades.shift(1).fillna(0) * comm_bps

    # Benchmark
    if benchmark == ticker:
        bench_returns = daily_returns
    else:
        bench_df = fetch_data(benchmark, period)
        bench_returns = bench_df["Close"].pct_change()

    # Align
    common_idx = strategy_returns.index.intersection(bench_returns.index)
    strategy_returns = strategy_returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    # Metrics
    ann_factor = 252
    sr_mean = strategy_returns.mean() * ann_factor
    sr_std = strategy_returns.std() * np.sqrt(ann_factor)
    sharpe = sr_mean / sr_std if sr_std > 0 else 0

    br_mean = bench_returns.mean() * ann_factor
    br_std = bench_returns.std() * np.sqrt(ann_factor)
    bench_sharpe = br_mean / br_std if br_std > 0 else 0

    cum = (1 + strategy_returns).cumprod()
    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    running_max = cum.cummax()
    drawdown = (cum / running_max) - 1
    max_dd = drawdown.min()

    bench_cum = (1 + bench_returns).cumprod()
    bench_total = bench_cum.iloc[-1] - 1 if len(bench_cum) > 0 else 0

    # Win rate
    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Number of trades (position changes)
    n_trades = (position.diff().abs() > 0).sum()

    # Average trade duration
    in_position = (position != 0)
    changes = in_position.astype(int).diff().abs()
    n_entries = (changes == 1).sum() // 2 + 1
    avg_duration = in_position.sum() / max(n_entries, 1)

    # Profit factor
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = strategy_returns[strategy_returns < 0].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    # Exposure
    exposure = in_position.mean()

    # Alpha (simple)
    alpha = sr_mean - bench_sharpe * sr_std

    metrics = {
        "sharpe_ratio": sharpe,
        "annual_return": sr_mean,
        "annual_volatility": sr_std,
        "max_drawdown": max_dd,
        "total_return": total_ret,
        "win_rate": win_rate,
        "n_trades": int(n_trades),
        "avg_trade_duration": avg_duration,
        "profit_factor": profit_factor,
        "exposure_pct": exposure,
        "benchmark_sharpe": bench_sharpe,
        "benchmark_return": br_mean,
        "alpha": alpha,
    }

    return TABacktestResult(**metrics, metrics=metrics)
