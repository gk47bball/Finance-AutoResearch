"""
Technical Analysis Backtest Engine
===================================
Takes a TA strategy configuration and runs a full signal-based backtest.

Three modes:
1. Alpha Test: Test each indicator individually for predictive power
2. Strategy Backtest: Combine signals and simulate a trading strategy
3. Scenario-Aware Backtest: Uses regime filtering, confluence gating,
   and signal strength filtering based on scenario analysis findings
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
import yfinance as yf

from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
from technical_analysis.indicators.core import sma
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


# ---------------------------------------------------------------------------
# Scenario-Aware Backtest Engine
# ---------------------------------------------------------------------------

def _classify_regime_series(df: pd.DataFrame) -> pd.Series:
    """Classify each day into a market regime for filtering."""
    c = df["Close"]
    ret_60d = c.pct_change(60)
    ret_20d = c.pct_change(20)
    daily_ret = c.pct_change()
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol_20d.rolling(252).median()
    ma200 = sma(c, 200)
    above_ma200 = c > ma200

    regime = pd.Series("sideways", index=df.index)
    regime[(above_ma200) & (ret_60d > 0.05)] = "bull"
    regime[(~above_ma200) & (ret_60d < -0.05)] = "bear"
    regime[(ret_60d.abs() < 0.05)] = "sideways"

    high_vol = vol_20d > vol_median * 1.3
    correction = (above_ma200) & (ret_20d < -0.05)
    recovery = (~above_ma200) & (ret_20d > 0.05)

    regime[high_vol & (regime == "bull")] = "bull_volatile"
    regime[high_vol & (regime == "bear")] = "bear_volatile"
    regime[correction] = "correction"
    regime[recovery] = "recovery"

    return regime


def _compute_confluence(df: pd.DataFrame, indicators_cfg: dict,
                        signal_rules: dict) -> pd.Series:
    """Count how many indicators are bullish (z-score > 0)."""
    lb = signal_rules.get("lookback_for_zscore", 63)
    n_bullish = pd.Series(0, index=df.index)
    n_total = pd.Series(0, index=df.index)

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
            if signal_rules.get("normalize", True):
                signal = _normalize_signal(signal, lb)

            valid = signal.notna()
            n_total += valid.astype(int)
            n_bullish += (signal > 0).astype(int) * valid.astype(int)
        except Exception:
            continue

    return n_bullish, n_total


def run_scenario_backtest(strategy_module, verbose: bool = True) -> TABacktestResult:
    """
    Scenario-aware backtest incorporating findings from scenario analysis:
    - Regime filtering: reduce exposure in sideways markets
    - Confluence gating: only trade at extreme confluence (0-2 or 7-9 bullish)
    - Signal strength filtering: require signal in extreme deciles
    - Minimum 15-day holding period (optimal horizon finding)
    """
    indicators_cfg = getattr(strategy_module, "INDICATORS", {})
    signal_rules = getattr(strategy_module, "SIGNAL_RULES", {})
    trading = getattr(strategy_module, "TRADING", {})
    eval_cfg = getattr(strategy_module, "EVALUATION", {})
    universe = getattr(strategy_module, "UNIVERSE", {})
    scenario_cfg = getattr(strategy_module, "SCENARIO_RULES", {})

    ticker = universe.get("tickers", ["SPY"])[0]
    period = universe.get("period", "10y")
    benchmark = eval_cfg.get("benchmark", "SPY")

    if verbose:
        print(f"  Fetching data: {ticker} ({period})...")

    df = fetch_data(ticker, period)
    if len(df) < 200:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} bars")

    # --- Compute all enabled indicator signals ---
    lb = signal_rules.get("lookback_for_zscore", 63)
    individual_signals = {}
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
            if signal_rules.get("normalize", True):
                signal = _normalize_signal(signal, lb)

            individual_signals[name] = signal
            weight = cfg["weight"]
            combined_signal += signal.fillna(0) * weight
            total_weight += weight
        except Exception as e:
            if verbose:
                print(f"  Warning: {name} failed — {e}")

    if total_weight > 0:
        combined_signal /= total_weight

    if signal_rules.get("flip_signal", False):
        combined_signal = -combined_signal

    # --- Scenario-aware signal modulation ---

    # 1. Regime classification
    regime = _classify_regime_series(df)
    regime_filter_enabled = scenario_cfg.get("regime_filter", False)
    blocked_regimes = scenario_cfg.get("blocked_regimes", [])

    # Regime-based signal amplification (not binary filter)
    regime_boost = scenario_cfg.get("regime_boost", {
        "bull_volatile": 1.5,   # Dips in uptrends: strongest contrarian edge
        "correction": 1.5,      # Sharp pullback in uptrend
        "bull": 1.0,            # Normal
        "bear_volatile": 0.8,   # Mixed signals in bear vol
        "recovery": 0.7,        # Contrarian dangerous in recovery
        "sideways": 0.6,        # IC near zero
        "bear": 0.8,            # Moderate
        "neutral": 1.2,         # Strong IC historically
    })

    regime_multiplier = regime.map(regime_boost).fillna(1.0).astype(float)
    if scenario_cfg.get("use_regime_boost", True):
        combined_signal = combined_signal * regime_multiplier

    # 2. Confluence-based signal amplification
    n_bullish, n_total = _compute_confluence(df, indicators_cfg, signal_rules)
    n_enabled = float(n_total.median())

    if scenario_cfg.get("use_confluence_boost", True) and n_enabled > 0:
        # Boost signal at confluence extremes (U-shaped edge)
        pct_bullish = n_bullish / n_total.replace(0, np.nan)
        # Oversold extreme (0-20% bullish) → amplify contrarian buy
        # Overbought extreme (80-100% bullish) → amplify momentum
        # Dead zone (30-70%) → dampen signal
        confluence_mult = pd.Series(1.0, index=df.index)
        confluence_mult[pct_bullish <= 0.25] = scenario_cfg.get("confluence_extreme_boost", 1.4)
        confluence_mult[pct_bullish >= 0.80] = scenario_cfg.get("confluence_momentum_boost", 1.2)
        confluence_mult[(pct_bullish > 0.35) & (pct_bullish < 0.65)] = scenario_cfg.get("confluence_dead_zone", 0.6)
        combined_signal = combined_signal * confluence_mult

    # 3. Regime-based position blocking (optional hard filter)
    regime_ok = pd.Series(True, index=df.index)
    if regime_filter_enabled and blocked_regimes:
        regime_ok = ~regime.isin(blocked_regimes)

    # --- Generate positions ---
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

    # Apply hard regime filter if enabled
    position = position * regime_ok.astype(float)

    # Apply minimum holding period
    min_hold = scenario_cfg.get("holding_period_min",
        trading.get("holding_period_min", 1))
    if min_hold > 1:
        smoothed = position.copy()
        i = 0
        while i < len(smoothed):
            if smoothed.iloc[i] != 0 and (i == 0 or smoothed.iloc[i-1] == 0):
                end = min(i + min_hold, len(smoothed))
                smoothed.iloc[i:end] = smoothed.iloc[i]
                i = end
            else:
                i += 1
        position = smoothed

    # --- Compute returns (same as standard backtest) ---
    daily_returns = df["Close"].pct_change()
    strategy_returns = position.shift(1) * daily_returns
    strategy_returns = strategy_returns.dropna()

    comm_bps = eval_cfg.get("commission_bps", 5) / 10000
    trades = position.diff().abs()
    strategy_returns -= trades.shift(1).fillna(0) * comm_bps

    if benchmark == ticker:
        bench_returns = daily_returns
    else:
        bench_df = fetch_data(benchmark, period)
        bench_returns = bench_df["Close"].pct_change()

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

    winning_days = (strategy_returns > 0).sum()
    total_days = (strategy_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    n_trades = (position.diff().abs() > 0).sum()

    in_position = (position != 0)
    changes = in_position.astype(int).diff().abs()
    n_entries = (changes == 1).sum() // 2 + 1
    avg_duration = in_position.sum() / max(n_entries, 1)

    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = strategy_returns[strategy_returns < 0].sum()
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

    exposure = in_position.mean()
    alpha = sr_mean - bench_sharpe * sr_std

    # Scenario-specific metrics
    regime_exposure = {}
    for r in regime.unique():
        mask = regime == r
        r_pos = position[mask]
        regime_exposure[r] = float((r_pos != 0).mean()) if len(r_pos) > 0 else 0

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
        "regime_exposure": regime_exposure,
        "scenario_filters": {
            "regime_filter": regime_filter_enabled,
            "regime_boost": scenario_cfg.get("use_regime_boost", True),
            "confluence_boost": scenario_cfg.get("use_confluence_boost", True),
            "min_hold": min_hold,
        },
    }

    if verbose:
        print(f"\n  Regime exposure:")
        for r, exp in sorted(regime_exposure.items(), key=lambda x: -x[1]):
            print(f"    {r:20s} {exp:.0%}")

    return TABacktestResult(**{k: metrics[k] for k in TABacktestResult.__dataclass_fields__
                               if k != "metrics"}, metrics=metrics)
