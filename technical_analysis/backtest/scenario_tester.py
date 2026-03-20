"""
Scenario Testing Framework
============================
Tests technical indicators across multiple dimensions to find WHERE they work:

1. STOCK TYPE: Large-cap vs mid-cap vs small-cap, growth vs value, high-vol vs low-vol
2. MARKET REGIME: Bull/bear/sideways, high-vol/low-vol, rising/falling rates
3. TIME HORIZON: Optimal holding period for each indicator (1d to 60d)
4. SIGNAL STRENGTH: Does the indicator work better at extremes?
5. SECTOR: Which sectors respond best to each indicator?
6. CONFLUENCE: Do indicators work better when multiple agree?

Returns a detailed report showing exactly when to use each indicator.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import yfinance as yf

from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
from technical_analysis.indicators.core import ema, sma, rsi, true_range
from technical_analysis.backtest.signal_tester import fetch_data, compute_forward_returns


# ---------------------------------------------------------------------------
# Stock universes for testing
# ---------------------------------------------------------------------------
STOCK_UNIVERSES = {
    "mega_cap": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "LLY", "JPM", "V"],
    "large_cap": ["HD", "PG", "JNJ", "UNH", "MA", "ABBV", "KO", "PEP", "MRK", "COST"],
    "mid_cap": ["DECK", "WSM", "POOL", "TOST", "DUOL", "FND", "WING", "CAVA", "TXRH", "WFRD"],
    "small_cap_etf": ["IWM"],  # Use ETF as proxy
    "high_beta": ["TSLA", "AMD", "COIN", "MARA", "RIVN", "PLTR", "SOFI", "RBLX", "SNAP", "ROKU"],
    "low_beta": ["PG", "JNJ", "KO", "PEP", "WMT", "CL", "GIS", "K", "MO", "PM"],
    "growth": ["NVDA", "TSLA", "AMD", "CRM", "NOW", "SHOP", "SNOW", "DDOG", "NET", "CRWD"],
    "value": ["BRK-B", "JPM", "BAC", "WFC", "CVX", "XOM", "PFE", "VZ", "T", "IBM"],
    "sector_tech": ["XLK"],
    "sector_health": ["XLV"],
    "sector_finance": ["XLF"],
    "sector_energy": ["XLE"],
    "sector_staples": ["XLP"],
    "sector_discretionary": ["XLY"],
    "sector_industrial": ["XLI"],
    "sector_utilities": ["XLU"],
    "index_spy": ["SPY"],
    "index_qqq": ["QQQ"],
    "index_iwm": ["IWM"],
    "index_dia": ["DIA"],
}

# Grouped for reporting
STOCK_TYPE_GROUPS = {
    "By Market Cap": ["mega_cap", "large_cap", "mid_cap", "small_cap_etf"],
    "By Style": ["growth", "value", "high_beta", "low_beta"],
    "By Sector": ["sector_tech", "sector_health", "sector_finance", "sector_energy",
                  "sector_staples", "sector_discretionary", "sector_industrial", "sector_utilities"],
    "By Index": ["index_spy", "index_qqq", "index_iwm", "index_dia"],
}


@dataclass
class ScenarioResult:
    """Result from testing one indicator in one scenario."""
    indicator: str
    scenario: str
    dimension: str  # "stock_type", "regime", "horizon", "signal_strength", "sector"
    ic: float
    ic_pvalue: float
    long_short_spread_ann: float
    hit_rate: float
    n_obs: int
    sharpe_of_spread: float


@dataclass
class HorizonResult:
    """Optimal holding period analysis for one indicator."""
    indicator: str
    ticker: str
    horizons: dict  # {horizon_days: {"ic": x, "spread": x, "hit": x}}
    optimal_horizon: int
    optimal_ic: float


@dataclass
class SignalStrengthResult:
    """How does performance vary with signal strength (deciles)?"""
    indicator: str
    ticker: str
    decile_returns: dict  # {decile: {horizon: mean_return}}
    extreme_spread: float  # top decile - bottom decile
    works_at_extremes: bool


@dataclass
class RegimeResult:
    """How does the indicator perform across market regimes?"""
    indicator: str
    regimes: dict  # {regime_name: {"ic": x, "spread": x, "n_obs": x}}
    best_regime: str
    worst_regime: str


@dataclass
class ConfluenceResult:
    """How do indicators perform when multiple agree?"""
    n_agreeing: int
    ic: float
    spread: float
    hit_rate: float
    n_obs: int


def _classify_regimes(benchmark_df: pd.DataFrame) -> pd.Series:
    """Classify each day into a market regime."""
    c = benchmark_df["Close"]

    # Rolling returns
    ret_60d = c.pct_change(60)
    ret_20d = c.pct_change(20)

    # Rolling volatility
    daily_ret = c.pct_change()
    vol_20d = daily_ret.rolling(20).std() * np.sqrt(252)
    vol_median = vol_20d.rolling(252).median()

    # 200-day MA trend
    ma200 = sma(c, 200)
    above_ma200 = c > ma200

    regime = pd.Series("neutral", index=benchmark_df.index)

    # Bull: above 200MA + positive 60d return
    regime[(above_ma200) & (ret_60d > 0.05)] = "bull"

    # Bear: below 200MA + negative 60d return
    regime[(~above_ma200) & (ret_60d < -0.05)] = "bear"

    # Sideways: within +/-5% over 60 days
    regime[(ret_60d.abs() < 0.05)] = "sideways"

    # Overlays
    high_vol = vol_20d > vol_median * 1.3
    low_vol = vol_20d < vol_median * 0.7

    # Corrections (sharp short-term drops in uptrend)
    correction = (above_ma200) & (ret_20d < -0.05)

    # Recovery (rising from bear)
    recovery = (~above_ma200) & (ret_20d > 0.05)

    regime[high_vol & (regime == "bull")] = "bull_volatile"
    regime[high_vol & (regime == "bear")] = "bear_volatile"
    regime[correction] = "correction"
    regime[recovery] = "recovery"
    regime[low_vol] = regime[low_vol].str.replace("neutral", "low_vol")

    return regime


def test_by_stock_type(indicator_name: str,
                       horizons: list = [5, 10, 20],
                       period: str = "10y",
                       verbose: bool = True) -> list:
    """Test one indicator across all stock type universes."""
    if indicator_name not in INDICATOR_REGISTRY:
        raise ValueError(f"Unknown indicator: {indicator_name}")

    reg = INDICATOR_REGISTRY[indicator_name]
    results = []

    for group_name, universe_keys in STOCK_TYPE_GROUPS.items():
        for ukey in universe_keys:
            tickers = STOCK_UNIVERSES[ukey]
            ics = []
            spreads = []
            hits = []
            n_total = 0

            for ticker in tickers:
                try:
                    df = fetch_data(ticker, period)
                    if len(df) < 300:
                        continue

                    ind_df = reg["fn"](df, **reg.get("params", {}))
                    signal = ind_df[reg["signal_col"]]
                    fwd = compute_forward_returns(df, horizons)

                    combined = pd.DataFrame({"signal": signal}, index=df.index)
                    for h in horizons:
                        combined[f"fwd_{h}d"] = fwd[f"fwd_{h}d"]
                    combined = combined.dropna()

                    if len(combined) < 100:
                        continue

                    n_total += len(combined)

                    # Use 10d horizon as the default evaluation horizon
                    eval_h = 10 if 10 in horizons else horizons[len(horizons)//2]
                    col = f"fwd_{eval_h}d"

                    from scipy.stats import spearmanr
                    corr, pval = spearmanr(combined["signal"], combined[col])
                    ics.append(corr)

                    # Quintile spread
                    combined["q"] = pd.qcut(combined["signal"], 5, labels=False, duplicates="drop")
                    qr = combined.groupby("q")[col].mean()
                    if len(qr) >= 5:
                        spread = (qr.iloc[-1] - qr.iloc[0]) * (252 / eval_h)
                        spreads.append(spread)

                    hit = (np.sign(combined["signal"]) == np.sign(combined[col])).mean()
                    hits.append(hit)

                except Exception:
                    continue

            if ics:
                avg_ic = np.mean(ics)
                avg_spread = np.mean(spreads) if spreads else 0
                avg_hit = np.mean(hits) if hits else 0.5

                # Sharpe of the spread
                spread_sharpe = avg_spread / (np.std(spreads) + 0.001) if len(spreads) > 1 else 0

                results.append(ScenarioResult(
                    indicator=indicator_name,
                    scenario=ukey,
                    dimension=group_name,
                    ic=avg_ic,
                    ic_pvalue=0,  # Averaged across tickers
                    long_short_spread_ann=avg_spread,
                    hit_rate=avg_hit,
                    n_obs=n_total,
                    sharpe_of_spread=spread_sharpe,
                ))

                if verbose:
                    label = f"{ukey:25s}"
                    print(f"    {label} IC={avg_ic:+.4f}  Spread={avg_spread:+.1%}  "
                          f"Hit={avg_hit:.1%}  n={n_total}")

    return results


def test_by_regime(indicator_name: str,
                   ticker: str = "SPY",
                   horizons: list = [5, 10, 20],
                   period: str = "10y",
                   verbose: bool = True) -> RegimeResult:
    """Test how an indicator performs across market regimes."""
    reg = INDICATOR_REGISTRY[indicator_name]

    df = fetch_data(ticker, period)
    regimes = _classify_regimes(df)

    ind_df = reg["fn"](df, **reg.get("params", {}))
    signal = ind_df[reg["signal_col"]]
    fwd = compute_forward_returns(df, horizons)

    eval_h = 10
    combined = pd.DataFrame({
        "signal": signal,
        "regime": regimes,
        f"fwd_{eval_h}d": fwd[f"fwd_{eval_h}d"],
    }).dropna()

    from scipy.stats import spearmanr

    regime_results = {}
    for regime_name in combined["regime"].unique():
        mask = combined["regime"] == regime_name
        subset = combined[mask]
        if len(subset) < 50:
            continue

        corr, pval = spearmanr(subset["signal"], subset[f"fwd_{eval_h}d"])

        # Quintile spread
        subset = subset.copy()
        try:
            subset["q"] = pd.qcut(subset["signal"], 5, labels=False, duplicates="drop")
            qr = subset.groupby("q")[f"fwd_{eval_h}d"].mean()
            spread = (qr.iloc[-1] - qr.iloc[0]) * (252 / eval_h) if len(qr) >= 5 else 0
        except Exception:
            spread = 0

        regime_results[regime_name] = {
            "ic": corr,
            "pvalue": pval,
            "spread": spread,
            "n_obs": len(subset),
            "pct_time": len(subset) / len(combined),
        }

        if verbose:
            print(f"    {regime_name:20s} IC={corr:+.4f}  Spread={spread:+.1%}  "
                  f"n={len(subset)} ({len(subset)/len(combined):.0%})")

    # Find best/worst
    best = max(regime_results, key=lambda r: abs(regime_results[r]["ic"])) if regime_results else "none"
    worst = min(regime_results, key=lambda r: abs(regime_results[r]["ic"])) if regime_results else "none"

    return RegimeResult(
        indicator=indicator_name,
        regimes=regime_results,
        best_regime=best,
        worst_regime=worst,
    )


def test_optimal_horizon(indicator_name: str,
                         ticker: str = "SPY",
                         horizons: list = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 60],
                         period: str = "10y",
                         verbose: bool = True) -> HorizonResult:
    """Find the optimal holding period for an indicator."""
    reg = INDICATOR_REGISTRY[indicator_name]

    df = fetch_data(ticker, period)
    ind_df = reg["fn"](df, **reg.get("params", {}))
    signal = ind_df[reg["signal_col"]]
    fwd = compute_forward_returns(df, horizons)

    from scipy.stats import spearmanr

    horizon_data = {}
    for h in horizons:
        col = f"fwd_{h}d"
        combined = pd.DataFrame({"signal": signal, "fwd": fwd[col]}).dropna()
        if len(combined) < 100:
            continue

        corr, pval = spearmanr(combined["signal"], combined["fwd"])

        combined["q"] = pd.qcut(combined["signal"], 5, labels=False, duplicates="drop")
        qr = combined.groupby("q")["fwd"].mean()
        spread = (qr.iloc[-1] - qr.iloc[0]) * (252 / h) if len(qr) >= 5 else 0
        hit = (np.sign(combined["signal"]) == np.sign(combined["fwd"])).mean()

        horizon_data[h] = {"ic": corr, "spread": spread, "hit": hit, "pvalue": pval}

        if verbose:
            print(f"    {h:3d}d  IC={corr:+.4f}  Spread={spread:+.1%}  Hit={hit:.1%}")

    # Find optimal (highest absolute IC)
    optimal_h = max(horizon_data, key=lambda h: abs(horizon_data[h]["ic"])) if horizon_data else 10

    return HorizonResult(
        indicator=indicator_name,
        ticker=ticker,
        horizons=horizon_data,
        optimal_horizon=optimal_h,
        optimal_ic=horizon_data.get(optimal_h, {}).get("ic", 0),
    )


def test_signal_strength(indicator_name: str,
                         ticker: str = "SPY",
                         eval_horizon: int = 10,
                         period: str = "10y",
                         verbose: bool = True) -> SignalStrengthResult:
    """Test whether the indicator works better at extremes (decile analysis)."""
    reg = INDICATOR_REGISTRY[indicator_name]

    df = fetch_data(ticker, period)
    ind_df = reg["fn"](df, **reg.get("params", {}))
    signal = ind_df[reg["signal_col"]]
    fwd = compute_forward_returns(df, [eval_horizon])

    combined = pd.DataFrame({
        "signal": signal,
        "fwd": fwd[f"fwd_{eval_horizon}d"],
    }).dropna()

    # Split into deciles
    combined["decile"] = pd.qcut(combined["signal"], 10, labels=False, duplicates="drop") + 1

    decile_returns = {}
    for d in sorted(combined["decile"].unique()):
        subset = combined[combined["decile"] == d]
        mean_ret = subset["fwd"].mean()
        ann_ret = mean_ret * (252 / eval_horizon)
        decile_returns[int(d)] = {
            "mean_return": mean_ret,
            "annualized": ann_ret,
            "n_obs": len(subset),
            "signal_mean": subset["signal"].mean(),
        }
        if verbose:
            direction = "▲" if ann_ret > 0 else "▼"
            print(f"    Decile {d:2d}  Signal={subset['signal'].mean():+8.3f}  "
                  f"Fwd {eval_horizon}d={ann_ret:+6.1%} {direction}  n={len(subset)}")

    # Extreme spread
    d_keys = sorted(decile_returns.keys())
    if len(d_keys) >= 2:
        extreme = (decile_returns[d_keys[-1]]["annualized"] -
                   decile_returns[d_keys[0]]["annualized"])
    else:
        extreme = 0

    # Check monotonicity in outer deciles
    works_extreme = False
    if len(d_keys) >= 4:
        bottom_2 = np.mean([decile_returns[d_keys[0]]["annualized"],
                           decile_returns[d_keys[1]]["annualized"]])
        top_2 = np.mean([decile_returns[d_keys[-1]]["annualized"],
                        decile_returns[d_keys[-2]]["annualized"]])
        mid = np.mean([decile_returns[d_keys[len(d_keys)//2]]["annualized"]])
        # Works at extremes if outer deciles differ significantly from middle
        works_extreme = abs(top_2 - bottom_2) > abs(mid) * 0.5

    return SignalStrengthResult(
        indicator=indicator_name,
        ticker=ticker,
        decile_returns=decile_returns,
        extreme_spread=extreme,
        works_at_extremes=works_extreme,
    )


def test_confluence(indicator_names: list,
                    ticker: str = "SPY",
                    eval_horizon: int = 10,
                    period: str = "10y",
                    verbose: bool = True) -> list:
    """Test how forward returns improve when multiple indicators agree on direction."""
    df = fetch_data(ticker, period)
    fwd = compute_forward_returns(df, [eval_horizon])

    # Compute all indicator signals and their directions
    signals = pd.DataFrame(index=df.index)
    for name in indicator_names:
        if name not in INDICATOR_REGISTRY:
            continue
        reg = INDICATOR_REGISTRY[name]
        try:
            ind_df = reg["fn"](df, **reg.get("params", {}))
            signal = ind_df[reg["signal_col"]]
            # Normalize to z-score
            mean = signal.rolling(63, min_periods=20).mean()
            std = signal.rolling(63, min_periods=20).std().replace(0, np.nan)
            z = (signal - mean) / std
            signals[name] = z
        except Exception:
            continue

    if signals.empty:
        return []

    # Count how many indicators are positive (bullish)
    n_bullish = (signals > 0).sum(axis=1)
    n_total = signals.notna().sum(axis=1)

    combined = pd.DataFrame({
        "n_bullish": n_bullish,
        "n_total": n_total,
        "pct_bullish": n_bullish / n_total.replace(0, np.nan),
        "fwd": fwd[f"fwd_{eval_horizon}d"],
    }).dropna()

    results = []

    # Test by number of agreeing indicators
    from scipy.stats import spearmanr

    for n_agree in sorted(combined["n_bullish"].unique()):
        subset = combined[combined["n_bullish"] == n_agree]
        if len(subset) < 30:
            continue

        mean_fwd = subset["fwd"].mean()
        ann_ret = mean_fwd * (252 / eval_horizon)
        hit = (subset["fwd"] > 0).mean()

        cr = ConfluenceResult(
            n_agreeing=int(n_agree),
            ic=0,  # Not applicable at this level
            spread=ann_ret,
            hit_rate=hit,
            n_obs=len(subset),
        )
        results.append(cr)

        if verbose:
            bar = "█" * int(hit * 40)
            print(f"    {int(n_agree):2d}/{int(combined['n_total'].median())} bullish  "
                  f"Fwd {eval_horizon}d={ann_ret:+6.1%}  Hit={hit:.1%}  "
                  f"n={len(subset):4d}  {bar}")

    return results


def run_full_scenario_analysis(indicator_name: str, verbose: bool = True) -> dict:
    """Run all scenario tests for one indicator."""
    results = {}

    if verbose:
        print(f"\n{'='*70}")
        print(f"  SCENARIO ANALYSIS: {indicator_name}")
        print(f"  {INDICATOR_REGISTRY[indicator_name]['description']}")
        print(f"{'='*70}")

    # 1. Stock type analysis
    if verbose:
        print(f"\n  [1] STOCK TYPE ANALYSIS")
        print(f"  {'─'*60}")
    results["stock_type"] = test_by_stock_type(indicator_name, verbose=verbose)

    # 2. Market regime analysis
    if verbose:
        print(f"\n  [2] MARKET REGIME ANALYSIS (SPY)")
        print(f"  {'─'*60}")
    results["regime"] = test_by_regime(indicator_name, verbose=verbose)

    # 3. Optimal horizon
    if verbose:
        print(f"\n  [3] OPTIMAL HOLDING PERIOD (SPY)")
        print(f"  {'─'*60}")
    results["horizon"] = test_optimal_horizon(indicator_name, verbose=verbose)

    # 4. Signal strength (decile analysis)
    if verbose:
        print(f"\n  [4] SIGNAL STRENGTH — DECILE ANALYSIS (SPY)")
        print(f"  {'─'*60}")
    results["signal_strength"] = test_signal_strength(indicator_name, verbose=verbose)

    return results
