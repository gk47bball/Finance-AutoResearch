"""
Housing Alpha Backtester
=========================
Monthly-rebalance backtester for the housing signal → ETF strategy.

Key differences from the Four Pillars backtester:
  - Monthly rebalance (not daily) — housing data is monthly
  - Signals are lagged by 1 month (you can't trade on data the same month it's released)
  - Multiple tickers can be traded simultaneously
  - Transaction costs are per-rebalance, not per-day
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from technical_analysis.bot.housing_alpha.engine import (
    HousingAlphaEngine,
    HOUSING_TICKERS,
    DEFAULT_PARAMS,
)


def _fetch_etf_prices(ticker: str, start: str = "2005-01-01") -> pd.Series:
    """Fetch monthly ETF prices using yfinance."""
    try:
        from technical_analysis.backtest.signal_tester import fetch_data
        df = fetch_data(ticker, period="max")
        # Resample to monthly (end of month)
        monthly = df["Close"].resample("ME").last()
        monthly = monthly.loc[start:]
        return monthly
    except Exception as e:
        print(f"  [housing-bt] Failed to fetch {ticker}: {e}")
        return pd.Series(dtype=float)


def backtest_housing_alpha(
    ticker: str = "XHB",
    start: str = "2005-01-01",
    end: Optional[str] = None,
    params: Optional[dict] = None,
    verbose: bool = True,
    commission_bps: float = 10,  # 10bps round-trip per monthly rebalance
) -> dict:
    """
    Backtest the housing alpha strategy on a single ticker.

    Args:
        ticker: ETF to trade (XHB, ITB, XLRE, VNQ)
        start: backtest start date
        end: backtest end date (default: now)
        params: parameter dict (default: load from file)
        verbose: print progress
        commission_bps: round-trip commission in basis points

    Returns:
        dict with keys: sharpe_ratio, annual_return, max_drawdown,
        benchmark_sharpe, benchmark_annual_return, trade_count,
        win_rate, regime_summary, monthly_returns
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()

    engine = HousingAlphaEngine(params=params)

    # Get historical positions (monthly)
    hist = engine.compute_historical(ticker=ticker, start=start)
    if hist.empty:
        return {"sharpe_ratio": 0, "error": "No housing data"}

    # Get ETF prices (monthly)
    prices = _fetch_etf_prices(ticker, start=start)
    if prices.empty:
        return {"sharpe_ratio": 0, "error": f"No price data for {ticker}"}

    # Align: use intersection of dates
    common_idx = hist.index.intersection(prices.index)
    if len(common_idx) < 24:
        return {"sharpe_ratio": 0, "error": f"Only {len(common_idx)} months of overlap"}

    hist = hist.loc[common_idx]
    prices = prices.loc[common_idx]

    if end:
        end_ts = pd.Timestamp(end)
        hist = hist.loc[:end_ts]
        prices = prices.loc[:end_ts]

    # Lag positions by 1 month (can't trade on data same month it's released)
    position = hist["position"].shift(1).fillna(0)

    # Compute returns
    etf_returns = prices.pct_change().fillna(0)

    # Strategy returns = position * ETF return - commissions on rebalances
    position_changes = position.diff().abs().fillna(0)
    commission_cost = position_changes * (commission_bps / 10000)

    strategy_returns = position * etf_returns - commission_cost

    # Benchmark = buy and hold
    benchmark_returns = etf_returns

    # Compute metrics
    n_years = len(strategy_returns) / 12

    # Strategy metrics
    strat_mean = float(strategy_returns.mean()) * 12  # annualized
    strat_std = float(strategy_returns.std()) * np.sqrt(12)
    sharpe = strat_mean / strat_std if strat_std > 0 else 0

    strat_cum = (1 + strategy_returns).cumprod()
    strat_annual = float(strat_cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    strat_dd = float((strat_cum / strat_cum.cummax() - 1).min())

    # Benchmark metrics
    bench_mean = float(benchmark_returns.mean()) * 12
    bench_std = float(benchmark_returns.std()) * np.sqrt(12)
    bench_sharpe = bench_mean / bench_std if bench_std > 0 else 0

    bench_cum = (1 + benchmark_returns).cumprod()
    bench_annual = float(bench_cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0

    # Trade count (meaningful position changes > rebalance threshold)
    rebalance_t = params.get("rebalance_threshold", 0.10)
    trades = (position_changes > rebalance_t).sum()

    # Win rate (months with positive strategy return when invested)
    invested_months = strategy_returns[position > 0]
    win_rate = float((invested_months > 0).sum() / len(invested_months)) if len(invested_months) > 0 else 0

    # Regime summary
    regime_summary = {}
    for regime in ["HOUSING_BULL", "HOUSING_NEUTRAL", "HOUSING_BEAR"]:
        mask = hist["regime"] == regime
        if mask.sum() > 0:
            regime_ret = strategy_returns[mask]
            regime_summary[regime] = {
                "months": int(mask.sum()),
                "avg_monthly_return": round(float(regime_ret.mean()) * 100, 2),
                "avg_position": round(float(position[mask].mean()), 2),
            }

    results = {
        "ticker": ticker,
        "period": f"{hist.index[0].strftime('%Y-%m')} to {hist.index[-1].strftime('%Y-%m')}",
        "months": len(strategy_returns),
        "sharpe_ratio": round(sharpe, 4),
        "annual_return": round(strat_annual, 4),
        "max_drawdown": round(strat_dd, 4),
        "total_return": round(float(strat_cum.iloc[-1] - 1), 4),
        "benchmark_sharpe": round(bench_sharpe, 4),
        "benchmark_annual_return": round(bench_annual, 4),
        "benchmark_total_return": round(float(bench_cum.iloc[-1] - 1), 4),
        "trade_count": int(trades),
        "win_rate": round(win_rate, 4),
        "regime_summary": regime_summary,
        "beats_benchmark": bool(sharpe > bench_sharpe),
    }

    if verbose:
        print(f"\n  Housing Alpha Backtest: {ticker}")
        print(f"  Period: {results['period']} ({results['months']} months)")
        print(f"  ────────────────────────────────────────")
        print(f"  Strategy Sharpe:     {sharpe:+.3f}")
        print(f"  Benchmark Sharpe:    {bench_sharpe:+.3f}")
        bm_flag = "✓" if sharpe > bench_sharpe else "✗"
        print(f"  Beats Benchmark:     {bm_flag}")
        print(f"  Strategy Return:     {strat_annual:+.1%} ann ({strat_cum.iloc[-1] - 1:+.1%} total)")
        print(f"  Benchmark Return:    {bench_annual:+.1%} ann ({bench_cum.iloc[-1] - 1:+.1%} total)")
        print(f"  Max Drawdown:        {strat_dd:.1%}")
        print(f"  Win Rate:            {win_rate:.0%}")
        print(f"  Trades:              {trades}")
        print(f"  ────────────────────────────────────────")
        for regime, info in regime_summary.items():
            print(f"  {regime}: {info['months']}mo, avg pos {info['avg_position']:.0%}, "
                  f"avg ret {info['avg_monthly_return']:+.2f}%/mo")

    return results


def backtest_housing_multi(
    tickers: Optional[list[str]] = None,
    start: str = "2005-01-01",
    end: Optional[str] = None,
    params: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Backtest housing alpha on multiple tickers, compute composite Sharpe.

    Returns:
        dict with per-ticker results and composite_sharpe
    """
    if tickers is None:
        tickers = ["XHB", "ITB"]  # Focus on pure homebuilders

    if params is None:
        from technical_analysis.bot.housing_alpha.engine import load_params
        params = load_params()

    results = {}
    sharpes = []
    bench_sharpes = []

    # Equal weight for now
    weights = {t: 1.0 / len(tickers) for t in tickers}

    for ticker in tickers:
        r = backtest_housing_alpha(
            ticker=ticker,
            start=start,
            end=end,
            params=params,
            verbose=verbose,
        )
        results[ticker] = r
        if "error" not in r:
            sharpes.append(r["sharpe_ratio"] * weights[ticker])
            bench_sharpes.append(r["benchmark_sharpe"] * weights[ticker])

    # Composite Sharpe (weighted average with penalty for underperforming benchmark)
    composite = 0
    for ticker, r in results.items():
        if "error" in r:
            continue
        w = weights[ticker]
        s = r["sharpe_ratio"]
        # 10% penalty if underperforming benchmark
        if not r.get("beats_benchmark", False):
            s *= 0.90
        composite += s * w

    results["composite_sharpe"] = round(composite, 4)

    if verbose:
        print(f"\n  ═══════════════════════════════════════")
        print(f"  COMPOSITE SHARPE: {composite:+.4f}")
        for t in tickers:
            r = results.get(t, {})
            flag = "✓" if r.get("beats_benchmark") else "✗"
            print(f"    {t}: {r.get('sharpe_ratio', 0):+.3f} (BM: {r.get('benchmark_sharpe', 0):+.3f}) {flag}")
        print(f"  ═══════════════════════════════════════")

    return results
