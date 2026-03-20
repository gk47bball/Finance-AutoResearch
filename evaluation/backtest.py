"""Walk-forward backtester with quarterly rebalancing."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from data.prices import get_prices, get_returns, get_benchmark_returns
from data.fundamentals import get_key_ratios
from analysis.screener import run_screen
from analysis.scoring import score_stocks, select_portfolio
from evaluation.metrics import compute_all_metrics, PRIMARY_METRIC
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    metrics: dict = field(default_factory=dict)
    portfolios: list = field(default_factory=list)  # [{date, tickers, weights}]
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    benchmark_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    primary_metric: float = 0.0


def _generate_rebalance_dates(start: str, end: str, months: int) -> list[pd.Timestamp]:
    """Generate rebalancing dates at regular intervals."""
    dates = pd.date_range(start=start, end=end, freq=f"{months}MS")
    return list(dates)


def run_backtest(
    strategy_module,
    lookback_years: int = 5,
    rebalance_months: int = 3,
    benchmark: str = "SPY",
    initial_capital: float = 100000,
    commission_bps: float = 5,
    show_progress: bool = True,
) -> BacktestResult:
    """Run a walk-forward backtest of the strategy.

    For each rebalance date:
      1. Screen stocks using strategy.SCREENS
      2. Score surviving stocks with strategy.FACTORS
      3. Select top-N portfolio per strategy.PORTFOLIO
      4. Hold until next rebalance, compute daily returns
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Get strategy config
    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    screens = getattr(strategy_module, "SCREENS", [])
    factors = getattr(strategy_module, "FACTORS", {})
    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})

    # Build universe once (use current constituents — known survivorship bias)
    from data.universe import build_universe
    full_universe = build_universe(universe_cfg)

    # Generate rebalance dates
    rebalance_dates = _generate_rebalance_dates(start_str, end_str, rebalance_months)
    if len(rebalance_dates) < 2:
        return BacktestResult()

    # Get benchmark returns for full period
    bench_returns = get_benchmark_returns(benchmark, start_str, end_str)
    if bench_returns.empty:
        return BacktestResult()

    all_port_returns = []
    portfolios_log = []

    progress_ctx = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        disable=not show_progress,
    )

    with progress_ctx as progress:
        task = progress.add_task("Backtesting...", total=len(rebalance_dates) - 1)

        for i in range(len(rebalance_dates) - 1):
            reb_date = rebalance_dates[i]
            next_reb = rebalance_dates[i + 1]

            # Screen: use ratios as of rebalance date
            # (Limitation: yfinance gives current ratios, not point-in-time)
            screen_result = run_screen(full_universe, screens, show_progress=False)
            passed = screen_result.passed

            if not passed:
                # No stocks pass — hold cash (0 return)
                period_dates = pd.date_range(reb_date, next_reb, freq="B")[1:]
                cash_returns = pd.Series(0.0, index=period_dates)
                all_port_returns.append(cash_returns)
                portfolios_log.append({
                    "date": str(reb_date.date()),
                    "tickers": [],
                    "weights": {},
                })
                progress.update(task, advance=1)
                continue

            # Score
            scored = score_stocks(passed, factors, ratios_cache=screen_result.data)
            if scored.empty:
                progress.update(task, advance=1)
                continue

            # Select portfolio
            portfolio = select_portfolio(scored, portfolio_cfg)
            holdings = dict(zip(portfolio["ticker"], portfolio["weight"]))

            portfolios_log.append({
                "date": str(reb_date.date()),
                "tickers": list(holdings.keys()),
                "weights": holdings,
            })

            # Compute portfolio returns for this period
            period_start = reb_date.strftime("%Y-%m-%d")
            period_end = next_reb.strftime("%Y-%m-%d")

            weighted_returns = None
            for ticker, weight in holdings.items():
                ret = get_returns(ticker, period_start, period_end)
                if ret.empty:
                    continue
                if weighted_returns is None:
                    weighted_returns = ret * weight
                else:
                    # Align indices
                    weighted_returns, aligned_ret = weighted_returns.align(ret * weight, fill_value=0)
                    weighted_returns = weighted_returns + aligned_ret

            if weighted_returns is not None:
                # Deduct commission at rebalance
                commission = commission_bps / 10000
                if not weighted_returns.empty:
                    weighted_returns.iloc[0] -= commission
                all_port_returns.append(weighted_returns)

            progress.update(task, advance=1)

    if not all_port_returns:
        return BacktestResult()

    # Concatenate all period returns
    port_returns = pd.concat(all_port_returns).sort_index()
    port_returns = port_returns[~port_returns.index.duplicated(keep="first")]

    # Align with benchmark
    aligned = pd.concat([port_returns, bench_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return BacktestResult()
    aligned.columns = ["portfolio", "benchmark"]

    # Compute metrics
    metrics = compute_all_metrics(aligned["portfolio"], aligned["benchmark"])

    # Equity curves
    equity_curve = (1 + aligned["portfolio"]).cumprod() * initial_capital
    benchmark_curve = (1 + aligned["benchmark"]).cumprod() * initial_capital

    return BacktestResult(
        portfolio_returns=aligned["portfolio"],
        benchmark_returns=aligned["benchmark"],
        metrics=metrics,
        portfolios=portfolios_log,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        primary_metric=metrics.get(PRIMARY_METRIC, 0.0),
    )
