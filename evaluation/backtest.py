"""Walk-forward backtester with flexible rebalancing, turnover tracking, and short support."""

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
    # Turnover tracking
    turnover_per_rebalance: list = field(default_factory=list)
    avg_annual_turnover: float = 0.0
    total_commission_drag: float = 0.0


def _compute_turnover(prev_weights: dict, new_weights: dict) -> float:
    """One-way turnover: sum of absolute weight changes / 2."""
    all_tickers = set(list(prev_weights.keys()) + list(new_weights.keys()))
    total_change = sum(
        abs(new_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
        for t in all_tickers
    )
    return total_change / 2.0


def _generate_rebalance_dates(start: str, end: str, freq: str = "quarterly", months: int = 3) -> list[pd.Timestamp]:
    """Generate rebalancing dates at regular intervals.

    freq: 'weekly', 'monthly', 'quarterly', or 'custom' (uses months param)
    """
    freq_map = {
        "weekly": "W-FRI",
        "monthly": "MS",
        "quarterly": "3MS",
    }
    if freq in freq_map:
        pd_freq = freq_map[freq]
    else:
        pd_freq = f"{months}MS"

    dates = pd.date_range(start=start, end=end, freq=pd_freq)
    return list(dates)


def run_backtest(
    strategy_module,
    lookback_years: int = 5,
    rebalance_months: int = 3,
    rebalance_freq: str = None,
    benchmark: str = "SPY",
    initial_capital: float = 100000,
    commission_bps: float = 5,
    start_date: str = None,
    end_date: str = None,
    allow_short: bool = False,
    short_n: int = 0,
    borrow_cost_bps: float = 0,
    show_progress: bool = True,
) -> BacktestResult:
    """Run a walk-forward backtest of the strategy.

    For each rebalance date:
      1. Screen stocks using strategy.SCREENS
      2. Score surviving stocks with strategy.FACTORS
      3. Select top-N portfolio per strategy.PORTFOLIO
      4. Optionally short bottom-N stocks
      5. Hold until next rebalance, compute daily returns
      6. Track turnover and realistic transaction costs
    """
    # Date range
    if end_date:
        end_dt = pd.Timestamp(end_date)
    else:
        end_dt = pd.Timestamp(datetime.now())

    if start_date:
        start_dt = pd.Timestamp(start_date)
    else:
        start_dt = end_dt - timedelta(days=lookback_years * 365)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # Get strategy config
    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    screens = getattr(strategy_module, "SCREENS", [])
    factors = getattr(strategy_module, "FACTORS", {})
    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})

    # Build universe once (use current constituents — known survivorship bias)
    from data.universe import build_universe
    full_universe = build_universe(universe_cfg)

    # Determine rebalance frequency
    if rebalance_freq is None:
        freq_str = portfolio_cfg.get("rebalance_frequency", "quarterly")
    else:
        freq_str = rebalance_freq

    # Generate rebalance dates
    rebalance_dates = _generate_rebalance_dates(start_str, end_str, freq=freq_str, months=rebalance_months)
    if len(rebalance_dates) < 2:
        return BacktestResult()

    # Get benchmark returns for full period
    bench_returns = get_benchmark_returns(benchmark, start_str, end_str)
    if bench_returns.empty:
        return BacktestResult()

    all_port_returns = []
    portfolios_log = []
    turnover_log = []
    total_commission = 0.0
    prev_weights = {}

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
                turnover = _compute_turnover(prev_weights, {})
                turnover_log.append(turnover)
                prev_weights = {}
                progress.update(task, advance=1)
                continue

            # Score
            scored = score_stocks(passed, factors, ratios_cache=screen_result.data)
            if scored.empty:
                progress.update(task, advance=1)
                continue

            # Select long portfolio
            portfolio = select_portfolio(scored, portfolio_cfg)
            long_holdings = dict(zip(portfolio["ticker"], portfolio["weight"]))

            # Select short portfolio if enabled
            short_holdings = {}
            if allow_short and short_n > 0 and len(scored) >= short_n:
                bottom = scored.nsmallest(short_n, "composite_score")
                short_weight = 1.0 / short_n  # equal weight shorts
                short_holdings = {row["ticker"]: short_weight for _, row in bottom.iterrows()}

            # Combined weights for turnover calculation (shorts are negative)
            combined_weights = {t: w * 0.5 for t, w in long_holdings.items()} if allow_short else dict(long_holdings)
            if short_holdings:
                for t, w in short_holdings.items():
                    combined_weights[t] = combined_weights.get(t, 0) - w * 0.5

            # Compute turnover
            turnover = _compute_turnover(prev_weights, combined_weights)
            turnover_log.append(turnover)
            prev_weights = combined_weights

            # Commission based on turnover
            period_commission = turnover * (commission_bps / 10000) * 2  # two-way
            total_commission += period_commission

            portfolios_log.append({
                "date": str(reb_date.date()),
                "tickers": list(long_holdings.keys()),
                "weights": long_holdings,
                "short_tickers": list(short_holdings.keys()) if short_holdings else [],
                "short_weights": short_holdings if short_holdings else {},
                "turnover": turnover,
            })

            # Compute portfolio returns for this period
            period_start = reb_date.strftime("%Y-%m-%d")
            period_end = next_reb.strftime("%Y-%m-%d")

            # Long leg returns
            weighted_returns = None
            for ticker, weight in long_holdings.items():
                ret = get_returns(ticker, period_start, period_end)
                if ret.empty:
                    continue
                w = weight * 0.5 if allow_short else weight
                if weighted_returns is None:
                    weighted_returns = ret * w
                else:
                    weighted_returns, aligned_ret = weighted_returns.align(ret * w, fill_value=0)
                    weighted_returns = weighted_returns + aligned_ret

            # Short leg returns (negative weight = profit when stock goes down)
            if short_holdings:
                for ticker, weight in short_holdings.items():
                    ret = get_returns(ticker, period_start, period_end)
                    if ret.empty:
                        continue
                    short_ret = -ret * weight * 0.5  # profit from price decline
                    if weighted_returns is None:
                        weighted_returns = short_ret
                    else:
                        weighted_returns, aligned_ret = weighted_returns.align(short_ret, fill_value=0)
                        weighted_returns = weighted_returns + aligned_ret

                # Deduct daily borrow cost for shorts
                if weighted_returns is not None and borrow_cost_bps > 0:
                    daily_borrow = (borrow_cost_bps / 10000) / 252
                    total_short_weight = sum(short_holdings.values()) * 0.5
                    weighted_returns -= daily_borrow * total_short_weight

            if weighted_returns is not None:
                # Deduct commission at rebalance
                if not weighted_returns.empty:
                    weighted_returns.iloc[0] -= period_commission
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

    # Turnover stats
    years = len(aligned) / 252
    avg_turnover = sum(turnover_log) / max(years, 0.01) if turnover_log else 0.0

    return BacktestResult(
        portfolio_returns=aligned["portfolio"],
        benchmark_returns=aligned["benchmark"],
        metrics=metrics,
        portfolios=portfolios_log,
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        primary_metric=metrics.get(PRIMARY_METRIC, 0.0),
        turnover_per_rebalance=turnover_log,
        avg_annual_turnover=avg_turnover,
        total_commission_drag=total_commission,
    )
