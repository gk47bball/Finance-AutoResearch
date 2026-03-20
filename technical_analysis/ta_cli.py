"""
Technical Analysis CLI
=======================
Commands for testing, backtesting, and optimizing technical indicators.

Usage:
    python -m technical_analysis.ta_cli alpha-test        # Test each indicator's predictive power
    python -m technical_analysis.ta_cli backtest           # Run combined strategy backtest
    python -m technical_analysis.ta_cli indicator-report   # Full report on all indicators
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def _load_strategy():
    """Load the TA strategy module."""
    import importlib
    spec = importlib.util.spec_from_file_location(
        "strategy_ta",
        os.path.join(os.path.dirname(__file__), "strategy_ta.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@click.group()
def cli():
    """Technical Analysis Mastery — AutoResearch for TA Indicators."""
    pass


@cli.command("alpha-test")
@click.option("--ticker", "-t", default=None, help="Test on specific ticker only")
@click.option("--indicator", "-i", default=None, help="Test specific indicator only")
def alpha_test(ticker, indicator):
    """Test each indicator's standalone predictive power."""
    from technical_analysis.backtest.ta_backtest import run_alpha_test

    strategy = _load_strategy()

    if ticker:
        strategy.EVALUATION["test_tickers"] = [ticker.upper()]

    console.print("\n[bold blue]Technical Analysis Alpha Test[/bold blue]")
    console.print(f"Testing on: {strategy.EVALUATION.get('test_tickers', ['SPY'])}\n")

    if indicator:
        # Test only one indicator
        from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
        if indicator not in INDICATOR_REGISTRY:
            console.print(f"[red]Unknown indicator: {indicator}[/red]")
            console.print(f"Available: {', '.join(INDICATOR_REGISTRY.keys())}")
            return
        # Temporarily enable only this indicator
        for name in strategy.INDICATORS:
            strategy.INDICATORS[name]["enabled"] = (name == indicator)
            if name == indicator:
                strategy.INDICATORS[name]["weight"] = 1.0

    results = run_alpha_test(strategy, verbose=True)

    # Summary table
    console.print("\n")
    table = Table(title="Indicator Alpha Summary", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Consistency", justify="right")
    table.add_column("IC (5d)", justify="right")
    table.add_column("IC (10d)", justify="right")
    table.add_column("IC (20d)", justify="right")
    table.add_column("Spread (10d)", justify="right")
    table.add_column("Verdict", justify="center")

    for name, result in sorted(results.items(),
                               key=lambda x: x[1].composite_score, reverse=True):
        score = result.composite_score
        if score >= 40:
            verdict = "[bold green]ALPHA[/bold green]"
        elif score >= 20:
            verdict = "[yellow]WEAK[/yellow]"
        else:
            verdict = "[red]NOISE[/red]"

        table.add_row(
            name,
            f"{score:.1f}",
            f"{result.consistency:.0%}",
            f"{result.avg_ic.get(5, 0):+.4f}",
            f"{result.avg_ic.get(10, 0):+.4f}",
            f"{result.avg_ic.get(20, 0):+.4f}",
            f"{result.avg_spread.get(10, 0):+.1%}",
            verdict,
        )

    console.print(table)


@cli.command()
def backtest():
    """Run the combined TA strategy backtest."""
    from technical_analysis.backtest.ta_backtest import run_strategy_backtest

    strategy = _load_strategy()
    console.print("\n[bold blue]TA Strategy Backtest[/bold blue]\n")

    result = run_strategy_backtest(strategy, verbose=True)

    table = Table(title="TA Strategy Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
    table.add_row("Annual Return", f"{result.annual_return:.1%}")
    table.add_row("Annual Volatility", f"{result.annual_volatility:.1%}")
    table.add_row("Max Drawdown", f"{result.max_drawdown:.1%}")
    table.add_row("Total Return", f"{result.total_return:.1%}")
    table.add_row("Win Rate", f"{result.win_rate:.1%}")
    table.add_row("# Trades", str(result.n_trades))
    table.add_row("Avg Trade Duration", f"{result.avg_trade_duration:.1f} days")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Exposure", f"{result.exposure_pct:.1%}")
    table.add_row("─" * 20, "─" * 10)
    table.add_row("Benchmark Sharpe", f"{result.benchmark_sharpe:.3f}")
    table.add_row("Benchmark Return", f"{result.benchmark_return:.1%}")
    table.add_row("Alpha", f"{result.alpha:.1%}")

    console.print(table)


@cli.command("scenario-backtest")
def scenario_backtest():
    """Run the scenario-aware backtest (regime + confluence + strength filters)."""
    from technical_analysis.backtest.ta_backtest import run_scenario_backtest, run_strategy_backtest

    strategy = _load_strategy()

    # Run both for comparison
    console.print("\n[bold blue]Scenario-Aware TA Backtest[/bold blue]")
    console.print("[dim]Comparing: standard backtest vs scenario-filtered backtest[/dim]\n")

    console.print("[bold]Standard Backtest:[/bold]")
    standard = run_strategy_backtest(strategy, verbose=False)

    console.print("[bold]Scenario-Aware Backtest:[/bold]")
    scenario = run_scenario_backtest(strategy, verbose=True)

    # Comparison table
    table = Table(title="Standard vs Scenario-Aware", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Standard", justify="right")
    table.add_column("Scenario", justify="right", style="green")
    table.add_column("Delta", justify="right")

    metrics = [
        ("Sharpe Ratio", "sharpe_ratio", ".3f"),
        ("Annual Return", "annual_return", ".1%"),
        ("Annual Volatility", "annual_volatility", ".1%"),
        ("Max Drawdown", "max_drawdown", ".1%"),
        ("Total Return", "total_return", ".1%"),
        ("Win Rate", "win_rate", ".1%"),
        ("# Trades", "n_trades", "d"),
        ("Avg Hold Days", "avg_trade_duration", ".1f"),
        ("Profit Factor", "profit_factor", ".2f"),
        ("Exposure", "exposure_pct", ".1%"),
    ]

    for label, key, fmt in metrics:
        std_val = getattr(standard, key)
        scn_val = getattr(scenario, key)
        delta = scn_val - std_val
        fmt_str = f"{{:{fmt}}}"
        delta_str = f"{delta:+{fmt[1:]}}" if fmt != "d" else f"{delta:+d}"
        color = "[green]" if delta > 0 and key not in ("annual_volatility",) else ""
        if key == "max_drawdown":
            color = "[green]" if delta > 0 else "[red]"
        elif key in ("annual_volatility",):
            color = "[green]" if delta < 0 else ""
        table.add_row(label, fmt_str.format(std_val), fmt_str.format(scn_val), f"{color}{delta_str}")

    table.add_row("─" * 20, "─" * 10, "─" * 10, "─" * 10)
    table.add_row("Benchmark Sharpe", f"{standard.benchmark_sharpe:.3f}",
                  f"{scenario.benchmark_sharpe:.3f}", "")

    console.print("\n")
    console.print(table)

    # Show scenario filter details
    sf = scenario.metrics.get("scenario_filters", {})
    console.print(f"\n[dim]Scenario mode: regime_boost={sf.get('regime_boost')}, "
                  f"confluence_boost={sf.get('confluence_boost')}, "
                  f"regime_filter={sf.get('regime_filter')}, "
                  f"min_hold={sf.get('min_hold')}d[/dim]")


@cli.command("multi-ticker")
def multi_ticker():
    """Run the strategy across all scenario-validated tickers."""
    from technical_analysis.backtest.ta_backtest import run_strategy_backtest
    import copy

    strategy = _load_strategy()
    universe = getattr(strategy, "UNIVERSE", {})

    # Test across primary + multi-ticker + sector ETFs
    tickers = list(set(
        universe.get("multi_ticker", ["SPY"]) +
        universe.get("sector_etfs", [])
    ))
    tickers.sort()

    console.print("\n[bold blue]Multi-Ticker Strategy Backtest[/bold blue]")
    console.print(f"[dim]Testing {len(tickers)} ETFs validated by scenario analysis[/dim]\n")

    table = Table(title="Strategy Performance by Ticker", box=box.ROUNDED)
    table.add_column("Ticker", style="cyan")
    table.add_column("Sharpe", justify="right")
    table.add_column("Annual Ret", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Exposure", justify="right")
    table.add_column("Bench Sharpe", justify="right", style="dim")
    table.add_column("Alpha", justify="right")
    table.add_column("Verdict", justify="center")

    total_sharpe = 0
    total_alpha = 0
    n_tested = 0

    for ticker in tickers:
        try:
            # Modify universe for this ticker
            strategy.UNIVERSE["tickers"] = [ticker]
            strategy.EVALUATION["benchmark"] = ticker

            result = run_strategy_backtest(strategy, verbose=False)

            sharpe_color = "[green]" if result.sharpe_ratio > result.benchmark_sharpe else "[red]"
            alpha_color = "[green]" if result.alpha > 0 else "[red]"
            verdict = "[bold green]WIN[/bold green]" if result.sharpe_ratio > result.benchmark_sharpe else "[red]LOSE[/red]"

            table.add_row(
                ticker,
                f"{sharpe_color}{result.sharpe_ratio:.3f}",
                f"{result.annual_return:.1%}",
                f"{result.max_drawdown:.1%}",
                f"{result.win_rate:.1%}",
                str(result.n_trades),
                f"{result.exposure_pct:.0%}",
                f"{result.benchmark_sharpe:.3f}",
                f"{alpha_color}{result.alpha:.1%}",
                verdict,
            )

            total_sharpe += result.sharpe_ratio
            total_alpha += result.alpha
            n_tested += 1

        except Exception as e:
            table.add_row(ticker, "—", "—", "—", "—", "—", "—", "—", "—", f"[red]ERR[/red]")

    console.print(table)

    if n_tested > 0:
        console.print(f"\n  Average Sharpe across {n_tested} tickers: {total_sharpe/n_tested:.3f}")
        console.print(f"  Average Alpha: {total_alpha/n_tested:.1%}")
        wins = sum(1 for _ in range(n_tested))  # placeholder
        console.print(f"\n[dim]Scenario analysis finding: indices (SPY/QQQ/DIA) should outperform sector ETFs[/dim]")


@cli.command("indicator-report")
def indicator_report():
    """Full report on all available JK indicators."""
    from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY

    console.print("\n[bold blue]JK Indicator Registry[/bold blue]\n")

    table = Table(title="Available Indicators", box=box.ROUNDED)
    table.add_column("#", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Signal Column", style="green")
    table.add_column("Key Params", style="dim")

    for i, (name, reg) in enumerate(INDICATOR_REGISTRY.items(), 1):
        params = reg.get("params", {})
        param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        if len(params) > 3:
            param_str += "..."
        table.add_row(
            str(i), name, reg["description"],
            reg["signal_col"], param_str,
        )

    console.print(table)
    console.print(f"\nTotal: {len(INDICATOR_REGISTRY)} indicators")
    console.print("Translated from Jonathan Kornblatt's TradeStation EasyLanguage (2014)\n")


@cli.command("scenario")
@click.option("--indicator", "-i", default=None, help="Test specific indicator (default: all top performers)")
@click.option("--test", "-T", type=click.Choice(["all", "stock-type", "regime", "horizon", "strength", "confluence"]),
              default="all", help="Which scenario test to run")
def scenario(indicator, test):
    """Run scenario analysis — find WHERE each indicator works best."""
    from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
    from technical_analysis.backtest.scenario_tester import (
        run_full_scenario_analysis, test_by_stock_type, test_by_regime,
        test_optimal_horizon, test_signal_strength, test_confluence,
    )

    console.print("\n[bold blue]Scenario Analysis — Where Do Indicators Work?[/bold blue]\n")

    # Default: test the top-performing indicators
    if indicator:
        indicators_to_test = [indicator]
    else:
        indicators_to_test = [
            "z_factor", "ve_rsi", "multimac_fib", "multimac",
            "hybrid_osc", "mfoo", "z_hybrid", "obos", "trend_score",
        ]

    all_results = {}
    for ind_name in indicators_to_test:
        if ind_name not in INDICATOR_REGISTRY:
            console.print(f"[red]Unknown indicator: {ind_name}[/red]")
            continue

        if test == "all":
            all_results[ind_name] = run_full_scenario_analysis(ind_name, verbose=True)
        elif test == "stock-type":
            console.print(f"\n[bold]Stock Type Analysis: {ind_name}[/bold]")
            all_results[ind_name] = {"stock_type": test_by_stock_type(ind_name, verbose=True)}
        elif test == "regime":
            console.print(f"\n[bold]Regime Analysis: {ind_name}[/bold]")
            all_results[ind_name] = {"regime": test_by_regime(ind_name, verbose=True)}
        elif test == "horizon":
            console.print(f"\n[bold]Optimal Horizon: {ind_name}[/bold]")
            all_results[ind_name] = {"horizon": test_optimal_horizon(ind_name, verbose=True)}
        elif test == "strength":
            console.print(f"\n[bold]Signal Strength: {ind_name}[/bold]")
            all_results[ind_name] = {"signal_strength": test_signal_strength(ind_name, verbose=True)}
        elif test == "confluence":
            console.print(f"\n[bold]Confluence Analysis[/bold]")
            all_results["confluence"] = {"confluence": test_confluence(indicators_to_test, verbose=True)}
            break  # Confluence tests all at once

    # Print summary tables
    if test in ("all", "stock-type") and any("stock_type" in r for r in all_results.values()):
        _print_stock_type_summary(all_results)
    if test in ("all", "regime") and any("regime" in r for r in all_results.values()):
        _print_regime_summary(all_results)
    if test in ("all", "horizon") and any("horizon" in r for r in all_results.values()):
        _print_horizon_summary(all_results)
    if test in ("all", "strength") and any("signal_strength" in r for r in all_results.values()):
        _print_strength_summary(all_results)


def _print_stock_type_summary(all_results):
    """Print stock type comparison table."""
    table = Table(title="Best Stock Types per Indicator", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan")
    table.add_column("Best Universe", style="green")
    table.add_column("IC", justify="right")
    table.add_column("Spread", justify="right")
    table.add_column("Worst Universe", style="red")
    table.add_column("IC", justify="right")

    for name, results in all_results.items():
        if "stock_type" not in results:
            continue
        st = results["stock_type"]
        if not st:
            continue
        best = max(st, key=lambda x: abs(x.ic))
        worst = min(st, key=lambda x: abs(x.ic))
        table.add_row(
            name,
            best.scenario, f"{best.ic:+.4f}", f"{best.long_short_spread_ann:+.1%}",
            worst.scenario, f"{worst.ic:+.4f}",
        )

    console.print("\n")
    console.print(table)


def _print_regime_summary(all_results):
    """Print regime comparison table."""
    table = Table(title="Best/Worst Regimes per Indicator", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan")
    table.add_column("Best Regime", style="green")
    table.add_column("IC", justify="right")
    table.add_column("Worst Regime", style="red")
    table.add_column("IC", justify="right")

    for name, results in all_results.items():
        if "regime" not in results:
            continue
        rr = results["regime"]
        best_r = rr.best_regime
        worst_r = rr.worst_regime
        best_ic = rr.regimes.get(best_r, {}).get("ic", 0)
        worst_ic = rr.regimes.get(worst_r, {}).get("ic", 0)
        table.add_row(name, best_r, f"{best_ic:+.4f}", worst_r, f"{worst_ic:+.4f}")

    console.print("\n")
    console.print(table)


def _print_horizon_summary(all_results):
    """Print optimal horizon table."""
    table = Table(title="Optimal Holding Period per Indicator", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan")
    table.add_column("Optimal Days", justify="right", style="green")
    table.add_column("IC at Optimal", justify="right")
    table.add_column("IC @ 1d", justify="right", style="dim")
    table.add_column("IC @ 5d", justify="right")
    table.add_column("IC @ 10d", justify="right")
    table.add_column("IC @ 20d", justify="right")
    table.add_column("IC @ 60d", justify="right", style="dim")

    for name, results in all_results.items():
        if "horizon" not in results:
            continue
        hr = results["horizon"]
        table.add_row(
            name,
            str(hr.optimal_horizon),
            f"{hr.optimal_ic:+.4f}",
            f"{hr.horizons.get(1, {}).get('ic', 0):+.4f}",
            f"{hr.horizons.get(5, {}).get('ic', 0):+.4f}",
            f"{hr.horizons.get(10, {}).get('ic', 0):+.4f}",
            f"{hr.horizons.get(20, {}).get('ic', 0):+.4f}",
            f"{hr.horizons.get(60, {}).get('ic', 0):+.4f}",
        )

    console.print("\n")
    console.print(table)


def _print_strength_summary(all_results):
    """Print signal strength summary."""
    table = Table(title="Signal Strength Analysis", box=box.ROUNDED)
    table.add_column("Indicator", style="cyan")
    table.add_column("Extreme Spread", justify="right", style="green")
    table.add_column("Works at Extremes?", justify="center")
    table.add_column("Bottom Decile", justify="right", style="red")
    table.add_column("Top Decile", justify="right", style="green")

    for name, results in all_results.items():
        if "signal_strength" not in results:
            continue
        ss = results["signal_strength"]
        d_keys = sorted(ss.decile_returns.keys())
        bottom = ss.decile_returns.get(d_keys[0], {}).get("annualized", 0) if d_keys else 0
        top = ss.decile_returns.get(d_keys[-1], {}).get("annualized", 0) if d_keys else 0
        table.add_row(
            name,
            f"{ss.extreme_spread:+.1%}",
            "[green]YES[/green]" if ss.works_at_extremes else "[red]NO[/red]",
            f"{bottom:+.1%}",
            f"{top:+.1%}",
        )

    console.print("\n")
    console.print(table)


if __name__ == "__main__":
    cli()
