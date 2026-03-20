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


if __name__ == "__main__":
    cli()
