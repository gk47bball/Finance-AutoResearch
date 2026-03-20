"""FinAutoResearch CLI — the command-line interface."""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import click
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

# Load .env file if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="FinAutoResearch")
def cli():
    """FinAutoResearch — Karpathy's AutoResearch applied to investing."""
    pass


@cli.command()
@click.option("--no-deep-dive", is_flag=True, help="Skip LLM-powered deep analysis")
@click.option("--output", default="reports/", help="Output directory for reports")
def research(no_deep_dive, output):
    """Run one research cycle with the current strategy."""
    from run import run_research
    run_research(deep_dive=not no_deep_dive, output_dir=output)


@cli.command()
@click.option("--experiments", "-n", default=20, help="Max experiments to run")
@click.option("--time-limit", "-t", default=60, help="Time limit in minutes")
def optimize(experiments, time_limit):
    """Run the AutoResearch optimization loop."""
    from loop import run_loop
    run_loop(max_experiments=experiments, time_limit_minutes=time_limit)


@cli.command()
def backtest():
    """Backtest the current strategy and show metrics."""
    from prepare import load_strategy, run_full_cycle
    console.print("\n[bold blue]Running backtest...[/bold blue]\n")
    strategy = load_strategy()
    result = run_full_cycle(strategy, show_progress=True)

    m = result.backtest.metrics
    if m:
        table = Table(title="Backtest Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.3f}")
        table.add_row("Sortino Ratio", f"{m.get('sortino_ratio', 0):.3f}")
        table.add_row("Annual Return", f"{m.get('annual_return', 0):.1%}")
        table.add_row("Annual Volatility", f"{m.get('annual_volatility', 0):.1%}")
        table.add_row("Max Drawdown", f"{m.get('max_drawdown', 0):.1%}")
        table.add_row("Alpha", f"{m.get('alpha', 0):.3f}")
        table.add_row("Beta", f"{m.get('beta', 0):.3f}")
        table.add_row("Information Ratio", f"{m.get('information_ratio', 0):.3f}")
        table.add_row("Total Return", f"{m.get('total_return', 0):.1%}")
        table.add_row("Win Rate", f"{m.get('win_rate', 0):.1%}")

        console.print(table)
    else:
        console.print("[red]No backtest results available.[/red]")


@cli.command("show-strategy")
def show_strategy():
    """Display the current strategy configuration."""
    from prepare import load_strategy
    strategy = load_strategy()

    console.print("\n[bold blue]Current Strategy Configuration[/bold blue]\n")

    # Universe
    universe = getattr(strategy, "UNIVERSE", {})
    console.print("[bold]Universe:[/bold]")
    console.print(f"  Source: {universe.get('source', 'sp500')}")
    console.print(f"  Min Market Cap: ${universe.get('min_market_cap', 0):,.0f}")
    console.print(f"  Excluded Sectors: {universe.get('exclude_sectors', [])}")
    console.print()

    # Screens
    screens = getattr(strategy, "SCREENS", [])
    console.print("[bold]Screens:[/bold]")
    for s in screens:
        console.print(f"  {s['metric']} {s['op']} {s['value']}")
    console.print()

    # Factors
    factors = getattr(strategy, "FACTORS", {})
    console.print("[bold]Factor Model:[/bold]")
    for name, cfg in factors.items():
        console.print(f"  [cyan]{name.title()}[/cyan] (weight: {cfg.get('weight', 0):.0%})")
        for sf, sw in cfg.get("sub_factors", {}).items():
            console.print(f"    {sf}: {sw:.0%}")
    console.print()

    # Portfolio
    port = getattr(strategy, "PORTFOLIO", {})
    console.print("[bold]Portfolio Construction:[/bold]")
    console.print(f"  Top N: {port.get('top_n', 20)}")
    console.print(f"  Weighting: {port.get('weighting', 'equal')}")
    console.print(f"  Max Sector: {port.get('max_sector_pct', 0.3):.0%}")
    console.print(f"  Rebalance: {port.get('rebalance_frequency', 'quarterly')}")


@cli.command("experiment-log")
def experiment_log():
    """Show the experiment history from optimization runs."""
    import json
    log_path = os.path.join(os.path.dirname(__file__), "experiments", "log.jsonl")
    if not os.path.exists(log_path):
        console.print("[yellow]No experiments logged yet. Run 'optimize' first.[/yellow]")
        return

    table = Table(title="Experiment Log")
    table.add_column("#", style="dim")
    table.add_column("Hypothesis", max_width=50)
    table.add_column("Sharpe", justify="right")
    table.add_column("Result", justify="center")

    with open(log_path) as f:
        for line in f:
            exp = json.loads(line.strip())
            sharpe = exp.get("sharpe")
            sharpe_str = f"{sharpe:.4f}" if sharpe is not None else "N/A"
            kept = exp.get("kept", False)
            result_str = "[green]KEPT[/green]" if kept else "[red]REVERTED[/red]"
            table.add_row(
                str(exp.get("experiment_id", "?")),
                exp.get("hypothesis", "N/A"),
                sharpe_str,
                result_str,
            )

    console.print(table)


@cli.command()
@click.argument("ticker")
def analyze(ticker):
    """Deep-dive analysis on a single stock using Claude."""
    ticker = ticker.upper()
    console.print(f"\n[bold blue]Analyzing {ticker}...[/bold blue]\n")

    try:
        from agent.researcher import ResearchAgent
        from data.fundamentals import get_key_ratios

        ratios = get_key_ratios(ticker)
        if not ratios:
            console.print(f"[red]Could not fetch data for {ticker}[/red]")
            return

        agent = ResearchAgent()
        from prepare import load_strategy
        strategy = load_strategy()
        analysis = agent.analyze_stock(
            ticker,
            ratios,
            getattr(strategy, "DEEP_ANALYSIS", {}),
        )

        console.print(f"[bold]{analysis.get('ticker', ticker)}[/bold]")
        console.print()
        if analysis.get("summary"):
            console.print(analysis["summary"])
        console.print()
        if analysis.get("competitive_moat"):
            console.print(f"[bold]Moat:[/bold] {analysis['competitive_moat']}")
        if analysis.get("key_risks"):
            console.print("\n[bold]Key Risks:[/bold]")
            for r in analysis["key_risks"]:
                console.print(f"  - {r}")
        if analysis.get("growth_catalysts"):
            console.print("\n[bold]Growth Catalysts:[/bold]")
            for c in analysis["growth_catalysts"]:
                console.print(f"  - {c}")
        if analysis.get("conviction"):
            console.print(f"\n[bold]Conviction:[/bold] {analysis['conviction']}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Ensure ANTHROPIC_API_KEY is set in .env[/dim]")


@cli.command()
def macro():
    """Show current macroeconomic environment snapshot."""
    from data.macro import get_macro_snapshot

    snapshot = get_macro_snapshot()
    if not snapshot:
        console.print("[yellow]No macro data available. Set FRED_API_KEY in .env[/yellow]")
        return

    table = Table(title="Macro Environment")
    table.add_column("Indicator", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_column("As Of", style="dim")

    labels = {
        "gdp_growth": "GDP Growth (%)",
        "cpi_yoy": "CPI",
        "unemployment": "Unemployment (%)",
        "fed_funds": "Fed Funds Rate (%)",
        "treasury_10y": "10Y Treasury (%)",
        "treasury_2y": "2Y Treasury (%)",
        "treasury_3m": "3M T-Bill (%)",
        "yield_curve_spread": "Yield Curve (10Y-2Y)",
        "vix": "VIX",
        "baa_spread": "BAA Spread (%)",
    }

    for key, label in labels.items():
        data = snapshot.get(key, {})
        val = data.get("value")
        date = data.get("date", "")
        if val is not None:
            table.add_row(label, f"{val:.2f}", date)
        else:
            table.add_row(label, "N/A", "")

    console.print(table)


if __name__ == "__main__":
    cli()
