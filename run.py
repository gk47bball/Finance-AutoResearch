"""Execute one research cycle with the current strategy."""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from prepare import load_strategy, run_full_cycle
from analysis.report import generate_report, save_report
from rich.console import Console

console = Console()


def run_research(deep_dive: bool = True, output_dir: str = "reports"):
    """Run one complete research cycle and generate a report."""
    console.print("\n[bold blue]FinAutoResearch[/bold blue] — Running Research Cycle\n")

    # Load strategy
    strategy = load_strategy()
    console.print("[dim]Strategy loaded.[/dim]")

    # Run full cycle
    console.print("[yellow]Running pipeline: Universe → Screen → Score → Backtest...[/yellow]\n")
    result = run_full_cycle(strategy, show_progress=True)

    # Display summary
    m = result.backtest.metrics
    if m:
        console.print(f"\n[bold green]Backtest Results:[/bold green]")
        console.print(f"  Sharpe Ratio:     {m.get('sharpe_ratio', 0):>8.3f}")
        console.print(f"  Sortino Ratio:    {m.get('sortino_ratio', 0):>8.3f}")
        console.print(f"  Annual Return:    {m.get('annual_return', 0):>8.1%}")
        console.print(f"  Max Drawdown:     {m.get('max_drawdown', 0):>8.1%}")
        console.print(f"  Alpha:            {m.get('alpha', 0):>8.3f}")
        console.print(f"  Total Return:     {m.get('total_return', 0):>8.1%}")

    # Show top holdings
    if not result.portfolio.empty:
        console.print(f"\n[bold]Top Holdings:[/bold]")
        for i, row in result.portfolio.head(10).iterrows():
            console.print(
                f"  {i+1:>2}. {row.get('ticker', '???'):<6} "
                f"Score: {row.get('composite_score', 0):>5.1f}  "
                f"Sector: {row.get('sector', 'N/A')}"
            )

    # Deep analysis
    deep_analyses = []
    if deep_dive and not result.portfolio.empty:
        n_deep = getattr(strategy, "DEEP_ANALYSIS", {}).get("top_n_for_deep_dive", 5)
        top_tickers = result.portfolio.head(n_deep)["ticker"].tolist()
        console.print(f"\n[yellow]Running deep-dive analysis on {len(top_tickers)} stocks...[/yellow]")

        try:
            from agent.researcher import ResearchAgent
            agent = ResearchAgent()
            for ticker in top_tickers:
                console.print(f"  Analyzing {ticker}...")
                analysis = agent.analyze_stock(
                    ticker,
                    result.backtest.metrics,
                    getattr(strategy, "DEEP_ANALYSIS", {}),
                )
                deep_analyses.append(analysis)
        except Exception as e:
            console.print(f"[red]Deep analysis unavailable: {e}[/red]")
            console.print("[dim]Set ANTHROPIC_API_KEY in .env for LLM-powered analysis.[/dim]")

    # Generate report
    report = generate_report(result, deep_analyses, strategy)
    path = save_report(report, output_dir)
    console.print(f"\n[bold green]Report saved:[/bold green] {path}")

    return result


if __name__ == "__main__":
    run_research(deep_dive=False)
