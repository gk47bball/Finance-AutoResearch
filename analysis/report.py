"""Markdown report generation for research output."""

import os
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def _equity_curve_chart(equity_curve, benchmark_curve) -> str:
    """Generate equity curve chart and return base64-encoded PNG."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve.index, equity_curve.values, label="Strategy", linewidth=2)
    ax.plot(benchmark_curve.index, benchmark_curve.values, label="Benchmark (SPY)",
            linewidth=2, alpha=0.7)
    ax.set_title("Equity Curve: Strategy vs Benchmark", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("")
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_report(
    cycle_result,
    deep_analyses: list = None,
    strategy_module=None,
) -> str:
    """Generate a full markdown research report."""
    lines = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# FinAutoResearch Report")
    lines.append(f"**Generated:** {now}")
    lines.append("")

    # --- Performance Summary ---
    lines.append("## Performance Summary")
    lines.append("")
    m = cycle_result.backtest.metrics
    if m:
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Sharpe Ratio | {m.get('sharpe_ratio', 0):.3f} |")
        lines.append(f"| Sortino Ratio | {m.get('sortino_ratio', 0):.3f} |")
        lines.append(f"| Annual Return | {m.get('annual_return', 0):.1%} |")
        lines.append(f"| Annual Volatility | {m.get('annual_volatility', 0):.1%} |")
        lines.append(f"| Max Drawdown | {m.get('max_drawdown', 0):.1%} |")
        lines.append(f"| Alpha | {m.get('alpha', 0):.3f} |")
        lines.append(f"| Beta | {m.get('beta', 0):.3f} |")
        lines.append(f"| Information Ratio | {m.get('information_ratio', 0):.3f} |")
        lines.append(f"| Total Return | {m.get('total_return', 0):.1%} |")
        lines.append(f"| Win Rate | {m.get('win_rate', 0):.1%} |")
        lines.append("")
    else:
        lines.append("*No backtest metrics available.*")
        lines.append("")

    # --- Equity Curve ---
    bt = cycle_result.backtest
    if not bt.equity_curve.empty and not bt.benchmark_curve.empty:
        b64 = _equity_curve_chart(bt.equity_curve, bt.benchmark_curve)
        lines.append("## Equity Curve")
        lines.append("")
        lines.append(f"![Equity Curve](data:image/png;base64,{b64})")
        lines.append("")

    # --- Current Portfolio ---
    lines.append("## Current Portfolio (Top Holdings)")
    lines.append("")
    port = cycle_result.portfolio
    if not port.empty:
        lines.append("| Rank | Ticker | Score | Weight | Sector |")
        lines.append("|------|--------|-------|--------|--------|")
        for i, row in port.iterrows():
            lines.append(
                f"| {i+1} | {row.get('ticker', 'N/A')} | "
                f"{row.get('composite_score', 0):.1f} | "
                f"{row.get('weight', 0):.1%} | "
                f"{row.get('sector', 'N/A')} |"
            )
        lines.append("")

        # Sector breakdown
        lines.append("### Sector Breakdown")
        lines.append("")
        if "sector" in port.columns and "weight" in port.columns:
            sector_wt = port.groupby("sector")["weight"].sum().sort_values(ascending=False)
            lines.append("| Sector | Weight |")
            lines.append("|--------|--------|")
            for sector, wt in sector_wt.items():
                lines.append(f"| {sector} | {wt:.1%} |")
            lines.append("")
    else:
        lines.append("*No portfolio constructed.*")
        lines.append("")

    # --- Pipeline Stats ---
    lines.append("## Pipeline Summary")
    lines.append("")
    lines.append(f"- **Universe size:** {cycle_result.universe_size}")
    lines.append(f"- **Passed screens:** {cycle_result.screened_count}")
    lines.append(f"- **Scored stocks:** {len(cycle_result.scored_df) if not cycle_result.scored_df.empty else 0}")
    lines.append(f"- **Portfolio size:** {len(port) if not port.empty else 0}")
    lines.append("")

    # --- Strategy Summary ---
    if strategy_module:
        lines.append("## Strategy Configuration")
        lines.append("")
        factors = getattr(strategy_module, "FACTORS", {})
        lines.append("### Factor Weights")
        lines.append("")
        for fname, fcfg in factors.items():
            lines.append(f"- **{fname.title()}**: {fcfg.get('weight', 0):.0%}")
            for sf, sw in fcfg.get("sub_factors", {}).items():
                lines.append(f"  - {sf}: {sw:.0%}")
        lines.append("")

    # --- Deep Analyses ---
    if deep_analyses:
        lines.append("## Deep-Dive Analyses")
        lines.append("")
        for analysis in deep_analyses:
            lines.append(f"### {analysis.get('ticker', 'Unknown')}")
            lines.append("")
            if analysis.get("summary"):
                lines.append(analysis["summary"])
                lines.append("")
            if analysis.get("competitive_moat"):
                lines.append(f"**Competitive Moat:** {analysis['competitive_moat']}")
                lines.append("")
            if analysis.get("key_risks"):
                lines.append("**Key Risks:**")
                for risk in analysis["key_risks"]:
                    lines.append(f"- {risk}")
                lines.append("")
            if analysis.get("growth_catalysts"):
                lines.append("**Growth Catalysts:**")
                for cat in analysis["growth_catalysts"]:
                    lines.append(f"- {cat}")
                lines.append("")
            if analysis.get("conviction"):
                lines.append(f"**Conviction:** {analysis['conviction']}")
                lines.append("")

    # --- Disclaimer ---
    lines.append("---")
    lines.append("")
    lines.append("*This report is generated by FinAutoResearch for educational and informational purposes only. "
                 "It does not constitute investment advice. Past performance does not guarantee future results. "
                 "Always do your own research before making investment decisions.*")

    return "\n".join(lines)


def save_report(content: str, output_dir: str = "reports") -> str:
    """Save report to file, return path."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"research_{datetime.now().strftime('%Y-%m-%d_%H%M')}.md"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(content)
    return path
