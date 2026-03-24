"""
Housing Alpha CLI
==================
Command-line interface for the housing alpha system.

Usage:
    python -m technical_analysis.bot.housing_alpha.cli signal
    python -m technical_analysis.bot.housing_alpha.cli backtest
    python -m technical_analysis.bot.housing_alpha.cli backtest --tickers XHB,ITB
    python -m technical_analysis.bot.housing_alpha.cli learn -n 30 --time 60
    python -m technical_analysis.bot.housing_alpha.cli trade
    python -m technical_analysis.bot.housing_alpha.cli trade --tickers XHB,ITB
    python -m technical_analysis.bot.housing_alpha.cli status
    python -m technical_analysis.bot.housing_alpha.cli params
    python -m technical_analysis.bot.housing_alpha.cli snapshot
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)


def cmd_signal(args):
    """Compute and display current housing signals."""
    from technical_analysis.bot.housing_alpha.engine import HousingAlphaEngine
    from technical_analysis.bot.housing_alpha.discord_poster import post_housing_signals

    tickers = args.tickers.split(",") if args.tickers else None
    engine = HousingAlphaEngine()
    signals = engine.compute_signals(tickers=tickers)
    engine.print_dashboard()

    if not args.no_discord:
        post_housing_signals(signals, engine)


def cmd_backtest(args):
    """Run housing alpha backtest."""
    from technical_analysis.bot.housing_alpha.backtest import (
        backtest_housing_alpha,
        backtest_housing_multi,
    )
    from technical_analysis.bot.housing_alpha.discord_poster import post_backtest_results

    tickers = args.tickers.split(",") if args.tickers else ["XHB", "ITB"]

    if len(tickers) == 1:
        results = backtest_housing_alpha(
            ticker=tickers[0],
            start=args.start,
            verbose=True,
        )
        # Wrap for discord
        wrapped = {tickers[0]: results, "composite_sharpe": results.get("sharpe_ratio", 0)}
    else:
        wrapped = backtest_housing_multi(
            tickers=tickers,
            start=args.start,
            verbose=True,
        )
        results = wrapped

    if not args.no_discord:
        post_backtest_results(wrapped)


def cmd_learn(args):
    """Run AutoResearch learning loop."""
    from technical_analysis.bot.housing_alpha.self_learner import run_learning_loop

    tickers = args.tickers.split(",") if args.tickers else ["XHB", "ITB"]

    params, sharpe = run_learning_loop(
        max_experiments=args.n,
        time_limit_minutes=args.time,
        model_backend=args.backend,
        model=args.model,
        tickers=tickers,
    )

    print(f"\n  Final best params saved. Composite Sharpe: {sharpe:+.4f}")


def cmd_trade(args):
    """Run monthly rebalance — compute signals and place Alpaca orders."""
    from technical_analysis.bot.housing_alpha.paper_trader import run_monthly_rebalance
    from technical_analysis.bot.housing_alpha.discord_poster import post_housing_signals
    from technical_analysis.bot.housing_alpha.engine import HousingAlphaEngine

    tickers = args.tickers.split(",") if args.tickers else ["XHB", "ITB"]

    # Post signal to Discord first
    engine = HousingAlphaEngine()
    signals = engine.compute_signals(tickers=tickers)
    if not args.no_discord:
        post_housing_signals(signals, engine)

    # Execute trades
    trades = run_monthly_rebalance(tickers=tickers, verbose=True)
    print(f"\n  Executed {len(trades)} trade(s)")


def cmd_status(args):
    """Show current housing portfolio status."""
    from technical_analysis.bot.housing_alpha.paper_trader import get_status

    status = get_status()
    print(f"\n  Housing Alpha Portfolio")
    print(f"  {'─' * 50}")
    print(f"  NAV:             ${status['nav']:,.2f}")
    print(f"  Cash:            ${status['cash']:,.2f}")
    print(f"  Last Rebalance:  {status['last_rebalance'] or 'Never'}")
    print(f"  Total Trades:    {status['trade_count']}")
    print(f"  {'─' * 50}")
    if status["positions"]:
        for ticker, pos in status["positions"].items():
            print(f"  {ticker}: {pos['shares']:.1f} sh @ ${pos['entry_price']:.2f} "
                  f"→ ${pos['value']:,.0f} ({pos['pnl_pct']:+.1f}%) "
                  f"| target {pos['target_pct']:.0%} | {pos['regime']}")
    else:
        print("  (no positions)")
    print()


def cmd_params(args):
    """Show current parameters."""
    from technical_analysis.bot.housing_alpha.engine import load_params, DEFAULT_PARAMS

    params = load_params()
    print("\n  Housing Alpha Parameters:")
    print("  " + "─" * 50)
    for k, v in sorted(params.items()):
        default = DEFAULT_PARAMS.get(k, "?")
        marker = " *" if v != default else ""
        print(f"    {k:<30s} = {v}{marker}")
    print("  " + "─" * 50)
    print("  (* = differs from default)\n")


def cmd_snapshot(args):
    """Show current housing market snapshot."""
    from technical_analysis.bot.housing_alpha.data_fetcher import get_housing_snapshot

    snap = get_housing_snapshot()
    if "error" in snap:
        print(f"  Error: {snap['error']}")
        return

    print("\n  Housing Market Snapshot:")
    print("  " + "─" * 70)
    print(f"  {'Indicator':<30s} {'Value':>12s} {'MoM %':>8s} {'YoY %':>8s}  {'Date':>12s}")
    print("  " + "─" * 70)

    for name, info in sorted(snap.items()):
        val = info.get("value", "N/A")
        mom = info.get("mom_pct", "—")
        yoy = info.get("yoy_pct", "—")
        date = info.get("date", "—")

        mom_str = f"{mom:+.1f}%" if isinstance(mom, (int, float)) else str(mom)
        yoy_str = f"{yoy:+.1f}%" if isinstance(yoy, (int, float)) else str(yoy)

        if isinstance(val, float) and val > 1000:
            val_str = f"{val:,.0f}"
        elif isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)

        print(f"  {name:<30s} {val_str:>12s} {mom_str:>8s} {yoy_str:>8s}  {date:>12s}")

    print("  " + "─" * 70)
    print()


def main():
    parser = argparse.ArgumentParser(description="Housing Alpha Trading System")
    sub = parser.add_subparsers(dest="command")

    # signal
    p_signal = sub.add_parser("signal", help="Compute current housing signals")
    p_signal.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)
    p_signal.add_argument("--no-discord", action="store_true")

    # backtest
    p_bt = sub.add_parser("backtest", help="Run housing alpha backtest")
    p_bt.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)
    p_bt.add_argument("--start", "-s", default="2005-01-01", help="Start date")
    p_bt.add_argument("--no-discord", action="store_true")

    # learn
    p_learn = sub.add_parser("learn", help="Run AutoResearch learning loop")
    p_learn.add_argument("-n", type=int, default=30, help="Max experiments")
    p_learn.add_argument("--time", type=int, default=60, help="Time limit (minutes)")
    p_learn.add_argument("--tickers", "-t", default=None)
    p_learn.add_argument("--backend", default="ollama", choices=["ollama", "anthropic"])
    p_learn.add_argument("--model", default="qwen3:4b")

    # trade
    p_trade = sub.add_parser("trade", help="Run monthly rebalance (Alpaca paper)")
    p_trade.add_argument("--tickers", "-t", default=None)
    p_trade.add_argument("--no-discord", action="store_true")

    # status
    sub.add_parser("status", help="Show housing portfolio status")

    # params
    sub.add_parser("params", help="Show current parameters")

    # snapshot
    sub.add_parser("snapshot", help="Show housing market snapshot")

    args = parser.parse_args()

    if args.command == "signal":
        cmd_signal(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "learn":
        cmd_learn(args)
    elif args.command == "trade":
        cmd_trade(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "params":
        cmd_params(args)
    elif args.command == "snapshot":
        cmd_snapshot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
