"""
JK Trading Bot CLI
===================
Command-line interface for the Four Pillars trading bot.

Usage:
    python -m technical_analysis.bot.cli scan              # Scan all tickers for signals
    python -m technical_analysis.bot.cli trade             # Run paper trading cycle
    python -m technical_analysis.bot.cli backtest          # Backtest Four Pillars strategy
    python -m technical_analysis.bot.cli status            # Show portfolio status
    python -m technical_analysis.bot.cli history           # Show trade history
"""

import argparse
import sys


def cmd_scan(args):
    """Scan tickers and show Four Pillars readings."""
    from technical_analysis.bot.pillars import FourPillarsEngine
    from technical_analysis.backtest.signal_tester import fetch_data
    from technical_analysis.bot.alerts import send_discord_scan

    engine = FourPillarsEngine(period="2y")
    # Default universe: 4 broad market ETFs + all 11 SPDR sectors.
    # The paper's mean-reversion finding is stronger across sector ETFs
    # (genuine dispersion) than across broad indices alone.
    DEFAULT_SCAN_TICKERS = [
        "SPY", "QQQ", "DIA", "IWM",          # broad market
        "XLK", "XLF", "XLV", "XLE", "XLI",   # sectors: tech, finance, health, energy, industrial
        "XLY", "XLP", "XLU", "XLB", "XLRE",  # sectors: discretionary, staples, utilities, materials, real estate
        "XLC",                                  # sectors: communication
    ]
    tickers = args.tickers.split(",") if args.tickers else DEFAULT_SCAN_TICKERS

    snapshots = []
    prices = {}
    for ticker in tickers:
        try:
            snap = engine.compute(ticker)
            df = fetch_data(ticker, "5d")
            price = float(df["Close"].iloc[-1])
            snapshots.append(snap)
            prices[ticker] = price
        except Exception as e:
            print(f"  {ticker:<7} {'ERROR':>8} — {e}")

    # Apply cross-sectional laggard ranking (J. Kornblatt 2007)
    ranked = FourPillarsEngine.rank_snapshots(snapshots)

    print(f"\n  JK Four Pillars Scanner")
    print(f"  {'='*78}")
    print(f"  {'Rank':<5} {'Ticker':<7} {'Price':>8} {'Regime':<6} {'Timing':<14} {'Mom':>5} {'Vol':>5} {'P#':>3} {'Signal':<12} {'Pos':>5} {'MMRSI':>7}")
    print(f"  {'-'*78}")

    for snap in ranked:
        price = prices.get(snap.ticker, 0)
        mom = "Y" if snap.momentum_confirming else "-"
        vol = "Y" if snap.volume_confirming else "-"
        rank_str = f"#{snap.laggard_rank}" if snap.laggard_rank else ""
        print(f"  {rank_str:<5} {snap.ticker:<7} {price:>8.2f} {snap.regime:<6} {snap.timing_signal:<14} {mom:>5} {vol:>5} {snap.pillars_confirming:>3} {snap.signal_label:<12} {snap.position_pct:>4.0%} {snap.multimac_rsi_score:>7.2f}")

    print(f"  {'='*78}")
    print(f"  Rank = laggard rank by multimac_rsi (1=most washed-out, highest buy priority)")

    # Send to Discord if configured
    if ranked:
        send_discord_scan(ranked, prices)


def cmd_trade(args):
    """Run paper trading cycle."""
    from technical_analysis.bot.paper_trader import PaperTrader
    from technical_analysis.bot.alerts import send_alerts

    tickers = args.tickers.split(",") if args.tickers else ["SPY", "DIA"]
    trader = PaperTrader(
        initial_capital=args.capital,
        tickers=tickers,
    )

    print(f"\n  JK Four Pillars — Paper Trading")
    print(f"  Tickers: {tickers}")

    signals = trader.run_daily(verbose=True)

    for signal in signals:
        if signal.action != "HOLD":
            send_alerts(signal)


def cmd_backtest(args):
    """Backtest the Four Pillars strategy."""
    from technical_analysis.bot.backtest_pillars import backtest_four_pillars

    tickers = args.tickers.split(",") if args.tickers else ["SPY"]

    for ticker in tickers:
        results = backtest_four_pillars(
            ticker=ticker,
            period=args.period,
            verbose=True,
        )

    if len(tickers) > 1:
        print(f"\n  Use --tickers SPY to see detailed results for a single ticker.")


def cmd_status(args):
    """Show portfolio status."""
    from technical_analysis.bot.paper_trader import PaperTrader

    trader = PaperTrader()
    trader._update_positions()
    trader._print_summary()

    if trader.state.daily_nav:
        navs = trader.state.daily_nav
        print(f"\n  NAV History (last 10):")
        for entry in navs[-10:]:
            print(f"    {entry['date']}: ${entry['nav']:,.2f}")


def cmd_history(args):
    """Show trade history."""
    from technical_analysis.bot.paper_trader import PaperTrader

    trader = PaperTrader()
    trades = trader.state.trade_log

    if not trades:
        print("  No trades yet.")
        return

    print(f"\n  Trade History ({len(trades)} trades)")
    print(f"  {'='*80}")
    print(f"  {'Date':<12} {'Ticker':<6} {'Action':<10} {'Shares':>8} {'Price':>8} {'P&L':>10} {'Reason'}")
    print(f"  {'-'*80}")

    for t in trades[-20:]:  # last 20
        date = t["timestamp"][:10]
        pnl = f"${t.get('pnl', 0):+.2f}" if t.get("pnl") else ""
        reason = t["reason"][:40]
        print(f"  {date:<12} {t['ticker']:<6} {t['action']:<10} {t['shares']:>8.1f} {t['price']:>8.2f} {pnl:>10} {reason}")

    # Summary stats
    exits = [t for t in trades if t.get("pnl") is not None]
    if exits:
        total_pnl = sum(t["pnl"] for t in exits)
        wins = len([t for t in exits if t["pnl"] > 0])
        print(f"\n  Total P&L: ${total_pnl:+,.2f} | Win Rate: {wins}/{len(exits)} ({wins/len(exits):.0%})")


def cmd_validate(args):
    """
    Validate strategy robustness with two independent tests:

    1. Walk-forward: split history into train/test windows. A robust strategy
       degrades <15% Sharpe out-of-sample. >35% degradation = OVERFIT.

    2. Regime-only baseline: compare full Four Pillars against P1-only (regime
       filter, no P2/P3/P4 timing). Quantifies how much value the timing pillars
       actually add. If the gap is small (<0.10 Sharpe), the timing system is
       adding complexity without proportional return.

    Run this before every major AutoResearch push, and after any parameter
    change you plan to trade with real capital.
    """
    from technical_analysis.bot.backtest_pillars import (
        backtest_four_pillars, backtest_regime_only, walk_forward_validate
    )
    from technical_analysis.bot.self_learner import load_best_params

    ticker = args.ticker
    period = args.period
    params = load_best_params()

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  STRATEGY VALIDATION SUITE — {ticker:<10} ({period})      ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # ── Test 1: Walk-forward ────────────────────────────────────────────────
    print(f"\n  ── Test 1: Walk-Forward Validation (train {args.train_frac:.0%} / test {1-args.train_frac:.0%}) ──")
    wf = walk_forward_validate(
        ticker=ticker, period=period,
        train_frac=args.train_frac, params=params, verbose=True
    )

    # ── Test 2: Regime-only baseline comparison ─────────────────────────────
    print(f"\n  ── Test 2: Regime-Only Baseline (P1 only, no timing) ──")
    if ticker == "MULTI":
        # Run on individual tickers for regime-only (MULTI not supported here)
        for t in ["SPY", "QQQ", "DIA", "IWM"]:
            backtest_regime_only(ticker=t, period=period, params=params, verbose=True)
        full_result = None
    else:
        regime_result = backtest_regime_only(ticker=ticker, period=period, params=params, verbose=True)
        print(f"\n  ── Full Four Pillars (for comparison) ──")
        full_result = backtest_four_pillars(ticker=ticker, period=period, params=params, verbose=True)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n  ── Validation Summary ──")
    verdict_emoji = {"ROBUST": "✅", "MODERATE": "⚠️", "OVERFIT": "❌"}.get(wf["verdict"], "?")
    print(f"  Walk-forward verdict:  {verdict_emoji} {wf['verdict']}")
    print(f"    Train Sharpe: {wf['train_sharpe']:.4f} | Test Sharpe: {wf['test_sharpe']:.4f} | Degradation: {wf['degradation_pct']:+.1f}%")

    if full_result and ticker != "MULTI":
        timing_add = full_result["sharpe_ratio"] - regime_result["sharpe_ratio"]
        timing_emoji = "✅" if timing_add > 0.10 else ("⚠️" if timing_add > 0 else "❌")
        print(f"\n  Timing pillars value-add: {timing_emoji} {timing_add:+.4f} Sharpe")
        print(f"    Regime-only: {regime_result['sharpe_ratio']:.4f} → Full 4P: {full_result['sharpe_ratio']:.4f}")
        if timing_add < 0.05:
            print(f"    ⚠️  Timing adds minimal value. P1 regime filter doing most of the work.")
        elif timing_add > 0.15:
            print(f"    ✅ Strong timing value-add. P2/P3/P4 are meaningfully contributing.")

    print(f"\n  Recommendation:")
    if wf["verdict"] == "ROBUST":
        print(f"  ✅ Strategy is robust. Safe to continue AutoResearch optimization.")
    elif wf["verdict"] == "MODERATE":
        print(f"  ⚠️  Moderate overfitting detected. Proceed carefully with optimization.")
        print(f"     Ensure new params also improve test window, not just train.")
    else:
        print(f"  ❌ Significant overfitting. STOP further AutoResearch until investigated.")
        print(f"     The 10-year Sharpe is misleading. Focus on the test window Sharpe ({wf['test_sharpe']:.4f}).")


def cmd_learn(args):
    """Run self-learning loop to improve bot parameters."""
    from technical_analysis.bot.self_learner import run_learning_loop

    run_learning_loop(
        max_experiments=args.n,
        time_limit_minutes=args.time,
        model_backend=args.model,
        ollama_model=args.ollama_model,
        ticker=args.ticker,
        period=args.period,
        verbose=True,
    )


def cmd_params(args):
    """Show current best parameters."""
    from technical_analysis.bot.self_learner import load_best_params, load_experiment_history, DEFAULT_PARAMS

    params = load_best_params()
    history = load_experiment_history()

    print(f"\n  Current Bot Parameters")
    print(f"  {'='*50}")
    for key, val in sorted(params.items()):
        default = DEFAULT_PARAMS.get(key)
        changed = " *" if val != default else ""
        print(f"    {key:<25} {val}{changed}")

    if history:
        kept = [e for e in history if e.get("kept")]
        print(f"\n  Learning History: {len(history)} experiments, {len(kept)} kept")
        if kept:
            print(f"  Last improvement: {kept[-1].get('hypothesis', '?')[:70]}")
            print(f"  Best Sharpe achieved: {max(e.get('new_sharpe', 0) for e in kept):.4f}")


def main():
    parser = argparse.ArgumentParser(description="JK Four Pillars Trading Bot")
    sub = parser.add_subparsers(dest="command")

    # scan
    p_scan = sub.add_parser("scan", help="Scan tickers for signals")
    p_scan.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)

    # trade
    p_trade = sub.add_parser("trade", help="Run paper trading cycle")
    p_trade.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)
    p_trade.add_argument("--capital", type=float, default=100_000, help="Initial capital")

    # backtest
    p_bt = sub.add_parser("backtest", help="Backtest Four Pillars strategy")
    p_bt.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)
    p_bt.add_argument("--period", "-p", default="10y", help="Data period")

    # status
    sub.add_parser("status", help="Show portfolio status")

    # history
    sub.add_parser("history", help="Show trade history")

    # learn (self-improvement loop)
    p_learn = sub.add_parser("learn", help="Run self-learning loop")
    p_learn.add_argument("--model", "-m", default="haiku", choices=["haiku", "sonnet", "ollama"],
                         help="LLM backend")
    p_learn.add_argument("--ollama-model", default="qwen2.5-coder:7b", help="Ollama model name")
    p_learn.add_argument("-n", type=int, default=30, help="Max experiments")
    p_learn.add_argument("--time", type=int, default=120, help="Time limit in minutes")
    p_learn.add_argument("--ticker", default="SPY", help="Ticker to optimize on")
    p_learn.add_argument("--period", "-p", default="10y", help="Data period")

    # params
    sub.add_parser("params", help="Show current best parameters")

    # validate
    p_val = sub.add_parser("validate", help="Walk-forward + regime-only validation suite")
    p_val.add_argument("--ticker", "-t", default="MULTI",
                       help="Ticker or MULTI (default: MULTI)")
    p_val.add_argument("--period", "-p", default="10y", help="History period")
    p_val.add_argument("--train-frac", type=float, default=0.65,
                       help="Fraction of data used as training window (default 0.65)")

    args = parser.parse_args()

    commands = {
        "scan": cmd_scan,
        "trade": cmd_trade,
        "backtest": cmd_backtest,
        "status": cmd_status,
        "history": cmd_history,
        "learn": cmd_learn,
        "params": cmd_params,
        "validate": cmd_validate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
