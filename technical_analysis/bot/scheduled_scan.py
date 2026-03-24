"""
Scheduled Scanner
==================
Runs the JK Four Pillars scan and sends results to Discord.
Designed to be called by launchd on a schedule.

Also supports a "market hours" daemon mode that runs scans
at pre-market open and post-close, with optional intraday checks.
"""

import sys
import os
from datetime import datetime, time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)


def run_scan(tickers=None):
    """Run a scan and send results to Discord."""
    from technical_analysis.bot.pillars import FourPillarsEngine
    from technical_analysis.backtest.signal_tester import fetch_data
    from technical_analysis.bot.alerts import send_discord_scan

    if tickers is None:
        tickers = ["SPY", "DIA", "QQQ", "IWM", "XLU", "XLV", "XLF", "XLK", "XLE"]

    engine = FourPillarsEngine(period="2y")
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
            print(f"  Error scanning {ticker}: {e}")

    if snapshots:
        send_discord_scan(snapshots, prices)
        print(f"  [{datetime.now():%Y-%m-%d %H:%M}] Scan sent to Discord — {len(snapshots)} tickers")
    else:
        print(f"  [{datetime.now():%Y-%m-%d %H:%M}] No data — scan skipped")


def run_trade_cycle(tickers=None):
    """Run paper trading cycle and send alerts."""
    from technical_analysis.bot.paper_trader import PaperTrader
    from technical_analysis.bot.alerts import send_alerts

    if tickers is None:
        tickers = ["SPY", "DIA"]

    trader = PaperTrader(tickers=tickers)
    signals = trader.run_daily(verbose=False)

    for signal in signals:
        if signal.action != "HOLD":
            send_alerts(signal)
            print(f"  [{datetime.now():%Y-%m-%d %H:%M}] Signal: {signal.action} {signal.ticker}")


def is_market_day():
    """Check if today is a weekday (basic check — doesn't account for holidays)."""
    return datetime.now().weekday() < 5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="JK Bot Scheduled Scanner")
    parser.add_argument("--mode", choices=["scan", "trade", "both"], default="scan")
    parser.add_argument("--tickers", "-t", help="Comma-separated tickers", default=None)
    parser.add_argument("--force", action="store_true", help="Run even on weekends")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else None

    if not args.force and not is_market_day():
        print(f"  [{datetime.now():%Y-%m-%d %H:%M}] Weekend — skipping scan")
        sys.exit(0)

    if args.mode in ("scan", "both"):
        run_scan(tickers)
    if args.mode in ("trade", "both"):
        run_trade_cycle(tickers)
