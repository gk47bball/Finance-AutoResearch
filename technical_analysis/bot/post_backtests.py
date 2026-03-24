"""
Post Backtest Results to Discord #backtest-results
====================================================
Posts two types of content:
  1. Long-run backtests on all 11 sector ETFs + DIA (10-year Four Pillars performance)
  2. Intraday day-trading simulations on 3 historical dates × 5 tickers

Run:
  python technical_analysis/bot/post_backtests.py
"""

import os
import time
import json
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


CHANNEL_ID = os.environ.get("JK_DISCORD_BACKTEST_CHANNEL", "")
BOT_TOKEN = os.environ.get("JK_DISCORD_BOT_TOKEN", "")

# Sector ETFs + major indices
SECTOR_TICKERS = [
    ("XLK", "Technology"),
    ("XLF", "Financials"),
    ("XLE", "Energy"),
    ("XLV", "Health Care"),
    ("XLI", "Industrials"),
    ("XLP", "Consumer Staples"),
    ("XLY", "Consumer Discretionary"),
    ("XLU", "Utilities"),
    ("XLC", "Communication Svcs"),
    ("XLRE", "Real Estate"),
    ("XLB", "Materials"),
    ("DIA", "Dow Jones (DIA)"),
]

# Intraday simulation config: (date, interval, human label)
# - March 18 = "last week" — BEAR regime, correctly no trades (regime filter working)
# - Feb 18 = "~1 month ago" — CHOP/BULL regime, real trades fired
# - July 14 = "~9 months ago" — BULL regime, real trades fired (1h bars, 730-day yfinance window)
# - January 9 = "~2.5 months ago" (need 1h — past 60-day 5m limit)
# - September 19 = "~6 months ago" (need 1h — within 730-day 1h limit)
INTRADAY_DATES = [
    ("2026-03-18", "5m",  "Last Week (Mar 18) — BEAR Regime"),
    ("2026-02-18", "5m",  "~1 Month Ago (Feb 18) — CHOP/BULL Regime"),
    ("2025-07-14", "1h",  "~9 Months Ago (Jul 14) — BULL Regime"),
]

INTRADAY_TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLV"]

# Backtest period for sector ETFs
BACKTEST_PERIOD = "5y"  # sector ETFs may have shorter history than SPY; 5y safe for all


# ---------------------------------------------------------------------------
# Discord REST helper
# ---------------------------------------------------------------------------

def post_embed(embed: dict, delay: float = 1.0):
    """Post a single embed to the backtest channel."""
    if not CHANNEL_ID or not BOT_TOKEN:
        print("  ⚠️  Missing JK_DISCORD_BACKTEST_CHANNEL or JK_DISCORD_BOT_TOKEN")
        return False

    url = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
    headers = {"Authorization": f"Bot {BOT_TOKEN}", "Content-Type": "application/json"}
    payload = {"embeds": [embed]}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5)
            print(f"  Rate limited — waiting {retry_after:.1f}s...")
            time.sleep(retry_after + 0.5)
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        time.sleep(delay)
        return True
    except Exception as e:
        print(f"  ❌ Post failed: {e}")
        return False


def post_divider(title: str):
    """Post a section divider embed."""
    embed = {
        "title": title,
        "color": 0x2c3e50,
        "timestamp": datetime.utcnow().isoformat(),
    }
    post_embed(embed, delay=0.8)


# ---------------------------------------------------------------------------
# Part 1: Sector ETF long-run backtests
# ---------------------------------------------------------------------------

def run_sector_backtests() -> list:
    """Run backtest_four_pillars on all sector ETFs. Returns list of result dicts."""
    from technical_analysis.bot.backtest_pillars import backtest_four_pillars

    results = []
    for ticker, name in SECTOR_TICKERS:
        print(f"  Backtesting {ticker} ({name})...")
        try:
            r = backtest_four_pillars(ticker=ticker, period=BACKTEST_PERIOD, verbose=False)
            r["ticker"] = ticker
            r["name"] = name
            results.append(r)
        except Exception as e:
            print(f"  ❌ {ticker} failed: {e}")
            results.append({"ticker": ticker, "name": name, "error": str(e)})
        time.sleep(0.3)  # be kind to yfinance
    return results


def build_sector_summary_embed(results: list) -> dict:
    """Build a summary table embed comparing all sector ETF backtests."""
    lines = []
    beats_bm = 0
    total_valid = 0

    for r in results:
        if r.get("error"):
            lines.append(f"❌ **{r['ticker']}** — {r['error'][:50]}")
            continue

        total_valid += 1
        sharpe = r["sharpe_ratio"]
        bm_sharpe = r["benchmark_sharpe"]
        annual = r["annual_return"]
        bm_annual = r["benchmark_return"]
        dd = r["max_drawdown"]
        beats = sharpe >= bm_sharpe
        if beats:
            beats_bm += 1

        beat_emoji = "✅" if beats else "❌"
        lines.append(
            f"{beat_emoji} **{r['ticker']}** ({r['name']}) | "
            f"Sharpe: **{sharpe:.3f}** vs {bm_sharpe:.3f} | "
            f"Ann: {annual:.1%} vs {bm_annual:.1%} | "
            f"DD: {dd:.1%}"
        )

    description = "\n".join(lines)
    description += f"\n\n**{beats_bm}/{total_valid} tickers beat benchmark** | Period: {BACKTEST_PERIOD}"

    color = 0x2ecc71 if beats_bm >= total_valid * 0.6 else 0xe74c3c

    return {
        "title": f"📊 Sector ETF Backtests — Four Pillars Strategy ({BACKTEST_PERIOD})",
        "description": description,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "backtest_four_pillars | daily bars | 5bps commission | stop/trail stops active"},
    }


def build_sector_detail_embed(r: dict) -> dict:
    """Build a detailed embed for a single sector ETF backtest."""
    if r.get("error"):
        return {
            "title": f"❌ {r['ticker']} — Backtest Failed",
            "description": r["error"],
            "color": 0xe74c3c,
        }

    beats = r["sharpe_ratio"] >= r["benchmark_sharpe"]
    color = 0x2ecc71 if beats else 0xe74c3c
    beat_str = "✅ Beats Benchmark" if beats else "❌ Underperforms Benchmark"

    # Exit type breakdown
    exit_types = r.get("exit_types", {})
    exit_str = " | ".join(f"{k}: {v}" for k, v in exit_types.items()) if exit_types else "N/A"

    # Build trade log snippet (last 5 trades)
    trade_log = r.get("trade_log", [])
    completed = [t for t in trade_log if t.get("pnl_pct") is not None][-5:]
    trade_lines = []
    for t in completed:
        date_str = str(t.get("date", ""))[:10]
        pnl = t.get("pnl_pct", 0)
        days = t.get("days_held", 0)
        action = t.get("action", "?")
        sign = "+" if pnl >= 0 else ""
        trade_lines.append(f"  {date_str} {action} {sign}{pnl*100:.1f}% ({days}d)")

    return {
        "title": f"{r['ticker']} — {r['name']} ({BACKTEST_PERIOD})",
        "description": beat_str,
        "color": color,
        "fields": [
            {"name": "Sharpe Ratio", "value": f"**{r['sharpe_ratio']:.4f}** vs {r['benchmark_sharpe']:.4f} (benchmark)", "inline": True},
            {"name": "Annual Return", "value": f"**{r['annual_return']:.1%}** vs {r['benchmark_return']:.1%}", "inline": True},
            {"name": "Total Return", "value": f"**{r['total_return']:.1%}** vs {r['benchmark_total']:.1%}", "inline": True},
            {"name": "Max Drawdown", "value": f"**{r['max_drawdown']:.1%}**", "inline": True},
            {"name": "Volatility", "value": f"**{r['annual_volatility']:.1%}**", "inline": True},
            {"name": "Market Exposure", "value": f"**{r['exposure_pct']:.0%}**", "inline": True},
            {"name": "Trades", "value": f"**{r['n_exits']}** exits | **{r['win_rate']:.0%}** win rate", "inline": True},
            {"name": "Avg Win / Loss", "value": f"+{r['avg_win_pct']:.2f}% / {r['avg_loss_pct']:.2f}%", "inline": True},
            {"name": "Avg Hold", "value": f"**{r['avg_hold_days']:.0f} days**", "inline": True},
            {"name": "Exit Types", "value": exit_str, "inline": False},
            {"name": "Recent Trades (last 5)", "value": "```\n" + "\n".join(trade_lines) + "\n```" if trade_lines else "N/A", "inline": False},
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": f"JK Four Pillars | {BACKTEST_PERIOD} | daily bars"},
    }


def post_sector_backtests():
    """Run all sector ETF backtests and post to Discord."""
    print("\n=== PART 1: Sector ETF Backtests ===")

    post_divider("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📊  SECTOR ETF BACKTESTS  —  Four Pillars Strategy\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    results = run_sector_backtests()

    # Post summary first
    summary_embed = build_sector_summary_embed(results)
    print("  Posting summary embed...")
    post_embed(summary_embed, delay=1.5)

    # Post one detail embed per ticker
    for r in results:
        print(f"  Posting detail for {r['ticker']}...")
        detail_embed = build_sector_detail_embed(r)
        post_embed(detail_embed, delay=1.2)

    print(f"  ✅ Sector backtests posted ({len(results)} tickers)")
    return results


# ---------------------------------------------------------------------------
# Part 2: Intraday day-trading simulations
# ---------------------------------------------------------------------------

def post_intraday_sims():
    """Run intraday simulations for all dates/tickers and post to Discord."""
    from technical_analysis.bot.intraday_sim import run_intraday_batch, format_intraday_embeds

    print("\n=== PART 2: Intraday Day-Trading Simulations ===")

    post_divider("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n📈  INTRADAY DAY-TRADING SIMULATIONS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Explanation embed
    explanation = {
        "title": "ℹ️ How Intraday Simulation Works",
        "description": (
            "The Four Pillars strategy is adapted to intraday bars:\n\n"
            "**Regime Context:** Prior 40 daily bars → morning regime (BULL/CHOP/BEAR)\n"
            "**Entry Signal:** price < EMA20 − 0.15% + RSI < 45 — "
            "only in BULL or CHOP regimes\n"
            "**Exit Signal:** price > EMA20 + 0.1% (mean-reversion complete) OR RSI > 58 OR "
            "−1.5% stop loss OR end-of-day force close\n\n"
            "📅 Three dates tested:\n"
            "• **Mar 18** (last week, 5m) — BEAR regime, bot correctly sits out\n"
            "• **Feb 18** (~1mo ago, 5m) — CHOP/BULL regime, trades fired\n"
            "• **Jul 14** (~9mo ago, 1h) — BULL regime, trades fired\n"
            "📌 Note: yfinance limits intraday data — 5m bars available for ~60 days, "
            "1h bars available for ~730 days."
        ),
        "color": 0x3498db,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": "JK Four Pillars — Intraday Simulation"},
    }
    post_embed(explanation, delay=1.2)

    # Run all simulations
    batch_results = run_intraday_batch(INTRADAY_TICKERS, INTRADAY_DATES)

    # Post results for each date
    for day_batch in batch_results:
        print(f"\n  Posting {day_batch['label']} ({day_batch['date']})...")
        embeds = format_intraday_embeds(day_batch)
        for embed in embeds:
            post_embed(embed, delay=1.0)

    print(f"\n  ✅ Intraday simulations posted ({len(INTRADAY_DATES)} dates × {len(INTRADAY_TICKERS)} tickers)")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print(f"  JK BOT — Backtest Channel Refresh")
    print(f"  Channel: {CHANNEL_ID}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    if not CHANNEL_ID or not BOT_TOKEN:
        print("❌ Missing JK_DISCORD_BACKTEST_CHANNEL or JK_DISCORD_BOT_TOKEN in .env")
        return

    # Part 1: Sector ETF backtests
    post_sector_backtests()

    time.sleep(2)

    # Part 2: Intraday simulations
    post_intraday_sims()

    print(f"\n{'='*60}")
    print(f"  ✅ Backtest channel refresh complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
