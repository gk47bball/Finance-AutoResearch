"""
JK Trading Bot — Discord Command Bot
======================================
Listens to commands typed in Discord and responds with live data.

Commands:
  !scan [tickers]   — Run a scan and post results
  !whatif [ticker]  — Show distance from trade trigger (how far to BUY/REDUCE)
  !trade            — Run paper trading cycle
  !status           — Show portfolio status
  !history          — Show recent trade history
  !params           — Show current optimized parameters
  !learn [n]        — Start a quick self-learning run (default 10 experiments)
  !help             — Show command list

Usage:
  python -m technical_analysis.bot.discord_bot

Run in background:
  nohup python -m technical_analysis.bot.discord_bot > state/discord_bot.log 2>&1 &
"""

import os
import sys
import asyncio
import threading
from pathlib import Path
from datetime import datetime

import discord
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

TOKEN = os.environ.get("JK_DISCORD_BOT_TOKEN")
WEBHOOK_URL = os.environ.get("JK_DISCORD_WEBHOOK")
PREFIX = "!"

# Role name that grants bot command access (create this role in your server)
OPERATOR_ROLE = "Bot Operator"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------

def is_authorized(message: discord.Message) -> bool:
    """
    Allow only:
      1. The server owner (always)
      2. Members with the 'Bot Operator' role
    """
    if message.guild is None:
        return False  # No DMs
    if message.author.id == message.guild.owner_id:
        return True
    return any(r.name == OPERATOR_ROLE for r in message.author.roles)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def send_embed(channel, title: str, description: str, color: int = 0x3498db):
    embed = discord.Embed(title=title, description=description, color=color,
                          timestamp=datetime.now())
    embed.set_footer(text="JK Four Pillars Trading Bot")
    await channel.send(embed=embed)


def run_scan(tickers):
    """Run scan synchronously (called in thread).

    Applies cross-sectional laggard ranking (J. Kornblatt 2007 paper) after
    computing all snapshots — rank 1 = most washed-out = highest buy priority.
    """
    from technical_analysis.bot.pillars import FourPillarsEngine
    from technical_analysis.backtest.signal_tester import fetch_data

    engine = FourPillarsEngine(period="2y")
    snapshots, prices = [], {}
    for ticker in tickers:
        try:
            snap = engine.compute(ticker)
            df = fetch_data(ticker, "5d")
            price = float(df["Close"].iloc[-1])
            snapshots.append(snap)
            prices[ticker] = price
        except Exception as e:
            print(f"  Error scanning {ticker}: {e}")

    # Rank cross-sectionally by multimac_rsi (laggard = highest buy priority)
    ranked = FourPillarsEngine.rank_snapshots(snapshots)
    return ranked, prices


def format_scan_lines(snapshots, prices):
    lines = []
    for snap in snapshots:
        price = prices.get(snap.ticker, 0)
        regime_emoji = {"bull": "🟢", "chop": "🟡", "bear": "🔴"}.get(snap.regime, "⚪")
        signal_emoji = {"STRONG_BUY": "🚀", "BUY": "📈", "HOLD": "⏳",
                        "REDUCE": "⚠️", "FLAT": "😴"}.get(snap.signal_label, "")
        rank_str = f"[#{snap.laggard_rank}] " if snap.laggard_rank is not None else ""
        lines.append(
            f"{regime_emoji} {rank_str}**{snap.ticker}** ${price:.2f} — "
            f"**{snap.signal_label}** {signal_emoji} | "
            f"{snap.pillars_confirming}/4 pillars | z={snap.z_hybrid_zscore:+.2f} | mmrsi={snap.multimac_rsi_score:+.1f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

@client.event
async def on_ready():
    print(f"  JK Bot logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if not message.content.startswith(PREFIX):
        return

    parts = message.content[len(PREFIX):].strip().split()
    cmd = parts[0].lower() if parts else ""
    args = parts[1:]

    # Authorization check
    if not is_authorized(message):
        await message.channel.send(
            f"🔒 {message.author.mention} You don't have permission to run bot commands. "
            f"Ask the server owner to give you the **{OPERATOR_ROLE}** role."
        )
        return

    # -----------------------------------------------------------------------
    if cmd == "help":
        desc = (
            "`!scan [tickers]` — Scan tickers for Four Pillars signals\n"
            "`!whatif [ticker]` — Show distance from trade trigger (how far to BUY/REDUCE)\n"
            "`!trade` — Run paper trading cycle\n"
            "`!status` — Show portfolio NAV and positions\n"
            "`!history` — Show last 10 trades\n"
            "`!params` — Show current optimized parameters\n"
            "`!learn [n]` — Run self-learning loop (default 10 experiments)\n"
            "`!housing` — 🏠 Housing Alpha signal dashboard\n"
            "`!housing_status` — 🏠 Housing portfolio status\n"
            "`!housing_trade` — 🏠 Execute housing monthly rebalance\n"
            "`!help` — Show this message\n\n"
            "**Default tickers:** SPY, DIA, QQQ, IWM, XLU, XLV, XLF, XLK, XLE\n"
            "**Custom:** `!scan AAPL,TSLA,NVDA` | `!whatif QQQ`"
        )
        await send_embed(message.channel, "📖 JK Bot Commands", desc, 0x3498db)

    # -----------------------------------------------------------------------
    elif cmd == "scan":
        tickers = args[0].upper().split(",") if args else \
            ["SPY", "DIA", "QQQ", "IWM", "XLU", "XLV", "XLF", "XLK", "XLE"]
        await message.channel.send(f"⏳ Scanning {len(tickers)} tickers...")

        loop = asyncio.get_event_loop()
        snapshots, prices = await loop.run_in_executor(None, run_scan, tickers)

        if snapshots:
            desc = format_scan_lines(snapshots, prices)
            await send_embed(message.channel, "📡 Four Pillars Scan", desc, 0x3498db)
        else:
            await message.channel.send("❌ Scan failed — no data returned.")

    # -----------------------------------------------------------------------
    elif cmd == "trade":
        tickers = args[0].upper().split(",") if args else ["SPY", "DIA"]
        await message.channel.send(f"⏳ Running paper trading cycle for {tickers}...")

        def _trade():
            from technical_analysis.bot.paper_trader import PaperTrader
            from technical_analysis.bot.alerts import send_alerts
            trader = PaperTrader(tickers=tickers)
            signals = trader.run_daily(verbose=False)
            results = []
            for sig in signals:
                if sig.action != "HOLD":
                    send_alerts(sig)
                results.append(f"**{sig.ticker}**: {sig.action} @ {sig.position_pct:.0%} — {sig.reason[:80]}")
            return results

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, _trade)
        desc = "\n".join(results) if results else "No actionable signals — all HOLD."
        await send_embed(message.channel, "💼 Paper Trading Cycle", desc, 0x2ecc71)

    # -----------------------------------------------------------------------
    elif cmd == "status":
        def _status():
            from technical_analysis.bot.paper_trader import PaperTrader
            trader = PaperTrader()
            trader._update_positions()
            state = trader.state
            lines = [
                f"**Cash:** ${state.cash:,.2f}",
                f"**Positions:** {len(state.positions)}",
            ]
            for ticker, pos in state.positions.items():
                pnl = (pos.current_price - pos.avg_price) / pos.avg_price
                lines.append(f"  • {ticker}: {pos.shares:.1f} shares @ ${pos.current_price:.2f} ({pnl:+.1%})")
            if state.daily_nav:
                nav = state.daily_nav[-1]["nav"]
                lines.append(f"**NAV:** ${nav:,.2f}")
            return "\n".join(lines)

        loop = asyncio.get_event_loop()
        desc = await loop.run_in_executor(None, _status)
        await send_embed(message.channel, "📊 Portfolio Status", desc, 0x9b59b6)

    # -----------------------------------------------------------------------
    elif cmd == "history":
        def _history():
            from technical_analysis.bot.paper_trader import PaperTrader
            trader = PaperTrader()
            trades = trader.state.trade_log[-10:]
            if not trades:
                return "No trades yet."
            lines = []
            for t in trades:
                date = t["timestamp"][:10]
                pnl = f" ({t['pnl']:+.1%})" if t.get("pnl") else ""
                lines.append(f"`{date}` **{t['ticker']}** {t['action']}{pnl} — {t['reason'][:60]}")
            exits = [t for t in trader.state.trade_log if t.get("pnl") is not None]
            if exits:
                wins = len([t for t in exits if t["pnl"] > 0])
                lines.append(f"\n**Win Rate:** {wins}/{len(exits)} ({wins/len(exits):.0%})")
            return "\n".join(lines)

        loop = asyncio.get_event_loop()
        desc = await loop.run_in_executor(None, _history)
        await send_embed(message.channel, "📜 Trade History", desc, 0xf39c12)

    # -----------------------------------------------------------------------
    elif cmd == "params":
        def _params():
            from technical_analysis.bot.self_learner import load_best_params, load_experiment_history, DEFAULT_PARAMS
            params = load_best_params()
            history = load_experiment_history()
            lines = []
            for key, val in sorted(params.items()):
                default = DEFAULT_PARAMS.get(key)
                changed = " ✨" if val != default else ""
                lines.append(f"`{key}` = **{val}**{changed}")
            if history:
                kept = [e for e in history if e.get("kept")]
                lines.append(f"\n**Experiments:** {len(history)} total, {len(kept)} improvements")
                if kept:
                    best = max(e.get("new_sharpe", 0) for e in kept)
                    lines.append(f"**Best Sharpe:** {best:.4f}")
            return "\n".join(lines)

        loop = asyncio.get_event_loop()
        desc = await loop.run_in_executor(None, _params)
        await send_embed(message.channel, "⚙️ Bot Parameters", desc, 0x1abc9c)

    # -----------------------------------------------------------------------
    elif cmd == "whatif":
        ticker = args[0].upper() if args else "SPY"
        await message.channel.send(f"⏳ Analyzing {ticker}...")

        def _whatif(tkr):
            from technical_analysis.bot.pillars import FourPillarsEngine
            from technical_analysis.bot.self_learner import load_best_params, DEFAULT_PARAMS
            from technical_analysis.backtest.signal_tester import fetch_data

            params = load_best_params()
            oversold     = params.get("OVERSOLD", DEFAULT_PARAMS["OVERSOLD"])
            deep_oversold= params.get("DEEP_OVERSOLD", DEFAULT_PARAMS["DEEP_OVERSOLD"])
            overbought   = params.get("OVERBOUGHT", DEFAULT_PARAMS["OVERBOUGHT"])
            bull_thr     = params.get("BULL_THRESHOLD", DEFAULT_PARAMS["BULL_THRESHOLD"])
            bear_thr     = params.get("BEAR_THRESHOLD", DEFAULT_PARAMS["BEAR_THRESHOLD"])

            engine = FourPillarsEngine(period="2y")
            snap = engine.compute(tkr)
            df = fetch_data(tkr, "5d")
            price = float(df["Close"].iloc[-1])

            z = snap.z_hybrid_zscore
            regime = snap.regime
            timing = snap.timing_signal
            ts = snap.trend_score_raw

            regime_emoji = {"bull": "🟢", "chop": "🟡", "bear": "🔴"}.get(regime, "⚪")

            # --- Distance analysis ---
            lines = [f"**{tkr}** @ ${price:.2f}  {regime_emoji} **{regime.upper()}** regime"]
            lines.append(f"")
            lines.append(f"**Four Pillars Snapshot**")
            lines.append(f"  P1 Regime — trend_score={ts:+.1f} (BULL≥{bull_thr}, BEAR≤{bear_thr}) → **{regime.upper()}**")
            lines.append(f"  P2 Timing — z-score={z:+.3f} (OVERSOLD≤{oversold}, DEEP≤{deep_oversold}, OB≥{overbought}) → **{timing.upper()}**")
            lines.append(f"  P3 Momentum — {'✅ confirming' if snap.momentum_confirming else '❌ not confirming'}")
            lines.append(f"  P4 Volume   — {'✅ confirming' if snap.volume_confirming else '❌ not confirming'}")
            lines.append(f"  **{snap.pillars_confirming}/4 pillars confirming** → Signal: **{snap.signal_label}**")
            lines.append(f"")

            # --- What's needed for a BUY ---
            lines.append(f"**Distance to BUY trigger:**")

            if snap.signal_label in ("BUY", "STRONG_BUY"):
                lines.append(f"  ✅ Already at BUY — z={z:+.3f} ≤ {oversold}")
            else:
                gap = z - oversold
                if regime == "bear" and z > deep_oversold:
                    gap = z - deep_oversold
                    lines.append(f"  🔴 BEAR regime — needs z={deep_oversold} (DEEP OVERSOLD) + 1+ confirming pillars")
                    lines.append(f"  Currently {gap:+.2f}z-score units above deep-oversold threshold")
                elif z > oversold:
                    lines.append(f"  z-score needs to fall {gap:+.2f} more units (to ≤{oversold})")
                else:
                    # z is below oversold but something else is blocking
                    lines.append(f"  z={z:+.3f} already below OVERSOLD threshold ({oversold})")
                    if not snap.momentum_confirming and not snap.volume_confirming:
                        lines.append(f"  ❌ Blocked: need momentum or volume to confirm")
                    elif regime == "bear":
                        lines.append(f"  ❌ Blocked: BEAR regime suppresses buys")

            # --- What's needed to REDUCE ---
            lines.append(f"")
            lines.append(f"**Distance to REDUCE trigger:**")
            if z >= overbought:
                lines.append(f"  ⚠️ Already OVERBOUGHT (z={z:+.2f} ≥ {overbought})")
            else:
                ob_gap = overbought - z
                lines.append(f"  z-score needs to rise {ob_gap:.2f} more units (to ≥{overbought})")

            return "\n".join(lines)

        loop = asyncio.get_event_loop()
        try:
            desc = await loop.run_in_executor(None, _whatif, ticker)
            await send_embed(message.channel, f"🔍 What-If Analysis: {ticker}", desc, 0xf39c12)
        except Exception as e:
            await message.channel.send(f"❌ Error analyzing {ticker}: {e}")

    # -----------------------------------------------------------------------
    elif cmd == "learn":
        n = int(args[0]) if args and args[0].isdigit() else 10
        await message.channel.send(f"🧠 Starting self-learning loop ({n} experiments)... I'll update you when done.")

        def _learn():
            from technical_analysis.bot.self_learner import run_learning_loop
            run_learning_loop(
                max_experiments=n,
                time_limit_minutes=30,
                model_backend="haiku",
                ticker="SPY",
                period="10y",
                verbose=False,
            )

        # Run in background thread so bot stays responsive
        thread = threading.Thread(target=_learn, daemon=True)
        thread.start()

        async def _notify_done():
            await asyncio.get_event_loop().run_in_executor(None, thread.join)
            from technical_analysis.bot.self_learner import load_best_params, load_experiment_history
            history = load_experiment_history()
            kept = [e for e in history if e.get("kept")]
            best = max((e.get("new_sharpe", 0) for e in kept), default=0)
            await send_embed(
                message.channel,
                "🧠 Learning Complete",
                f"Ran {n} experiments.\n**Best Sharpe:** {best:.4f}\n**Total improvements:** {len(kept)}",
                0x2ecc71,
            )

        asyncio.ensure_future(_notify_done())

    # -----------------------------------------------------------------------
    elif cmd == "housing":
        await message.channel.send("🏠 Computing housing signals...")

        def _housing_signal():
            from technical_analysis.bot.housing_alpha.engine import HousingAlphaEngine
            engine = HousingAlphaEngine()
            return engine.compute_signals()

        loop = asyncio.get_event_loop()
        try:
            signals = await loop.run_in_executor(None, _housing_signal)
            if not signals:
                await message.channel.send("No housing data available.")
            else:
                s = signals[0]
                regime_emoji = {"HOUSING_BULL": "🏠📈", "HOUSING_NEUTRAL": "🏠➡️", "HOUSING_BEAR": "🏠📉"}
                lines = [
                    f"**Regime:** {regime_emoji.get(s.regime, '')} {s.regime}",
                    f"**Composite Z:** {s.composite_z:+.2f}",
                    "",
                    f"Activity:      `{s.activity_z:+.2f}`",
                    f"Affordability: `{s.affordability_z:+.2f}`",
                    f"Supply/Demand: `{s.supply_demand_z:+.2f}`",
                    f"Price Mom:     `{s.price_momentum_z:+.2f}`",
                    f"Rate Regime:   `{s.rate_regime_z:+.2f}`",
                    "",
                ]
                for sig in signals:
                    lines.append(f"**{sig.ticker}** → {sig.target_position:.0%} allocation")
                if s.rate_override_active:
                    lines.append("\n⚠️ Rate override active — positions reduced")
                color = 0x2ECC71 if "BULL" in s.regime else (0xE74C3C if "BEAR" in s.regime else 0xF39C12)
                await send_embed(message.channel, f"🏠 Housing Alpha — {s.date}", "\n".join(lines), color)
        except Exception as e:
            await message.channel.send(f"❌ Housing signal error: {e}")

    # -----------------------------------------------------------------------
    elif cmd == "housing_status":
        def _housing_status():
            from technical_analysis.bot.housing_alpha.paper_trader import get_status
            return get_status()

        loop = asyncio.get_event_loop()
        try:
            status = await loop.run_in_executor(None, _housing_status)
            lines = [
                f"**NAV:** ${status['nav']:,.2f}",
                f"**Cash:** ${status['cash']:,.2f}",
                f"**Last Rebalance:** {status['last_rebalance'] or 'Never'}",
                "",
            ]
            for ticker, pos in status["positions"].items():
                lines.append(f"**{ticker}:** {pos['shares']:.1f} sh @ ${pos['entry_price']:.2f} "
                            f"→ ${pos['value']:,.0f} ({pos['pnl_pct']:+.1f}%) | {pos['regime']}")
            if not status["positions"]:
                lines.append("_(no positions)_")
            await send_embed(message.channel, "🏠 Housing Alpha Portfolio", "\n".join(lines), 0x3498DB)
        except Exception as e:
            await message.channel.send(f"❌ Housing status error: {e}")

    # -----------------------------------------------------------------------
    elif cmd == "housing_trade":
        await message.channel.send("🏠 Running housing alpha rebalance...")

        def _housing_trade():
            from technical_analysis.bot.housing_alpha.paper_trader import run_monthly_rebalance
            return run_monthly_rebalance(verbose=False)

        loop = asyncio.get_event_loop()
        try:
            trades = await loop.run_in_executor(None, _housing_trade)
            if trades:
                lines = []
                for t in trades:
                    lines.append(f"**{t['action']}** {t['ticker']}: {t['shares']:.1f} sh @ ${t['price']:.2f} "
                                f"(${t['notional']:,.0f})")
                await send_embed(message.channel, "🏠 Housing Alpha — Trades Executed",
                               "\n".join(lines), 0x2ECC71)
            else:
                await send_embed(message.channel, "🏠 Housing Alpha — No Rebalance Needed",
                               "Current positions are within threshold.", 0xF39C12)
        except Exception as e:
            await message.channel.send(f"❌ Housing trade error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not TOKEN:
        print("ERROR: JK_DISCORD_BOT_TOKEN not set in .env")
        print("Get your token from: Discord Developer Portal → Your App → Bot → Token")
        sys.exit(1)
    client.run(TOKEN)
