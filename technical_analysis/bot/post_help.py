#!/usr/bin/env python3
"""
Post the JK Trading Bot help/guide to Discord #bot-guide.
============================================================
Posts a comprehensive, beginner-friendly guide to the dedicated
#bot-guide channel explaining EVERY capability of the bot in
plain English. No finance background required.

Usage:
    python technical_analysis/bot/post_help.py

Environment variables (in .env):
    JK_DISCORD_HELP_CHANNEL   — channel ID for #bot-guide (read-only)
    JK_DISCORD_BOT_TOKEN      — bot token for REST posting
"""

import os
import sys
import time
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

CHANNEL_ID = os.getenv("JK_DISCORD_HELP_CHANNEL", "")
BOT_TOKEN = os.getenv("JK_DISCORD_BOT_TOKEN", "")

if not CHANNEL_ID or not BOT_TOKEN:
    print("ERROR: Set JK_DISCORD_HELP_CHANNEL and JK_DISCORD_BOT_TOKEN in .env")
    sys.exit(1)

API_URL = f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages"
HEADERS = {"Authorization": f"Bot {BOT_TOKEN}", "Content-Type": "application/json"}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def ts():
    return datetime.now(timezone.utc).isoformat()


def post_embed(embed: dict, delay: float = 1.2):
    """Post a single embed to the #bot-guide channel via bot token."""
    payload = {"embeds": [embed]}
    try:
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=15)
        if resp.status_code == 429:
            retry_after = resp.json().get("retry_after", 5)
            print(f"  Rate limited — waiting {retry_after:.1f}s...")
            time.sleep(retry_after + 0.5)
            resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=15)
        if resp.status_code in (200, 201):
            print(f"  ✅ Posted: {embed.get('title', '(no title)')[:60]}")
        else:
            print(f"  ❌ HTTP {resp.status_code}: {resp.text[:200]}")
        time.sleep(delay)
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ─────────────────────────────────────────────────────────────
# Embed builders — one function per section
# ─────────────────────────────────────────────────────────────

def embed_intro():
    return {
        "title": "🤖 Welcome to the JK Trading Bot — Your Complete Guide",
        "color": 0x3498DB,
        "description": (
            "This channel is a **read-only reference guide**. "
            "Everything the bot does is explained here in plain English — "
            "no finance background needed.\n\n"
            "**What does this bot do?**\n"
            "Think of it as an automated analyst that watches the stock market 24/7. "
            "It reads price charts, measures momentum, detects trends, and "
            "decides when conditions look good to invest — and when to sit on the sidelines.\n\n"
            "It never trades with real money. It runs a **paper trading** simulation "
            "(fake money, real prices) so you can see exactly what it would have done "
            "without any financial risk.\n\n"
            "**How to use this channel:**\n"
            "Scroll through the sections below. Each one covers a specific topic. "
            "Real signals appear in <#alerts> and <#scans>. Backtest results are in "
            "<#backtest-results>."
        ),
        "footer": {"text": "JK Four Pillars Bot  •  Read-only guide  •  Updated " + ts()[:10]},
        "timestamp": ts(),
    }


def embed_tickers():
    return {
        "title": "📋 What Stocks / Funds Does the Bot Watch?",
        "color": 0x1ABC9C,
        "description": (
            "The bot trades **ETFs** — Exchange-Traded Funds. "
            "An ETF is like a basket of stocks bundled together into one thing you can buy. "
            "Instead of picking individual companies, you buy the whole basket at once.\n\n"
            "**Think of each ticker as a different 'basket':**"
        ),
        "fields": [
            {
                "name": "📦 The Core Four (daily monitoring)",
                "value": (
                    "**SPY** — The entire US stock market (500 biggest companies). "
                    "If SPY is up, America's biggest companies are doing well.\n"
                    "**QQQ** — The top 100 technology-heavy companies (Apple, Google, Nvidia, etc.)\n"
                    "**DIA** — The Dow Jones: 30 of America's most established companies\n"
                    "**IWM** — Small-cap stocks: 2,000 smaller American companies"
                ),
                "inline": False,
            },
            {
                "name": "🏭 The 11 Sector Baskets (backtest comparisons)",
                "value": (
                    "**XLK** — Technology (chips, software, hardware)\n"
                    "**XLF** — Financials (banks, insurance)\n"
                    "**XLE** — Energy (oil, gas, drilling)\n"
                    "**XLV** — Health Care (hospitals, pharma, biotech)\n"
                    "**XLI** — Industrials (factories, defense, transportation)\n"
                    "**XLP** — Consumer Staples (food, household goods people always buy)\n"
                    "**XLY** — Consumer Discretionary (cars, restaurants, shopping)\n"
                    "**XLU** — Utilities (electricity, water — very stable)\n"
                    "**XLC** — Communication Services (Meta, Netflix, Google)\n"
                    "**XLRE** — Real Estate (property companies, REITs)\n"
                    "**XLB** — Materials (mining, chemicals, packaging)\n"
                    "**DIA** — Dow Jones (also tracked as a sector comparison)"
                ),
                "inline": False,
            },
            {
                "name": "💡 Why ETFs and not individual stocks?",
                "value": (
                    "ETFs spread risk across many companies at once. "
                    "If one company has bad news, it barely moves the whole basket. "
                    "This makes the signals cleaner and more reliable."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_four_pillars_simple():
    return {
        "title": "🏛️ How the Bot Makes Decisions — The Four Pillars",
        "color": 0x2ECC71,
        "description": (
            "The bot uses **four independent checks** before deciding to buy anything. "
            "Think of it like a hiring panel — all four members need to say 'yes' "
            "for anything to happen. If even one says 'no', the bot stays on the sidelines.\n\n"
            "This is called the **Four Pillars** strategy:"
        ),
        "fields": [
            {
                "name": "🌤️ Pillar 1 — Regime: Is the market in a good mood?",
                "value": (
                    "The bot first asks: **is the overall trend going up, sideways, or down?**\n\n"
                    "It measures this with a 'trend score' from -5 to +5 by looking at:\n"
                    "• Are prices above their recent averages? (Moving Averages)\n"
                    "• Is the trend strong or weak? (ADX indicator)\n"
                    "• Is a new trend just starting? (Aroon indicator)\n\n"
                    "**Score ≥ 2 → BULL** 🟢 (market going up — can invest)\n"
                    "**Score -2 to +2 → CHOP** 🟡 (sideways — invest carefully)\n"
                    "**Score ≤ -2 → BEAR** 🔴 (market going down — stay flat)\n\n"
                    "Analogy: This is like checking the weather before a picnic. "
                    "If it's storming (BEAR), you don't go outside no matter what."
                ),
                "inline": False,
            },
            {
                "name": "⏱️ Pillar 2 — Timing: Is it a good moment to buy?",
                "value": (
                    "Even in a rising market, prices go up and down like waves. "
                    "The bot waits for a dip before entering.\n\n"
                    "It uses a 'z-score' — a measure of how far below normal the price is. "
                    "Think of it like a rubber band: the more it's stretched down, "
                    "the more likely it snaps back up.\n\n"
                    "**z-score ≤ -1.1** → Oversold (stretched down — buy signal) ✅\n"
                    "**z-score ≥ 2.5** → Overbought (stretched up — sell or reduce) ⚠️\n"
                    "**In between** → Neutral — nothing to do\n\n"
                    "Analogy: This is the 'buy on a dip' principle. "
                    "Don't buy stocks when they're at their peak — wait for a pullback."
                ),
                "inline": False,
            },
            {
                "name": "📈 Pillar 3 — Momentum: Is it starting to recover?",
                "value": (
                    "The bot doesn't just want a dip — it wants to see the price "
                    "*starting to turn back up* before entering. This avoids "
                    "'catching a falling knife.'\n\n"
                    "It checks the slope of an oscillator (a line that tracks price momentum). "
                    "If the line is flattening or turning upward, that's confirming.\n\n"
                    "Analogy: You wouldn't catch a ball while it's still falling. "
                    "Wait until it bounces slightly, then grab it."
                ),
                "inline": False,
            },
            {
                "name": "📊 Pillar 4 — Volume: Are big players participating?",
                "value": (
                    "Volume is how many shares are being traded. "
                    "High volume on a recovery = institutional investors (big funds) are buying. "
                    "Low volume = might just be noise.\n\n"
                    "The bot checks:\n"
                    "• Is trading volume higher than the 65-day average?\n"
                    "• Is the Volume-Enhanced RSI (a momentum measure) strong?\n\n"
                    "Analogy: If a restaurant is suddenly packed AND has great reviews, "
                    "it's probably actually good. If it's packed but all the regulars left, "
                    "something's off. Volume tells you who's actually showing up."
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "All 4 pillars must align for STRONG_BUY. 3/4 = BUY. Less = HOLD/FLAT."},
        "timestamp": ts(),
    }


def embed_regimes():
    return {
        "title": "🌤️ Understanding Market Regimes: BULL, CHOP, and BEAR",
        "color": 0xF39C12,
        "description": (
            "The most important decision the bot makes every day is: "
            "**what regime is the market in right now?** Everything else flows from this.\n\n"
            "The regime is determined by the **trend score** (a number from -5 to +5):"
        ),
        "fields": [
            {
                "name": "🟢 BULL Regime (trend score ≥ 2)",
                "value": (
                    "The market is in a clear uptrend. Prices are above their moving averages, "
                    "the trend is strong, and momentum is positive.\n\n"
                    "**What the bot does in BULL:**\n"
                    "• Always keeps at least 50% invested (baseline)\n"
                    "• Looks for dip-buy opportunities to scale up to 75–100%\n"
                    "• Sells overbought positions when the rubber band is stretched too far up\n\n"
                    "Think: 'The tide is coming in. Swim with it.'"
                ),
                "inline": False,
            },
            {
                "name": "🟡 CHOP Regime (trend score -2 to +2)",
                "value": (
                    "The market is going sideways — no clear direction. "
                    "Prices bounce between a range without making new highs or lows.\n\n"
                    "**What the bot does in CHOP:**\n"
                    "• Keeps 50% invested (same baseline as BULL — this was a key discovery)\n"
                    "• Still looks for mean-reversion trades within the range\n"
                    "• More conservative — no scaling to 100%\n\n"
                    "Think: 'The ocean is calm. Don't go too far from shore.'"
                ),
                "inline": False,
            },
            {
                "name": "🔴 BEAR Regime (trend score ≤ -2)",
                "value": (
                    "The market is in a clear downtrend. Most things are falling.\n\n"
                    "**What the bot does in BEAR:**\n"
                    "• Goes to 0% — completely flat, all cash\n"
                    "• Does NOT try to catch the falling market\n"
                    "• Exception: very rare 'deep oversold' counter-trend plays at 25%\n\n"
                    "Think: 'The tide is going out. Get out of the water.'\n\n"
                    "⚠️ This is one of the bot's most valuable features — it simply "
                    "stops trading in bear markets and protects capital."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_signal_types():
    return {
        "title": "🚦 Understanding Every Signal Type",
        "color": 0x3498DB,
        "description": (
            "When you see an alert in <#scans> or <#alerts>, here's exactly what it means:"
        ),
        "fields": [
            {
                "name": "🚀 STRONG_BUY — All 4 pillars confirming",
                "value": (
                    "The highest conviction signal. All four checks pass:\n"
                    "✅ Bull or Chop regime\n"
                    "✅ Price deeply oversold (z-score ≤ -1.5)\n"
                    "✅ Momentum turning upward\n"
                    "✅ Volume confirming\n\n"
                    "**Action:** Scale up to 75–100% of capital\n"
                    "**Example:** SPY has pulled back 2% in an uptrend, RSI is weak but rising, "
                    "volume is spiking. Classic dip-buy entry."
                ),
                "inline": False,
            },
            {
                "name": "📈 BUY — 3/4 pillars confirming",
                "value": (
                    "A solid signal with one pillar not fully confirming.\n"
                    "✅ Bull regime\n"
                    "✅ Oversold (z-score ≤ -1.1)\n"
                    "⚠️ Momentum or volume partially confirming\n\n"
                    "**Action:** Scale up to 50–75% of capital\n"
                    "Lower conviction than STRONG_BUY but still a valid entry."
                ),
                "inline": False,
            },
            {
                "name": "⏳ HOLD — In a position, no action needed",
                "value": (
                    "The bot is already invested and the conditions remain supportive. "
                    "Nothing to do — just stay in the trade.\n\n"
                    "**Action:** Maintain current position\n"
                    "This appears in the paper trading cycle to show existing trades."
                ),
                "inline": False,
            },
            {
                "name": "⚠️ REDUCE — Overbought or regime weakening",
                "value": (
                    "The price has stretched too far up (rubber band too tight) or "
                    "the trend is starting to weaken.\n"
                    "Z-score ≥ 2.5 (overbought) or regime shifting toward CHOP/BEAR\n\n"
                    "**Action:** Scale down toward baseline (50%) or lower\n"
                    "The bot is locking in some gains but not fully exiting."
                ),
                "inline": False,
            },
            {
                "name": "💤 FLAT — Bear regime or no setup",
                "value": (
                    "Either the market is in a downtrend (BEAR regime) or "
                    "conditions aren't favorable enough for any trade.\n\n"
                    "**Action:** 0% invested — hold cash\n"
                    "This is NOT a failure. Staying out of bad conditions "
                    "is one of the most valuable things a strategy can do. "
                    "Many strategies lose money by trading when they shouldn't."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_position_sizing():
    return {
        "title": "💰 Position Sizing: Why Doesn't the Bot Go 'All In'?",
        "color": 0xF1C40F,
        "description": (
            "The bot never bets everything on one trade. It uses **graduated position sizing** "
            "— like a dimmer switch, not an on/off light. Here's the full scale:\n\n"
        ),
        "fields": [
            {
                "name": "📊 The Position Size Table",
                "value": (
                    "```\n"
                    "Condition                              Position\n"
                    "────────────────────────────────────   ────────\n"
                    "Bear regime                            0%  (all cash)\n"
                    "Bull/Chop baseline (always)            50% (floor)\n"
                    "Bull + overbought                      25% (reduce)\n"
                    "Bull + oversold + partial confirm      75%\n"
                    "Bull + deep oversold + all 4 pillars   100% (max)\n"
                    "Bear + deep oversold + dual confirm    25% (rare)\n"
                    "```"
                ),
                "inline": False,
            },
            {
                "name": "🧠 Why not just go 100% every time?",
                "value": (
                    "Because that would expose you to big losses on bad signals. "
                    "The bot uses smaller positions when conviction is lower, "
                    "and larger positions when ALL evidence points the same direction.\n\n"
                    "The 50% 'baseline' in bull markets is intentional — "
                    "it's always somewhat invested in an uptrend, "
                    "but saves half for better entry points."
                ),
                "inline": False,
            },
            {
                "name": "📐 Example: A day in the life of position sizing",
                "value": (
                    "**Monday:** SPY is in a bull regime, no signal → **50% invested**\n"
                    "**Tuesday:** SPY pulls back, z-score hits -1.3, volume spikes → scale to **75%**\n"
                    "**Wednesday:** SPY rallies, z-score reaches -0.5 → **50%** (back to baseline)\n"
                    "**Thursday:** SPY rallies further, z-score = +2.8 → **25%** (overbought, reduce)\n"
                    "**Friday:** SPY continues up but regime shifts BEAR → **0%** (exit all)"
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_risk_management():
    return {
        "title": "🛡️ How the Bot Protects You — Risk Management Rules",
        "color": 0xE74C3C,
        "description": (
            "Three automatic 'stop' rules protect against big losses. "
            "These trigger on any position that went **above the 50% baseline** "
            "(i.e., an active tactical trade, not just the passive baseline holding)."
        ),
        "fields": [
            {
                "name": "🛑 Stop Loss — Hard floor at -5%",
                "value": (
                    "**What it is:** If a trade drops 5% from where the bot bought, "
                    "it immediately reduces back to the baseline (50% or 0%).\n\n"
                    "**Why:** Prevents a small loss from becoming a big one. "
                    "If the analysis was wrong, cut and move on.\n\n"
                    "**Example:** Bot buys SPY at $500. If SPY falls to $475 (-5%), "
                    "the extra position above baseline is closed. "
                    "A STOP_LOSS alert fires in the trade log."
                ),
                "inline": False,
            },
            {
                "name": "📈 Trailing Stop — Lock in gains (2% below peak)",
                "value": (
                    "**What it is:** Once a trade is UP 3% or more, "
                    "the bot starts following the price up. If it falls 2% from "
                    "the highest point it reached, the trade closes.\n\n"
                    "**Why:** Lets winners run, but locks in most of the gain. "
                    "If a trade goes up 8% and then falls back 2%, you still "
                    "keep most of the profit.\n\n"
                    "**Example:** Bot buys at $500. Price goes to $540 (+8%). "
                    "Trailing stop is now set at $529.20 (2% below $540). "
                    "If price falls to $529, a TRAIL_STOP alert fires and trade closes."
                ),
                "inline": False,
            },
            {
                "name": "⏰ Time Stop — Max 60 days per trade",
                "value": (
                    "**What it is:** If a trade is open for 60 days without "
                    "reaching a profit target or stop, it closes automatically.\n\n"
                    "**Why:** Prevents 'zombie trades' — positions that just sit there "
                    "tying up capital without doing anything useful. "
                    "Capital that isn't working should be freed up for better opportunities.\n\n"
                    "**Example:** Bot buys XLK on Jan 1. By March 1 (60 days), "
                    "it's still just flat at +0.5%. The time stop fires."
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "These rules are purely protective. They never increase position size."},
        "timestamp": ts(),
    }


def embed_exit_types():
    return {
        "title": "🚪 Understanding Exit Alerts: What Closed a Trade?",
        "color": 0x9B59B6,
        "description": (
            "When a trade closes, an alert shows **why it closed**. "
            "Here are all possible exit types:"
        ),
        "fields": [
            {
                "name": "🛑 STOP_LOSS",
                "value": (
                    "The trade fell 5% from entry. Cut to baseline.\n"
                    "What it means: The signal was wrong, or bad news hit. "
                    "The bot accepted a small loss to avoid a bigger one.\n"
                    "**Healthy to see occasionally** — it means the risk rules are working."
                ),
                "inline": False,
            },
            {
                "name": "🔒 TRAIL_STOP",
                "value": (
                    "The trade gained 3%+ then pulled back 2% from its peak. "
                    "Profit locked in.\n"
                    "What it means: A winning trade that the bot successfully exited "
                    "near the top. This is a *good* exit — profits were secured."
                ),
                "inline": False,
            },
            {
                "name": "⏰ TIME_STOP",
                "value": (
                    "The trade was open for 60 days with no major move. "
                    "Exited to free up capital.\n"
                    "What it means: The trade just sat there flat. "
                    "Not a loss necessarily — just dead capital reclaimed."
                ),
                "inline": False,
            },
            {
                "name": "📉 SIGNAL_EXIT",
                "value": (
                    "The Four Pillars changed — regime weakened, z-score went neutral, "
                    "or conditions no longer support the trade.\n"
                    "What it means: The original reason for holding is gone. "
                    "This is a clean, planned exit based on new information."
                ),
                "inline": False,
            },
            {
                "name": "📊 Summary",
                "value": (
                    "```\n"
                    "Exit Type    Why it fired              Good or bad?\n"
                    "──────────   ─────────────────────     ──────────────\n"
                    "STOP_LOSS    Price dropped -5%          Protective — good\n"
                    "TRAIL_STOP   Profit peak then -2%        Locking gains — great\n"
                    "TIME_STOP    60 days held, no move       Neutral — freeing capital\n"
                    "SIGNAL_EXIT  Signal conditions gone      Planned — normal\n"
                    "```"
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_reading_scan_alert():
    return {
        "title": "📖 How to Read a Scan Alert",
        "color": 0x1ABC9C,
        "description": (
            "Every morning (9:15 AM ET) and evening (4:15 PM ET), "
            "the bot scans all tickers and posts a summary to <#scans>. "
            "Here's how to read it:"
        ),
        "fields": [
            {
                "name": "📌 Example Scan Entry",
                "value": (
                    "```\n"
                    "SPY  🚀 STRONG_BUY  $527.43  pos=85%\n"
                    "  Regime: BULL (score: 3.2)\n"
                    "  Timing: z=-1.6 (deeply oversold)\n"
                    "  Momentum: ✅ confirming (slope: +0.8)\n"
                    "  Volume: ✅ confirming (ve_rsi=62, vol_ratio=1.4x)\n"
                    "```"
                ),
                "inline": False,
            },
            {
                "name": "Line by line breakdown:",
                "value": (
                    "**`SPY`** — The ticker being analyzed\n"
                    "**`🚀 STRONG_BUY`** — The signal (see signal types above)\n"
                    "**`$527.43`** — Current price\n"
                    "**`pos=85%`** — How much capital the bot would invest (85% of portfolio)\n"
                    "**`Regime: BULL (score: 3.2)`** — Market is in an uptrend. Score 3.2 out of max 5\n"
                    "**`Timing: z=-1.6`** — The rubber band is stretched DOWN 1.6 units — oversold\n"
                    "**`Momentum: ✅`** — The oscillator slope is turning upward (recovery starting)\n"
                    "**`Volume: ✅ (vol_ratio=1.4x)`** — 40% more volume than the 65-day average"
                ),
                "inline": False,
            },
            {
                "name": "💡 What the numbers mean at a glance",
                "value": (
                    "**z-score below -1.1** → oversold, looking for a bounce\n"
                    "**z-score above 2.5** → overbought, consider reducing\n"
                    "**vol_ratio > 1.2** → above-average volume (institutional interest)\n"
                    "**trend_score > 2** → bull regime; **< -2** → bear regime\n"
                    "**pos% = 0** → bot is completely out (FLAT or BEAR)"
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_reading_trade_alert():
    return {
        "title": "📖 How to Read a Trade Alert",
        "color": 0xE67E22,
        "description": (
            "When the bot executes a paper trade (buy or sell), it posts a detailed "
            "alert to <#alerts>. Here's how to read it:"
        ),
        "fields": [
            {
                "name": "📌 Example Trade Alert — Entry",
                "value": (
                    "```\n"
                    "📈 PAPER TRADE — BUY\n"
                    "Ticker:  SPY\n"
                    "Price:   $527.43\n"
                    "Size:    85% of portfolio ($85,000)\n"
                    "Signal:  STRONG_BUY\n"
                    "Regime:  BULL (score 3.2)\n"
                    "Reason:  z=-1.6 (oversold), vol=1.4x avg\n"
                    "```"
                ),
                "inline": False,
            },
            {
                "name": "📌 Example Trade Alert — Exit",
                "value": (
                    "```\n"
                    "✅ PAPER TRADE — SELL (TRAIL_STOP)\n"
                    "Ticker:  SPY\n"
                    "Entry:   $527.43  →  Exit: $548.12\n"
                    "P&L:     +3.9%  ($3,900 on $100k portfolio)\n"
                    "Held:    14 days\n"
                    "Exit:    Trailing stop fired (peak was $551.23)\n"
                    "```"
                ),
                "inline": False,
            },
            {
                "name": "Key fields explained:",
                "value": (
                    "**P&L** = Profit & Loss — how much was made or lost on this trade\n"
                    "**Held** = How many days the trade was open\n"
                    "**Exit type** = Why the trade closed (TRAIL_STOP, STOP_LOSS, SIGNAL_EXIT, TIME_STOP)\n"
                    "**Portfolio** = This is paper trading — no real money involved\n\n"
                    "The portfolio starts at **$100,000 simulated** and tracks gains/losses over time."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_backtest_results():
    return {
        "title": "📊 How to Read Backtest Results",
        "color": 0x8E44AD,
        "description": (
            "Backtest results in <#backtest-results> show how the strategy **would have performed** "
            "if it had been running in the past. Here's what every metric means:"
        ),
        "fields": [
            {
                "name": "📌 Example Backtest Output",
                "value": (
                    "```\n"
                    "SPY — 10-Year Backtest\n"
                    "Sharpe Ratio:   1.24  (benchmark: 0.89)\n"
                    "Annual Return:  13.2% (benchmark: 11.1%)\n"
                    "Total Return:   244%  (benchmark: 187%)\n"
                    "Max Drawdown:   -18.4%\n"
                    "Win Rate:       62%\n"
                    "Avg Win:        +4.1%  Avg Loss: -2.3%\n"
                    "```"
                ),
                "inline": False,
            },
            {
                "name": "🔢 Every metric explained",
                "value": (
                    "**Sharpe Ratio** — The 'quality' of returns. "
                    "It measures how much return you get for the risk you take. "
                    "Above 1.0 is good. Above 1.5 is excellent. "
                    "The benchmark Sharpe is what you'd get just buying and holding.\n\n"
                    "**Annual Return** — Average yearly gain as a percentage\n\n"
                    "**Total Return** — How much a $100k investment would have grown total over the period\n\n"
                    "**Max Drawdown** — The biggest loss from peak to trough at any point. "
                    "If your portfolio hit $150k then fell to $123k, that's -18%. "
                    "Smaller is better.\n\n"
                    "**Win Rate** — What % of closed trades were profitable. "
                    "Note: 60% win rate with good avg win/loss ratio is fine — "
                    "you don't need to be right all the time.\n\n"
                    "**Avg Win / Avg Loss** — If wins average +4% and losses average -2%, "
                    "the bot only needs to be right 33% of the time to break even. "
                    "This ratio matters more than win rate alone."
                ),
                "inline": False,
            },
            {
                "name": "✅ vs ❌ in the sector backtest table",
                "value": (
                    "**✅** = This ticker's Sharpe ratio beat 'just buying and holding' that ETF\n"
                    "**❌** = The strategy underperformed for this ticker\n\n"
                    "The goal: strategy beats benchmark on at least 7 out of 12 tickers.\n"
                    "The strategy is optimized using the Composite Sharpe across SPY, QQQ, DIA, and IWM."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_intraday_results():
    return {
        "title": "📈 Intraday Day-Trading Simulations Explained",
        "color": 0x16A085,
        "description": (
            "The <#backtest-results> channel also shows **intraday simulations** — "
            "what would have happened if the bot traded *during the day* on specific dates. "
            "These use the same Four Pillars logic but adapted to minute-by-minute price bars."
        ),
        "fields": [
            {
                "name": "How intraday sims work",
                "value": (
                    "1. **Morning regime check** — the bot reads the last 40 daily bars to "
                    "classify today's market regime (BULL/CHOP/BEAR)\n"
                    "2. **Intraday data** — fetches 5-minute or 1-hour bars for that day\n"
                    "3. **Entry rule** — price dips 0.15% below 20-period average AND RSI < 45\n"
                    "4. **Exit rule** — price recovers 0.1% above 20-period average OR RSI > 58 "
                    "OR -1.5% stop loss OR end of trading day (forced close)\n"
                    "5. **Bear filter** — if morning regime is BEAR, NO trades fire at all\n\n"
                    "These are meant to show the *style* of the strategy intraday, "
                    "not to recommend live day-trading."
                ),
                "inline": False,
            },
            {
                "name": "Why some dates show zero trades",
                "value": (
                    "In a strong bull market, prices rarely dip 0.15% below their "
                    "recent average — so the entry signal never fires. "
                    "This is correct behavior: if there's no dip to buy, the bot doesn't chase.\n\n"
                    "BEAR regime dates also show zero trades — the regime filter correctly "
                    "keeps the bot on the sidelines when the market is falling."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_self_learning():
    return {
        "title": "🧠 The Self-Learning System (AutoResearch)",
        "color": 0x2980B9,
        "description": (
            "This is one of the most unique features of the bot: "
            "it **automatically improves itself** over time using AI."
        ),
        "fields": [
            {
                "name": "How it works",
                "value": (
                    "1. **Start:** The bot has a set of 'parameters' — numbers that control "
                    "when to buy, when to sell, how much to invest, etc.\n"
                    "2. **Propose:** It asks Claude AI to suggest 3 small changes to "
                    "these parameters based on what's been tried before\n"
                    "3. **Test in parallel:** All 3 experiments run simultaneously, "
                    "backtesting 10 years of history each\n"
                    "4. **Keep the best:** Only the change that improved performance is saved. "
                    "Others are discarded.\n"
                    "5. **Log everything:** Every experiment is recorded so nothing "
                    "gets tested twice\n"
                    "6. **Out-of-sample gate:** A new 'best' must also perform well on a "
                    "separate 2-year window it wasn't optimized on. "
                    "This prevents the bot from just memorizing the past.\n"
                    "7. **Repeat** — runs 30–50 experiments per session\n\n"
                    "This runs every Saturday night automatically."
                ),
                "inline": False,
            },
            {
                "name": "What it's optimizing for",
                "value": (
                    "The bot tries to maximize the **Composite Sharpe Ratio** — "
                    "a single score that grades performance across SPY, QQQ, DIA, and IWM "
                    "simultaneously, with a penalty if any ticker underperforms just "
                    "holding that ETF.\n\n"
                    "Current best Composite Sharpe: **0.8705** (all 4 tickers beat benchmarks)"
                ),
                "inline": False,
            },
            {
                "name": "What has it learned so far? (44+ experiments)",
                "value": (
                    "• **Biggest single improvement:** Raising the CHOP baseline from 25% → 50%. "
                    "Being partially invested in sideways markets matters.\n"
                    "• **BULL baseline plateau:** More than 50% in bull markets hurts performance. "
                    "Counterintuitive, but the timing signals add more value than raw exposure.\n"
                    "• **Tight stops hurt:** A 2-3% stop loss gets triggered too easily "
                    "by normal market noise (whipsawing).\n"
                    "• **QQQ and IWM are hardest** — tech and small-cap move differently. "
                    "The multi-ticker objective forces the AI to find parameters "
                    "that work across all styles."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_commands():
    return {
        "title": "🤖 Discord Commands — Complete Reference",
        "color": 0x9B59B6,
        "description": (
            "These commands work in any channel where the bot is active. "
            "Type them and press Enter:\n\n"
            "*(Note: this channel is read-only — commands in <#bot-commands>)*"
        ),
        "fields": [
            {
                "name": "!scan [tickers]",
                "value": (
                    "Runs the Four Pillars analysis on tickers right now and posts results.\n"
                    "Example: `!scan SPY QQQ XLK`\n"
                    "Without tickers: scans the default watchlist (SPY, QQQ, DIA, IWM + sectors)"
                ),
                "inline": False,
            },
            {
                "name": "!trade [tickers]",
                "value": (
                    "Runs the paper trading cycle — evaluates all positions and executes "
                    "any new entries or exits based on current signals.\n"
                    "Example: `!trade SPY QQQ`"
                ),
                "inline": False,
            },
            {
                "name": "!status",
                "value": (
                    "Shows the current paper trading portfolio — all open positions, "
                    "unrealized P&L, total portfolio value vs the $100k starting capital."
                ),
                "inline": False,
            },
            {
                "name": "!history",
                "value": (
                    "Shows the last 10 closed trades from the paper trading log — "
                    "what was bought, when, at what price, and what the outcome was."
                ),
                "inline": False,
            },
            {
                "name": "!params",
                "value": (
                    "Shows the current optimized strategy parameters — "
                    "the exact thresholds being used for every decision."
                ),
                "inline": False,
            },
            {
                "name": "!whatif [ticker]",
                "value": (
                    "Shows how close a ticker is to triggering a buy or sell signal *right now*.\n"
                    "Example: `!whatif SPY`\n\n"
                    "Output shows:\n"
                    "• Current regime and trend score\n"
                    "• Current z-score vs the buy threshold (how far from triggering?)\n"
                    "• Which pillars are currently confirming\n"
                    "• What would need to change to trigger a buy\n\n"
                    "Useful for: 'How close is SPY to a buy signal?'"
                ),
                "inline": False,
            },
            {
                "name": "!backtest [ticker] [period]",
                "value": (
                    "Runs a full historical backtest for a ticker.\n"
                    "Example: `!backtest XLK 5y`\n"
                    "Periods: `1y` `2y` `5y` `10y`"
                ),
                "inline": False,
            },
            {
                "name": "!learn [n]",
                "value": (
                    "Runs n rounds of AutoResearch — the AI self-improvement loop. "
                    "Each round tests 3 hypotheses in parallel.\n"
                    "Example: `!learn 10`\n"
                    "Warning: takes several minutes. Best to run in background."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_schedule():
    return {
        "title": "🗓️ Automated Schedule — When Does the Bot Run?",
        "color": 0x27AE60,
        "description": "The bot runs automatically on this schedule (all times US Eastern):",
        "fields": [
            {
                "name": "⏰ Daily Automated Runs (weekdays only)",
                "value": (
                    "**9:15 AM ET — Pre-Market Scan**\n"
                    "Scans all tickers before the market opens. "
                    "Uses prior day's close prices. Posts to <#scans>.\n\n"
                    "**4:15 PM ET — Post-Close Scan + Paper Trade**\n"
                    "Full scan after market closes for the day. "
                    "Also runs the paper trading cycle — "
                    "executes any new entries or exits. Posts to <#scans> and <#alerts>."
                ),
                "inline": False,
            },
            {
                "name": "📅 Weekly Automated Run (Saturday night)",
                "value": (
                    "**Saturday 10 PM ET — AutoResearch Session**\n"
                    "The self-learning loop runs for up to 2 hours, "
                    "running up to 50 parameter experiments. "
                    "If better parameters are found, `best_params.json` is updated "
                    "and all future signals use the improved thresholds."
                ),
                "inline": False,
            },
            {
                "name": "🔄 Manual triggers",
                "value": (
                    "Any schedule can also be triggered manually via Discord commands "
                    "or CLI. The bot doesn't require manual intervention to operate."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_glossary():
    return {
        "title": "📚 Key Terms Glossary — Plain English Definitions",
        "color": 0x7F8C8D,
        "fields": [
            {
                "name": "Technical Terms",
                "value": (
                    "**Moving Average (SMA/EMA)** — The average price over the last N days. "
                    "A price above its 50-day moving average suggests upward momentum.\n\n"
                    "**RSI (Relative Strength Index)** — A 0–100 scale measuring "
                    "how overbought (>70) or oversold (<30) a stock is. "
                    "The bot uses a volume-enhanced version.\n\n"
                    "**Z-Score** — How many 'standard deviations' from normal something is. "
                    "Z = -1.5 means the price is 1.5 standard deviations below its recent average. "
                    "Used for the timing pillar.\n\n"
                    "**ADX (Average Directional Index)** — Measures trend strength, not direction. "
                    "ADX > 25 means a strong trend. Used in Pillar 1.\n\n"
                    "**Aroon** — Tracks how recently a new high or low was made. "
                    "Signals the *start* of a new trend. Used in Pillar 1."
                ),
                "inline": False,
            },
            {
                "name": "Strategy Terms",
                "value": (
                    "**Paper Trading** — Simulated trading with fake money at real prices. "
                    "No financial risk. Tests strategy in real conditions without real money.\n\n"
                    "**Backtesting** — Running a strategy on historical data to see how it "
                    "would have performed. The past doesn't guarantee the future, but "
                    "it's the best data we have.\n\n"
                    "**Composite Sharpe** — Single optimization target measuring performance "
                    "across SPY+QQQ+DIA+IWM with a penalty for underperforming any one ticker.\n\n"
                    "**Tactical Position** — Any investment *above* the baseline 50%. "
                    "This is the active part that stop-loss rules apply to.\n\n"
                    "**Baseline** — The 'always on' position held regardless of signals. "
                    "50% in bull/chop, 0% in bear. The bot never goes below this in non-bear regimes."
                ),
                "inline": False,
            },
            {
                "name": "Performance Terms",
                "value": (
                    "**Sharpe Ratio** — Return divided by risk. "
                    "> 1.0 = good, > 1.5 = excellent, > 2.0 = exceptional.\n\n"
                    "**Max Drawdown** — Largest peak-to-trough loss. "
                    "If your $100k account hit $120k then fell to $95k, drawdown = -20.8%.\n\n"
                    "**Win Rate** — Percentage of closed trades that made money. "
                    "Doesn't tell the full story — a 40% win rate with 3:1 reward:risk is profitable.\n\n"
                    "**Benchmark** — Just buying and holding the ETF. "
                    "The strategy's job is to beat 'do nothing' on a risk-adjusted basis."
                ),
                "inline": False,
            },
        ],
        "timestamp": ts(),
    }


def embed_disclaimers():
    return {
        "title": "⚠️ Important Notes",
        "color": 0x95A5A6,
        "description": (
            "A few things to be aware of before interpreting any signals:"
        ),
        "fields": [
            {
                "name": "📝 Paper Trading Only",
                "value": (
                    "All trades shown in this server are **simulated paper trades**. "
                    "No real money is involved. A $100,000 virtual portfolio is used "
                    "to track performance without any real financial exposure."
                ),
                "inline": False,
            },
            {
                "name": "📊 Past Performance ≠ Future Results",
                "value": (
                    "Backtests use historical data. The market in the future may "
                    "behave differently. A strategy that worked over the past 10 years "
                    "may not work the same way going forward. This is a research tool, "
                    "not financial advice."
                ),
                "inline": False,
            },
            {
                "name": "🎯 The Bot's Goal",
                "value": (
                    "The strategy is designed to achieve better **risk-adjusted returns** "
                    "than simply buying and holding an index fund — specifically by:\n"
                    "• Avoiding large drawdowns during bear markets (going to cash)\n"
                    "• Buying pullbacks within uptrends (mean-reversion timing)\n"
                    "• Scaling position size based on signal confidence\n\n"
                    "It is NOT designed to maximize absolute returns, "
                    "outperform individual growth stocks, or predict crashes in advance."
                ),
                "inline": False,
            },
            {
                "name": "🔒 This Channel",
                "value": (
                    "This channel is **read-only** and is refreshed periodically. "
                    "Use <#bot-commands> for running commands. "
                    "Signals appear in <#scans> and <#alerts>. "
                    "Backtest data is in <#backtest-results>."
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "JK Four Pillars Bot  •  Read-only guide  •  Not financial advice"},
        "timestamp": ts(),
    }


# ─────────────────────────────────────────────────────────────
# Build and post all embeds
# ─────────────────────────────────────────────────────────────

def build_all_embeds():
    return [
        embed_intro(),
        embed_tickers(),
        embed_four_pillars_simple(),
        embed_regimes(),
        embed_signal_types(),
        embed_position_sizing(),
        embed_risk_management(),
        embed_exit_types(),
        embed_reading_scan_alert(),
        embed_reading_trade_alert(),
        embed_backtest_results(),
        embed_intraday_results(),
        embed_self_learning(),
        embed_commands(),
        embed_schedule(),
        embed_glossary(),
        embed_disclaimers(),
    ]


def main():
    embeds = build_all_embeds()
    print(f"\n{'='*55}")
    print(f"  JK Bot — Posting Help Guide to Discord")
    print(f"  Embeds: {len(embeds)}")
    print(f"  Channel: {CHANNEL_ID}")
    print(f"{'='*55}\n")

    for embed in embeds:
        post_embed(embed, delay=1.2)

    print(f"\n{'='*55}")
    print(f"  ✅ Help guide posted ({len(embeds)} embeds)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
