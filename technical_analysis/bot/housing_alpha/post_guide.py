"""
Post the Housing Alpha guide embed to Discord and pin it.
"""
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

TOKEN = os.environ.get("JK_DISCORD_BOT_TOKEN", "")
CHANNEL_ID = "1485833592453070850"  # #housing-alpha

headers = {"Authorization": f"Bot {TOKEN}", "Content-Type": "application/json"}

embeds = [
    {
        "title": "🏠 Housing Alpha — What Is This?",
        "color": 0x2980B9,
        "description": (
            "This channel tracks a **self-improving trading strategy** that reads housing market data "
            "from the US government (FRED) and Zillow, then uses that to trade homebuilder ETFs "
            "(XHB and ITB) on a paper trading account.\n\n"
            "Think of it like a **weather forecast for the housing market** — it reads 24 different "
            "data points each month and tells you whether conditions are good, neutral, or bad "
            "for homebuilder stocks. The bot then places paper trades on Alpaca automatically."
        ),
    },
    {
        "title": "📊 The Three Regimes",
        "color": 0x27AE60,
        "fields": [
            {
                "name": "🏠📈  HOUSING_BULL",
                "value": (
                    "Housing market is **heating up**. Starts, permits, and sales are rising. "
                    "Rates are manageable or falling. Strategy goes **80% invested** in XHB/ITB.\n"
                    "*Examples: 2012–2018 housing recovery, 2020–2021 pandemic boom.*"
                ),
                "inline": False,
            },
            {
                "name": "🏠➡️  HOUSING_NEUTRAL",
                "value": (
                    "Mixed signals — some indicators up, some down. No strong trend. "
                    "Strategy stays **40% invested** — enough to participate but not overcommitted.\n"
                    "*Examples: 2018–2019 cooling period, early 2024.*"
                ),
                "inline": False,
            },
            {
                "name": "🏠📉  HOUSING_BEAR",
                "value": (
                    "Housing market is **deteriorating**. Rising rates crushing affordability, "
                    "inventory piling up, sales falling. Strategy drops to **10% invested** "
                    "(small contrarian position — not fully flat).\n"
                    "*Examples: 2022 rate shock when mortgage rates went from 3% to 7%.*"
                ),
                "inline": False,
            },
        ],
    },
    {
        "title": "🔬 The 5 Sub-Indicators Explained",
        "color": 0x8E44AD,
        "description": (
            "Each indicator shows a **z-score** — how far the current reading is from its "
            "3-year average. `+1.0` means 1 standard deviation above normal. "
            "`-2.0` means unusually weak.\n\u200b"
        ),
        "fields": [
            {
                "name": "🏗️  Activity Momentum",
                "value": (
                    "Combines **housing starts** (homes breaking ground), **building permits** "
                    "(homes about to break ground), and **new/existing home sales**.\n"
                    "**High = builders are busy, buyers are active.** Permits are the most "
                    "leading indicator — they signal what's coming 3–6 months from now."
                ),
                "inline": False,
            },
            {
                "name": "💸  Affordability Index  *(inverted — high = bearish)*",
                "value": (
                    "Measures how expensive it is to buy a home right now, using **30-year "
                    "mortgage rates**, **home price growth** (Case-Shiller), and **shelter inflation**.\n"
                    "**High = housing is unaffordable = bad for homebuilder stocks.** "
                    "The 2022 spike to 7%+ mortgage rates made this go through the roof."
                ),
                "inline": False,
            },
            {
                "name": "📦  Supply/Demand Balance  *(inverted — high = bearish)*",
                "value": (
                    "Looks at **months of supply** (how long to sell all listed homes at current pace) "
                    "and **active inventory** from Zillow.\n"
                    "**High = too much supply = price pressure = bad.** Under 4 months supply = tight "
                    "market = good. Over 6 months = buyer's market = bad for builders."
                ),
                "inline": False,
            },
            {
                "name": "📈  Price Momentum",
                "value": (
                    "Is the **rate of home price appreciation accelerating or decelerating?** "
                    "Uses the Case-Shiller national home price index and Zillow Home Value Index.\n"
                    "**Positive = prices still climbing fast.** Negative = price growth slowing "
                    "or reversing. This is a lagging signal — it confirms what already happened."
                ),
                "inline": False,
            },
            {
                "name": "🏦  Rate Regime",
                "value": (
                    "The **direction and speed of interest rate changes** — mortgage rates, "
                    "the Fed funds rate, and the yield curve (10Y minus 2Y Treasury).\n"
                    "**Positive = rates falling = tailwind for housing.** Negative = rates "
                    "rising = headwind. This is the single most important variable for "
                    "housing stocks in the short term."
                ),
                "inline": False,
            },
        ],
    },
    {
        "title": "⚠️ Rate Override — What Does It Mean?",
        "color": 0xE67E22,
        "description": (
            "When the **Rate Regime indicator falls sharply** (below -0.5 z-score), the strategy "
            "cuts all position targets in half — regardless of what the other indicators say.\n\n"
            "**Why?** Mortgage rates are the single biggest lever in housing. When rates are "
            "rising fast, even a strong housing market can crater quickly. The 2022 example: "
            "housing starts were still decent in early 2022, but XHB fell 40% as rates doubled "
            "from 3% to 7%. The rate override prevents staying fully invested while rates are "
            "actively being hiked."
        ),
    },
    {
        "title": "📈 The Two ETFs We Trade",
        "color": 0x16A085,
        "fields": [
            {
                "name": "XHB — SPDR S&P Homebuilders ETF",
                "value": (
                    "Holds ~35 homebuilder and housing-related stocks. Top names: "
                    "**D.R. Horton, Lennar, NVR, PulteGroup, Toll Brothers**. Also includes "
                    "Home Depot, floor/cabinet makers, etc. More diversified and lower volatility."
                ),
                "inline": False,
            },
            {
                "name": "ITB — iShares Home Construction ETF",
                "value": (
                    "More concentrated in **pure homebuilders** — D.R. Horton alone is ~15% of "
                    "the fund. Moves more aggressively than XHB in both directions. "
                    "Higher beta to the housing cycle. If we're right, it wins bigger. If wrong, it hurts more."
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "Both are paper trades on Alpaca — no real money at risk."},
    },
    {
        "title": "🧠 How It Self-Improves (AutoResearch)",
        "color": 0x2C3E50,
        "description": (
            "Every **Sunday at 11 PM**, the bot runs 30 automated experiments:\n\n"
            "1. It backtests the current parameters against **20 years of XHB/ITB data**\n"
            "2. A local AI proposes 3 tweaks (e.g. 'weight rate regime more heavily' or 'use 6-month lookback instead of 3')\n"
            "3. All 3 are backtested in parallel — takes about 3 minutes\n"
            "4. The best improvement is kept if it raises the **Sharpe ratio** (return per unit of risk)\n"
            "5. Everything is logged for full transparency\n\n"
            "After just the first 30 experiments the strategy improved **+42% in Sharpe ratio** "
            "(from 0.247 to 0.351 composite). Both XHB and ITB now beat their buy-and-hold benchmarks. "
            "The Four Pillars equity strategy has 700+ experiments behind it — this one is just getting started."
        ),
    },
    {
        "title": "💬 Bot Commands",
        "color": 0x7F8C8D,
        "fields": [
            {
                "name": "`!housing`",
                "value": "Show current regime, composite z-score, and all 5 sub-indicator readings",
                "inline": False,
            },
            {
                "name": "`!housing_status`",
                "value": "Show paper portfolio — NAV, current positions, entry prices, and unrealized P&L",
                "inline": False,
            },
            {
                "name": "`!housing_trade`",
                "value": "Manually trigger a rebalance (normally runs automatically on the 1st and 15th of each month)",
                "inline": False,
            },
        ],
        "footer": {
            "text": (
                "Signals post automatically on the 1st and 15th of each month after new FRED housing data releases. "
                "Data: FRED (St. Louis Fed) + Zillow Research"
            )
        },
    },
]

# Post the guide
r = requests.post(
    f"https://discord.com/api/v10/channels/{CHANNEL_ID}/messages",
    headers=headers,
    json={"content": "📌 **Housing Alpha Guide** — read this first", "embeds": embeds},
)
print(f"Post: {r.status_code}")
if r.status_code not in (200, 201):
    print(r.text[:500])
    raise SystemExit(1)

msg_id = r.json().get("id")
print(f"Message ID: {msg_id}")

# Pin it
r2 = requests.put(
    f"https://discord.com/api/v10/channels/{CHANNEL_ID}/pins/{msg_id}",
    headers=headers,
)
print(f"Pin: {r2.status_code} {'✅' if r2.status_code == 204 else r2.text[:200]}")
