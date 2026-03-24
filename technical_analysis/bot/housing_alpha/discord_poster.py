"""
Housing Alpha Discord Poster
==============================
Posts housing signals, backtest results, and learning progress to Discord.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

WEBHOOK_URL = os.environ.get("JK_DISCORD_HOUSING_WEBHOOK", "")


def _post_embed(embeds: list[dict], content: str = ""):
    """Post embed(s) to Discord via webhook."""
    if not WEBHOOK_URL:
        print("  [housing-discord] No webhook configured (JK_DISCORD_HOUSING_WEBHOOK)")
        return False

    payload = {"embeds": embeds}
    if content:
        payload["content"] = content

    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        print(f"  [housing-discord] Post failed: {e}")
        return False


def post_housing_signals(signals: list, engine=None):
    """Post current housing signals to Discord."""
    if not signals:
        return

    s = signals[0]  # all share same underlying composite

    # Regime color
    colors = {
        "HOUSING_BULL": 0x2ECC71,   # green
        "HOUSING_NEUTRAL": 0xF39C12,# orange
        "HOUSING_BEAR": 0xE74C3C,   # red
    }
    color = colors.get(s.regime, 0x95A5A6)

    # Regime emoji
    regime_emoji = {
        "HOUSING_BULL": "🏠📈",
        "HOUSING_NEUTRAL": "🏠➡️",
        "HOUSING_BEAR": "🏠📉",
    }

    # Sub-indicator bars
    def bar(val, label):
        import math
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return f"`{'░' * 10}` n/a {label}"
        filled = min(10, max(0, int((val + 2) / 4 * 10)))
        return f"`{'█' * filled}{'░' * (10 - filled)}` {val:+.2f} {label}"

    fields = [
        {"name": "Composite Signal", "value": f"**{s.composite_z:+.2f}** z-score", "inline": True},
        {"name": "Regime", "value": f"{regime_emoji.get(s.regime, '')} {s.regime}", "inline": True},
        {"name": "\u200b", "value": "\u200b", "inline": True},
        {"name": "Sub-Indicators", "value": (
            f"{bar(s.activity_z, 'Activity')}\n"
            f"{bar(-s.affordability_z, 'Afford (inv)')}\n"
            f"{bar(-s.supply_demand_z, 'Supply (inv)')}\n"
            f"{bar(s.price_momentum_z, 'Price Mom')}\n"
            f"{bar(s.rate_regime_z, 'Rate Regime')}"
        ), "inline": False},
    ]

    # Ticker allocations
    ticker_lines = []
    for sig in signals:
        ticker_lines.append(f"`{sig.ticker}` → **{sig.target_position:.0%}** allocation")
    fields.append({
        "name": "Target Allocations",
        "value": "\n".join(ticker_lines),
        "inline": False,
    })

    if s.rate_override_active:
        fields.append({
            "name": "⚠️ Rate Override",
            "value": "Mortgage rates rising sharply — positions reduced",
            "inline": False,
        })

    embed = {
        "title": f"🏠 Housing Alpha Signal — {s.date}",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Data: FRED + Zillow | Updated {datetime.now().strftime('%Y-%m-%d %H:%M')}"},
    }

    _post_embed([embed])
    print(f"  [housing-discord] Posted signal: {s.regime} (z={s.composite_z:+.2f})")


def post_backtest_results(results: dict):
    """Post backtest results to Discord."""
    composite = results.get("composite_sharpe", 0)

    fields = []
    for ticker in ["XHB", "ITB", "XLRE", "VNQ"]:
        r = results.get(ticker)
        if r is None or "error" in r:
            continue
        flag = "✅" if r.get("beats_benchmark") else "❌"
        fields.append({
            "name": f"{ticker} {flag}",
            "value": (
                f"Sharpe: **{r['sharpe_ratio']:+.3f}** (BM: {r['benchmark_sharpe']:+.3f})\n"
                f"Return: {r['annual_return']:+.1%}/yr | DD: {r['max_drawdown']:.1%}\n"
                f"Win Rate: {r['win_rate']:.0%} | Trades: {r['trade_count']}"
            ),
            "inline": True,
        })

    color = 0x2ECC71 if composite > 0.5 else (0xF39C12 if composite > 0 else 0xE74C3C)

    embed = {
        "title": f"📊 Housing Alpha Backtest — Composite Sharpe: {composite:+.4f}",
        "color": color,
        "fields": fields,
        "footer": {"text": f"Backtest run {datetime.now().strftime('%Y-%m-%d %H:%M')}"},
    }

    _post_embed([embed])
    print(f"  [housing-discord] Posted backtest: composite={composite:+.4f}")


def post_learning_improvement(
    old_sharpe: float,
    new_sharpe: float,
    changes: dict,
    hypothesis: str,
    experiment_num: int,
):
    """Post AutoResearch improvement to Discord."""
    improvement = new_sharpe - old_sharpe
    pct = (improvement / abs(old_sharpe) * 100) if old_sharpe != 0 else 0

    embed = {
        "title": f"🧠 Housing AutoResearch — Improvement Found!",
        "color": 0x2ECC71,
        "fields": [
            {"name": "Sharpe", "value": f"{old_sharpe:+.4f} → **{new_sharpe:+.4f}** (+{improvement:.4f}, {pct:+.1f}%)", "inline": False},
            {"name": "Changes", "value": f"```json\n{json.dumps(changes, indent=2)}\n```", "inline": False},
            {"name": "Hypothesis", "value": hypothesis[:200], "inline": False},
        ],
        "footer": {"text": f"Experiment #{experiment_num} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"},
    }

    _post_embed([embed])
