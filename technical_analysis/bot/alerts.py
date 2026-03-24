"""
Alert System
=============
Sends alerts via multiple channels when trade signals fire.
Supports: terminal, file log, Discord webhook, macOS notifications.

Setup Discord:
  1. In your Discord server, go to a channel → Edit Channel → Integrations → Webhooks
  2. Create a webhook, copy the URL
  3. Add to .env: JK_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from technical_analysis.bot.pillars import TradeSignal, PillarSnapshot


ALERT_LOG = Path(__file__).parent / "state" / "alerts.jsonl"


def format_alert(signal: TradeSignal) -> str:
    """Format a TradeSignal into a human-readable alert message."""
    snap = signal.snapshot
    lines = [
        f"{'='*50}",
        f"JK TRADING BOT — {signal.action} SIGNAL",
        f"{'='*50}",
        f"Ticker:    {signal.ticker}",
        f"Action:    {signal.action}",
        f"Position:  {signal.position_pct:.0%}",
        f"Reason:    {signal.reason}",
        f"Time:      {signal.timestamp}",
        f"",
        f"--- Four Pillars ---",
        f"P1 Regime:   {snap.regime.upper()} (trend_score={snap.trend_score_raw:+.0f})",
        f"P2 Timing:   {snap.timing_signal} (z_hybrid z={snap.z_hybrid_zscore:+.2f})",
        f"P3 Momentum: {'CONFIRMING' if snap.momentum_confirming else 'NOT confirming'} (slope={snap.hybrid_osc_slope:+.4f})",
        f"P4 Volume:   {'CONFIRMING' if snap.volume_confirming else 'NOT confirming'} (ve_rsi={snap.ve_rsi_raw:.1f}, vol={snap.volume_ratio:.2f}x)",
        f"Confidence:  {snap.confidence:.0%} ({snap.pillars_confirming}/4 pillars)",
    ]

    if signal.stop_price:
        lines.append(f"Stop Loss:   ${signal.stop_price:.2f}")
    if signal.trail_price:
        lines.append(f"Trail Stop:  ${signal.trail_price:.2f}")

    lines.append(f"{'='*50}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discord formatting
# ---------------------------------------------------------------------------

def format_discord_embed(signal: TradeSignal) -> dict:
    """Format a TradeSignal as a Discord webhook payload with rich embed."""
    snap = signal.snapshot
    color_map = {
        "BUY": 0x2ecc71,       # green
        "STRONG_BUY": 0x27ae60, # dark green
        "SELL": 0xe74c3c,       # red
        "REDUCE": 0xf39c12,     # orange
        "STOP_LOSS": 0xe74c3c,
        "TRAIL_STOP": 0xe67e22,
        "TIME_STOP": 0x95a5a6,
        "HOLD": 0x95a5a6,       # gray
    }
    color = color_map.get(signal.action, 0x95a5a6)

    # Emoji for action
    action_emoji = {
        "BUY": ":chart_with_upwards_trend:",
        "STRONG_BUY": ":rocket:",
        "SELL": ":chart_with_downwards_trend:",
        "REDUCE": ":warning:",
        "STOP_LOSS": ":octagonal_sign:",
        "TRAIL_STOP": ":shield:",
        "HOLD": ":hourglass:",
    }.get(signal.action, ":bell:")

    # Pillar status with checkmarks
    p1 = f"{':white_check_mark:' if snap.regime == 'bull' else ':x:'} **Regime:** {snap.regime.upper()} (score={snap.trend_score_raw:+.0f})"
    p2_ok = snap.timing_signal in ("oversold", "deep_oversold")
    p2 = f"{':white_check_mark:' if p2_ok else ':x:'} **Timing:** {snap.timing_signal} (z={snap.z_hybrid_zscore:+.2f})"
    p3 = f"{':white_check_mark:' if snap.momentum_confirming else ':x:'} **Momentum:** {'confirming' if snap.momentum_confirming else 'not confirming'} (slope={snap.hybrid_osc_slope:+.4f})"
    p4 = f"{':white_check_mark:' if snap.volume_confirming else ':x:'} **Volume:** ve_rsi={snap.ve_rsi_raw:.1f}, vol={snap.volume_ratio:.1f}x"

    fields = [
        {"name": "Action", "value": f"{action_emoji} **{signal.action}**", "inline": True},
        {"name": "Ticker", "value": f"**{signal.ticker}**", "inline": True},
        {"name": "Position", "value": f"**{signal.position_pct:.0%}**", "inline": True},
        {"name": "Four Pillars", "value": f"{p1}\n{p2}\n{p3}\n{p4}", "inline": False},
        {"name": "Confidence", "value": f"{snap.pillars_confirming}/4 pillars ({snap.confidence:.0%})", "inline": True},
    ]

    if signal.stop_price:
        fields.append({"name": "Stop Loss", "value": f"${signal.stop_price:.2f}", "inline": True})

    embed = {
        "title": f"JK Bot Signal — {signal.action} {signal.ticker}",
        "description": signal.reason,
        "color": color,
        "fields": fields,
        "timestamp": datetime.now().isoformat(),
        "footer": {"text": "JK Four Pillars Trading Bot"},
    }

    return {"embeds": [embed]}


def format_discord_scan_embed(snapshots: list, prices: dict) -> dict:
    """Format a daily scan summary as a Discord embed.

    Snapshots are expected to be pre-ranked by FourPillarsEngine.rank_snapshots().
    The laggard_rank field (from J. Kornblatt's 2007 'Leaders vs Laggards' paper)
    shows cross-sectional priority: rank 1 = most washed-out = highest buy priority.
    """
    lines = []
    for snap in snapshots:
        price = prices.get(snap.ticker, 0)
        regime_emoji = {"bull": ":green_circle:", "chop": ":yellow_circle:", "bear": ":red_circle:"}.get(snap.regime, ":white_circle:")
        signal_emoji = {"STRONG_BUY": ":rocket:", "BUY": ":chart_with_upwards_trend:",
                        "HOLD": ":hourglass:", "REDUCE": ":warning:", "FLAT": ":zzz:"}.get(snap.signal_label, "")
        rank_str = f"[#{snap.laggard_rank}] " if snap.laggard_rank is not None else ""
        lines.append(
            f"{regime_emoji} {rank_str}**{snap.ticker}** ${price:.2f} — "
            f"{snap.signal_label} {signal_emoji} | "
            f"{snap.pillars_confirming}/4 pillars | "
            f"z={snap.z_hybrid_zscore:+.2f} | mmrsi={snap.multimac_rsi_score:+.1f}"
        )

    # Highlight the top laggard if it's actionable
    top_laggard = next((s for s in snapshots if s.laggard_rank == 1), None)
    footer_note = ""
    if top_laggard and top_laggard.signal_label in ("BUY", "STRONG_BUY"):
        footer_note = f"  ★ Top laggard {top_laggard.ticker} is actionable — highest cross-sectional priority"
    elif top_laggard:
        footer_note = f"  Top laggard: {top_laggard.ticker} (mmrsi={top_laggard.multimac_rsi_score:+.1f}) — watching for entry"

    embed = {
        "title": ":satellite: JK Bot — Daily Scan",
        "description": "\n".join(lines) + (f"\n\n{footer_note}" if footer_note else ""),
        "color": 0x3498db,  # blue
        "timestamp": datetime.now().isoformat(),
        "footer": {"text": "JK Four Pillars Bot  |  Rank = laggard priority (1=most washed-out)"},
    }

    return {"embeds": [embed]}


def format_discord_learning_embed(experiment: dict) -> dict:
    """Format a learning session summary as a Discord embed. Called once per run."""
    kept = experiment.get("kept", False)
    old_sharpe = experiment.get("old_sharpe", "?")
    new_sharpe = experiment.get("new_sharpe", "?")
    hypothesis = experiment.get("hypothesis", "No hypothesis")

    if kept:
        color = 0x2ecc71
        title = ":brain: AutoResearch — Improvements Found"
        delta_str = f"{old_sharpe:.4f} → **{new_sharpe:.4f}**" if isinstance(new_sharpe, float) and isinstance(old_sharpe, float) else f"{old_sharpe} → {new_sharpe}"
    else:
        color = 0x95a5a6
        title = ":brain: AutoResearch — Params Stable"
        delta_str = f"{old_sharpe:.4f} (unchanged)" if isinstance(old_sharpe, float) else str(old_sharpe)

    fields = [
        {"name": ":chart_with_upwards_trend: Composite Sharpe", "value": delta_str, "inline": False},
    ]

    changes = experiment.get("changes", {})
    if changes:
        changes_lines = "\n".join(f"  {k}: {v}" for k, v in changes.items())
        fields.append({"name": ":wrench: Parameters Updated", "value": f"```{changes_lines}```", "inline": False})

    new_params = experiment.get("new_params")
    if new_params:
        # Show only the key tunable params, not the settled/fixed ones
        skip = {"BULL_BASELINE", "CHOP_BASELINE", "BEAR_BASELINE",
                "STOP_LOSS_PCT", "TIME_STOP_DAYS", "TRAIL_ACTIVATE_PCT", "BEAR_THRESHOLD"}
        key_params = {k: v for k, v in new_params.items() if k not in skip}
        params_lines = "\n".join(f"  {k}: {v}" for k, v in sorted(key_params.items()))
        fields.append({"name": ":clipboard: Current Best Params (key)", "value": f"```{params_lines}```", "inline": False})

    fields.append({"name": ":memo: Summary", "value": hypothesis, "inline": False})

    embed = {
        "title": title,
        "color": color,
        "fields": fields,
        "timestamp": datetime.now().isoformat(),
        "footer": {"text": "JK AutoResearch — posted once per session"},
    }
    return {"embeds": [embed]}


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------

def alert_terminal(signal: TradeSignal):
    """Print alert to terminal."""
    print(format_alert(signal))


def alert_log(signal: TradeSignal):
    """Append alert to JSONL log file."""
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    snap = signal.snapshot
    record = {
        "timestamp": str(signal.timestamp),
        "ticker": signal.ticker,
        "action": signal.action,
        "position_pct": signal.position_pct,
        "reason": signal.reason,
        "regime": snap.regime,
        "trend_score": snap.trend_score_raw,
        "z_hybrid_z": round(snap.z_hybrid_zscore, 3),
        "timing": snap.timing_signal,
        "momentum_confirming": snap.momentum_confirming,
        "volume_confirming": snap.volume_confirming,
        "ve_rsi": round(snap.ve_rsi_raw, 1),
        "vol_ratio": round(snap.volume_ratio, 2),
        "confidence": snap.confidence,
        "pillars": snap.pillars_confirming,
    }
    with open(ALERT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def alert_macos(signal: TradeSignal):
    """Send macOS notification via osascript."""
    import subprocess
    title = f"JK Bot: {signal.action} {signal.ticker}"
    body = f"{signal.reason[:100]}"
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification "{body}" with title "{title}" sound name "Glass"'],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def alert_discord(signal: TradeSignal, url: Optional[str] = None):
    """Send alert to Discord webhook."""
    url = url or os.environ.get("JK_DISCORD_WEBHOOK")
    if not url:
        return

    import requests
    payload = format_discord_embed(signal)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Discord webhook failed: {e}")


def send_discord_scan(snapshots: list, prices: dict, url: Optional[str] = None):
    """Send daily scan summary to Discord."""
    url = url or os.environ.get("JK_DISCORD_WEBHOOK")
    if not url:
        return

    import requests
    payload = format_discord_scan_embed(snapshots, prices)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Discord scan webhook failed: {e}")


def send_trade_log(signal: TradeSignal):
    """Post a trade execution to the #trade-log channel via bot token."""
    token = os.environ.get("JK_DISCORD_BOT_TOKEN")
    channel_id = os.environ.get("JK_DISCORD_TRADELOG_CHANNEL")
    if not token or not channel_id:
        return

    import requests
    snap = signal.snapshot
    color_map = {
        "BUY": 0x2ecc71, "STRONG_BUY": 0x27ae60,
        "SELL": 0xe74c3c, "REDUCE": 0xf39c12,
        "STOP_LOSS": 0xe74c3c, "TRAIL_STOP": 0xe67e22,
        "TIME_STOP": 0x95a5a6, "HOLD": 0x95a5a6,
    }
    action_emoji = {
        "BUY": "📈", "STRONG_BUY": "🚀", "SELL": "📉",
        "REDUCE": "⚠️", "STOP_LOSS": "🛑", "TRAIL_STOP": "🔒",
        "TIME_STOP": "⏰", "HOLD": "⏳",
    }.get(signal.action, "🔔")

    embed = {
        "title": f"{action_emoji} {signal.action} — {signal.ticker}",
        "description": signal.reason,
        "color": color_map.get(signal.action, 0x95a5a6),
        "fields": [
            {"name": "Position", "value": f"{signal.position_pct:.0%}", "inline": True},
            {"name": "Regime", "value": snap.regime.upper(), "inline": True},
            {"name": "Confidence", "value": f"{snap.pillars_confirming}/4 pillars", "inline": True},
            {"name": "Timing (z)", "value": f"{snap.z_hybrid_zscore:+.2f}", "inline": True},
            {"name": "Momentum", "value": "✅" if snap.momentum_confirming else "❌", "inline": True},
            {"name": "Volume", "value": "✅" if snap.volume_confirming else "❌", "inline": True},
        ],
        "timestamp": datetime.now().isoformat(),
        "footer": {"text": "JK Paper Trader — Trade Log"},
    }
    if signal.stop_price:
        embed["fields"].append({"name": "Stop Loss", "value": f"${signal.stop_price:.2f}", "inline": True})

    try:
        resp = requests.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            json={"embeds": [embed]},
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  Trade log post failed: {e}")


def send_discord_learning(experiment: dict, url: Optional[str] = None):
    """Send learning experiment result to Discord."""
    url = url or os.environ.get("JK_DISCORD_WEBHOOK")
    if not url:
        return

    import requests
    payload = format_discord_learning_embed(experiment)
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Discord learning webhook failed: {e}")


# ---------------------------------------------------------------------------
# Unified alerter
# ---------------------------------------------------------------------------

def send_alerts(signal: TradeSignal, channels: list[str] = None):
    """
    Send alert through configured channels.
    Default: terminal + log + discord (if configured).
    """
    if channels is None:
        channels = ["terminal", "log"]
        if os.environ.get("JK_DISCORD_WEBHOOK"):
            channels.append("discord")

    # Only alert on actionable signals
    if signal.action == "HOLD":
        return

    # Always log to #trade-log channel if configured
    if os.environ.get("JK_DISCORD_TRADELOG_CHANNEL"):
        try:
            send_trade_log(signal)
        except Exception as e:
            print(f"  Trade log failed: {e}")

    dispatch = {
        "terminal": alert_terminal,
        "log": alert_log,
        "macos": alert_macos,
        "discord": alert_discord,
    }

    for ch in channels:
        fn = dispatch.get(ch)
        if fn:
            try:
                fn(signal)
            except Exception as e:
                print(f"  Alert channel '{ch}' failed: {e}")
