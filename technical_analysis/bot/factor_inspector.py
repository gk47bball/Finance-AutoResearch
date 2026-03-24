"""
Factor Inspector
================
Posts an AI-generated factor heatmap to Discord 3x daily (open, midday, close).
Factors are computed across 15 market ETFs and scored on 8 dimensions.

Factors evolve over time — underperforming ones get replaced by Claude based on
hit-rate tracking and current market regime. Every new or modified factor is
logged to the #factor-inspector Discord channel.

Schedule (via launchd):
  9:45 AM ET  — Opening read (markets have been open 15 min)
  12:30 PM ET — Midday read
  3:55 PM ET  — Closing read (5 min before close)

Run manually:
  python -m technical_analysis.bot.factor_inspector            # single scan + post
  python -m technical_analysis.bot.factor_inspector --evolve   # trigger factor evolution
  python -m technical_analysis.bot.factor_inspector --test     # dry run, no Discord post
"""

import io
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import requests

from technical_analysis.bot.llm_client import llm_chat, llm_chat_json
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

from technical_analysis.backtest.signal_tester import fetch_data

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEBHOOK_URL  = os.environ.get("JK_DISCORD_FACTOR_WEBHOOK", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

STATE_DIR  = Path(__file__).parent / "state"
REGISTRY_FILE = STATE_DIR / "factor_registry.json"
HITRATE_FILE  = STATE_DIR / "factor_hitrates.jsonl"

# Tickers to analyze — core indices + all 11 sector ETFs
TICKERS = [
    "SPY", "QQQ", "DIA", "IWM",                        # core
    "XLK", "XLF", "XLE", "XLV", "XLI",                 # sectors
    "XLC", "XLY", "XLP", "XLRE", "XLB", "XLU",         # sectors cont.
]

# Lookback needed for all factor computations (trading days)
DATA_LOOKBACK = "1y"

# ---------------------------------------------------------------------------
# Factor definitions — each factor produces a float score per ticker.
# Positive = bullish, Negative = bearish, |score| = conviction.
# These are z-scored cross-sectionally (across tickers) before display.
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    dm_plus  = (hi - hi.shift()).clip(lower=0)
    dm_minus = (lo.shift() - lo).clip(lower=0)
    dm_plus  = dm_plus.ewm(alpha=1/period, adjust=False).mean()
    dm_minus = dm_minus.ewm(alpha=1/period, adjust=False).mean()
    di_plus  = 100 * dm_plus / (atr + 1e-10)
    di_minus = 100 * dm_minus / (atr + 1e-10)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def compute_factor_scores(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute all factor scores for all tickers.
    Returns DataFrame: rows=tickers, cols=factors, values=raw scores.
    """
    spy_ret_21 = data["SPY"]["Close"].pct_change(21).iloc[-1] if "SPY" in data else 0

    records = {}
    for ticker, df in data.items():
        if df is None or len(df) < 63:
            continue
        close  = df["Close"]
        volume = df["Volume"]

        # F1: Short Momentum (5-day return, annualized %)
        mom5 = close.pct_change(5).iloc[-1] * (252 / 5) * 100

        # F2: Medium Momentum (21-day return, annualized %)
        mom21 = close.pct_change(21).iloc[-1] * (252 / 21) * 100

        # F3: Mean Reversion Opportunity (oversold = positive/bullish)
        #     RSI below 50 = oversold = positive score; above 50 = overbought = negative
        rsi_val = _rsi(close, 14).iloc[-1]
        mean_rev = -(rsi_val - 50) / 50 * 100   # range ~ -100 to +100; positive = oversold

        # F4: Volatility Regime (low/compressing vol = bullish = positive)
        #     Score = -(10d realized vol / 63d realized vol - 1) * 100
        rets = close.pct_change().dropna()
        vol10 = rets.iloc[-10:].std() * (252 ** 0.5) if len(rets) >= 10 else np.nan
        vol63 = rets.iloc[-63:].std() * (252 ** 0.5) if len(rets) >= 63 else np.nan
        vol_regime = -(vol10 / (vol63 + 1e-10) - 1) * 100 if vol63 else 0

        # F5: Volume Thrust (high vol on recent up day = bullish)
        vol5_avg  = volume.iloc[-5:].mean()
        vol20_avg = volume.iloc[-20:].mean()
        price_dir = 1 if close.pct_change(5).iloc[-1] > 0 else -1
        vol_thrust = price_dir * (vol5_avg / (vol20_avg + 1e-10) - 1) * 100

        # F6: Trend Quality (ADX — trend strength; trending = can ride momentum)
        #     High ADX in uptrend = bullish; high ADX in downtrend = bearish
        adx_val = _adx(df, 14).iloc[-1]
        trend_dir = 1 if mom21 > 0 else -1
        trend_quality = trend_dir * adx_val  # positive if trending up, negative if trending down

        # F7: Relative Strength vs SPY (alpha, 21d)
        rel_strength = (close.pct_change(21).iloc[-1] - spy_ret_21) * 100

        # F8: Recovery from 52-Week Low (buying off lows = bullish signal)
        high_52w = close.rolling(252).max().iloc[-1]
        low_52w  = close.rolling(252).min().iloc[-1]
        range_52w = high_52w - low_52w + 1e-10
        # Position in 52w range: 0 = at low, 100 = at high
        pos_in_range = (close.iloc[-1] - low_52w) / range_52w * 100
        # Stocks near lows but with improving momentum = contrarian opportunity
        recovery_score = pos_in_range - 50   # positive = above midpoint (bullish)

        records[ticker] = {
            "Momentum 5d":   mom5,
            "Momentum 21d":  mom21,
            "Mean Reversion": mean_rev,
            "Vol Regime":    vol_regime,
            "Vol Thrust":    vol_thrust,
            "Trend Quality": trend_quality,
            "Rel Strength":  rel_strength,
            "52w Recovery":  recovery_score,
        }

    df_scores = pd.DataFrame(records).T
    df_scores.index.name = "Ticker"
    return df_scores


def zscore_factors(raw: pd.DataFrame) -> pd.DataFrame:
    """Z-score each factor column cross-sectionally and clip to [-2.5, 2.5]."""
    z = (raw - raw.mean()) / (raw.std() + 1e-10)
    return z.clip(-2.5, 2.5)


# ---------------------------------------------------------------------------
# Heatmap image generation
# ---------------------------------------------------------------------------

def build_heatmap_image(
    z_scores: pd.DataFrame,
    raw_scores: pd.DataFrame,
    session_label: str,       # e.g. "OPEN", "MIDDAY", "CLOSE"
) -> bytes:
    """
    Build a color-coded factor heatmap and return it as PNG bytes.
    Rows = tickers, Cols = factors.
    Green = bullish / strong, Red = bearish / weak, White = neutral.
    """
    # Ticker ordering: sort by average z-score (best to worst)
    avg_z = z_scores.mean(axis=1).sort_values(ascending=False)
    z_ordered = z_scores.loc[avg_z.index]
    r_ordered = raw_scores.loc[avg_z.index]

    n_tickers = len(z_ordered)
    n_factors = len(z_ordered.columns)

    fig_width  = max(14, n_factors * 1.6)
    fig_height = max(7,  n_tickers * 0.55 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Custom diverging colormap: dark red → white → dark green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "factor_cmap",
        ["#c0392b", "#e74c3c", "#f8f9fa", "#27ae60", "#1a7a42"],
        N=256,
    )

    im = ax.imshow(
        z_ordered.values,
        cmap=cmap, vmin=-2.5, vmax=2.5,
        aspect="auto", interpolation="nearest",
    )

    # Cell annotations: raw score value
    for i, ticker in enumerate(z_ordered.index):
        for j, factor in enumerate(z_ordered.columns):
            raw_val = r_ordered.loc[ticker, factor]
            z_val   = z_ordered.loc[ticker, factor]
            text_color = "black" if abs(z_val) < 1.2 else "white"
            label = f"{raw_val:.1f}"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7.5, color=text_color, fontweight="bold")

    # Ticker row labels with composite z-score badge
    ax.set_yticks(range(n_tickers))
    ticker_labels = []
    for t in z_ordered.index:
        z_avg = z_ordered.loc[t].mean()
        arrow = "▲" if z_avg > 0.3 else ("▼" if z_avg < -0.3 else "●")
        ticker_labels.append(f"{arrow} {t}")
    ax.set_yticklabels(ticker_labels, fontsize=9, color="white", fontweight="bold")

    # Factor column labels
    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(
        [c.replace(" ", "\n") for c in z_ordered.columns],
        fontsize=8, color="#adb5bd", rotation=0, ha="center",
    )
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, n_factors, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_tickers, 1), minor=True)
    ax.grid(which="minor", color="#1e2730", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.ax.tick_params(colors="white", labelsize=8)
    cbar.set_label("Factor Z-Score", color="#adb5bd", fontsize=8)
    cbar.ax.set_facecolor("#0d1117")

    # Title
    now_et = datetime.now()
    fig.suptitle(
        f"JK Factor Inspector  ·  {session_label}  ·  {now_et.strftime('%b %d %Y  %I:%M %p')} ET",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# LLM interpretation
# ---------------------------------------------------------------------------

INTERPRETATION_PROMPT = """You are a veteran market strategist reviewing a real-time factor scan.

You will receive:
1. A table of factor z-scores for 15 ETFs (indices + sectors)
2. The top bullish and bearish readings

Your job: write a crisp market interpretation that a smart non-quant trader can act on.

Format your response as JSON:
{
  "headline": "one punchy sentence summarizing the dominant factor theme right now",
  "top_signals": [
    {
      "ticker": "XLK",
      "signal": "LONG" or "SHORT",
      "factor": "which factor is driving this",
      "layman": "1-2 sentences in plain English — what does this mean and why does it matter",
      "trade": "specific actionable idea — e.g. 'Buy XLK calls expiring next week on any dip to $210'"
    }
  ],
  "market_regime": "one sentence describing the current market factor regime",
  "watch": "one thing to watch that the factors are NOT yet showing but could emerge"
}

Include 3-5 top_signals. Be direct and specific. No hedging. No disclaimers."""


def generate_interpretation(
    z_scores: pd.DataFrame,
    raw_scores: pd.DataFrame,
    session_label: str,
) -> Optional[dict]:
    """Use Claude to generate a market interpretation from factor readings."""
    if not ANTHROPIC_KEY:
        return None

    # Build a readable summary of the top signals
    flat = []
    for ticker in z_scores.index:
        for factor in z_scores.columns:
            z = z_scores.loc[ticker, factor]
            raw = raw_scores.loc[ticker, factor]
            flat.append((abs(z), ticker, factor, z, raw))

    flat.sort(reverse=True)
    top_lines = "\n".join(
        f"  {ticker} / {factor}: z={z:+.2f}, raw={raw:.1f}"
        for _, ticker, factor, z, raw in flat[:20]
    )

    # Also include composite score per ticker
    composite = z_scores.mean(axis=1).sort_values(ascending=False)
    composite_lines = "\n".join(
        f"  {t}: composite z={v:+.2f}" for t, v in composite.items()
    )

    user_msg = (
        f"Session: {session_label}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M ET')}\n\n"
        f"TOP FACTOR SIGNALS (by absolute z-score):\n{top_lines}\n\n"
        f"COMPOSITE FACTOR SCORE BY TICKER (avg across all factors):\n{composite_lines}\n\n"
        "Generate the factor interpretation."
    )

    try:
        return llm_chat_json(
            system=INTERPRETATION_PROMPT,
            user=user_msg,
            max_tokens=1500,
            temperature=0.3,
        )
    except Exception as e:
        print(f"  [factor] LLM interpretation error: {e}")
    return None


# ---------------------------------------------------------------------------
# Factor evolution
# ---------------------------------------------------------------------------

EVOLUTION_PROMPT = """You are a quantitative researcher reviewing the Factor Inspector system.

Current active factors:
{factor_list}

Recent factor hit rates (% of correct directional calls over last 30 days):
{hit_rates}

Current market regime context:
{regime_context}

Your job: decide if any factors should be REPLACED, MODIFIED, or if new factors should be ADDED.

Rules:
- Replace factors with hit rate < 45% over 30+ observations
- Add factors that would capture something the current set misses
- Keep total factors at 8
- Factors must be computable from daily OHLCV data (no fundamentals, no options data)

Output as JSON:
{
  "changes": [
    {
      "action": "KEEP" | "REPLACE" | "ADD",
      "old_factor": "name of factor being replaced (if REPLACE)",
      "new_factor": "Name of New Factor",
      "formula_description": "how to compute it from OHLCV in plain English",
      "rationale": "why this factor is relevant now",
      "layman_description": "what this measures in simple terms"
    }
  ],
  "regime_summary": "one sentence on current market regime"
}

Only include non-KEEP entries in the changes array. If everything should stay, return empty changes array."""


def load_factor_registry() -> dict:
    """Load the factor registry (or return defaults)."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)

    # Default registry — matches the factors in compute_factor_scores()
    default = {
        "factors": [
            {"name": "Momentum 5d",    "description": "5-day price momentum (annualized). Measures short-term price trend — are buyers or sellers in control right now?", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Momentum 21d",   "description": "21-day price momentum (annualized). The medium-term trend — where has this ETF been heading over the past month?", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Mean Reversion", "description": "RSI-based mean reversion signal. Positive = oversold (potential bounce), Negative = overbought (potential pullback). Contra-trend opportunity indicator.", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Vol Regime",     "description": "Volatility regime: is short-term vol expanding or compressing vs the 63-day baseline? Compressing vol = calmer markets ahead (bullish). Expanding = uncertainty rising.", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Vol Thrust",     "description": "Volume thrust: is today's trading volume unusually high, and in which direction? High volume on up moves confirms the trend. High vol on down moves = selling pressure.", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Trend Quality",  "description": "ADX-based trend quality. High positive score = strong uptrend. High negative = strong downtrend. Near zero = choppy/sideways — no clear direction.", "created": datetime.now().isoformat(), "version": 1},
            {"name": "Rel Strength",   "description": "21-day return vs SPY (alpha). Which sectors are beating the market? Positive = outperforming SPY, Negative = lagging behind.", "created": datetime.now().isoformat(), "version": 1},
            {"name": "52w Recovery",   "description": "Position within 52-week price range. +50 = at 52-week high. -50 = at 52-week low. Shows where prices are relative to a full year of history — high scores suggest strength, low scores suggest potential deep value.", "created": datetime.now().isoformat(), "version": 1},
        ],
        "evolution_history": [],
        "last_evolved": None,
    }
    save_factor_registry(default)
    return default


def save_factor_registry(registry: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def evolve_factors(regime_context: str = ""):
    """Ask Claude to review and potentially evolve the active factor set."""
    registry = load_factor_registry()
    factor_list = "\n".join(
        f"  {i+1}. {f['name']}: {f['description']}"
        for i, f in enumerate(registry["factors"])
    )

    # Read recent hit rates from log
    hit_rates_text = "(insufficient history — first evolution run)"
    if HITRATE_FILE.exists():
        recent = []
        with open(HITRATE_FILE) as f:
            for line in f:
                try:
                    recent.append(json.loads(line))
                except Exception:
                    pass
        if recent:
            cutoff = datetime.now() - timedelta(days=30)
            recent = [r for r in recent if datetime.fromisoformat(r["timestamp"]) > cutoff]
            # Aggregate by factor
            factor_stats: dict[str, list] = {}
            for r in recent:
                for factor, correct in r.get("correct_calls", {}).items():
                    factor_stats.setdefault(factor, []).append(correct)
            lines = []
            for factor, calls in factor_stats.items():
                rate = sum(calls) / len(calls) * 100 if calls else 0
                lines.append(f"  {factor}: {rate:.0f}% ({len(calls)} obs)")
            hit_rates_text = "\n".join(lines) if lines else "(no logged observations yet)"

    prompt = EVOLUTION_PROMPT.format(
        factor_list=factor_list,
        hit_rates=hit_rates_text,
        regime_context=regime_context or "No regime context provided.",
    )

    try:
        result = llm_chat_json(
            system="",
            user=prompt,
            max_tokens=2000,
            temperature=0.5,
        )
        changes = result.get("changes", [])

        if not changes:
            print("  [factor] No factor changes needed")
            return

        # Apply changes to registry
        for change in changes:
            action = change.get("action")
            if action == "REPLACE":
                old = change.get("old_factor")
                for i, f in enumerate(registry["factors"]):
                    if f["name"] == old:
                        new_factor = {
                            "name": change["new_factor"],
                            "description": change.get("layman_description", ""),
                            "formula_description": change.get("formula_description", ""),
                            "created": datetime.now().isoformat(),
                            "version": f.get("version", 1) + 1,
                            "replaces": old,
                        }
                        registry["factors"][i] = new_factor
                        print(f"  [factor] REPLACED '{old}' → '{change['new_factor']}'")
                        break
            elif action == "ADD":
                if len(registry["factors"]) < 8:
                    registry["factors"].append({
                        "name": change["new_factor"],
                        "description": change.get("layman_description", ""),
                        "formula_description": change.get("formula_description", ""),
                        "created": datetime.now().isoformat(),
                        "version": 1,
                    })
                    print(f"  [factor] ADDED '{change['new_factor']}'")

        registry["last_evolved"] = datetime.now().isoformat()
        registry["evolution_history"].append({
            "timestamp": datetime.now().isoformat(),
            "changes": changes,
            "regime_summary": result.get("regime_summary", ""),
        })
        save_factor_registry(registry)

        # Post factor evolution log to Discord
        _post_factor_evolution_log(changes, result.get("regime_summary", ""))

    except Exception as e:
        print(f"  [factor] Evolution error: {e}")


def _post_factor_evolution_log(changes: list, regime_summary: str):
    """Post a factor change log to the Discord channel."""
    if not WEBHOOK_URL or not changes:
        return

    fields = []
    for c in changes:
        action = c.get("action", "?")
        emoji = {"REPLACE": "🔄", "ADD": "➕", "KEEP": "✅"}.get(action, "📝")
        name = c.get("new_factor", c.get("old_factor", "?"))
        value = c.get("layman_description") or c.get("formula_description") or c.get("rationale", "")
        if c.get("old_factor"):
            value = f"Replaces: **{c['old_factor']}**\n{value}"
        fields.append({
            "name": f"{emoji} {name}",
            "value": value[:1024],
            "inline": False,
        })

    payload = {"embeds": [{
        "title": "📊 Factor Inspector — Factor Evolution Update",
        "description": f"**Market Regime:** {regime_summary}",
        "color": 0x3498db,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "JK Factor Inspector — Factor registry updated"},
    }]}
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=15)
    except Exception as e:
        print(f"  [factor] Discord evolution post failed: {e}")


# ---------------------------------------------------------------------------
# Discord posting
# ---------------------------------------------------------------------------

def post_to_discord(
    img_bytes: bytes,
    interpretation: Optional[dict],
    session_label: str,
    z_scores: pd.DataFrame,
):
    """Post heatmap image + LLM interpretation to Discord."""
    if not WEBHOOK_URL:
        print("  [factor] No webhook URL set")
        return

    # Build embed
    now_str = datetime.now().strftime("%b %d  %I:%M %p ET")
    color   = {"OPEN": 0x3498db, "MIDDAY": 0xf39c12, "CLOSE": 0x9b59b6}.get(session_label, 0x2ecc71)

    embed: dict = {
        "title": f"📊 Factor Inspector — {session_label}  ·  {now_str}",
        "color": color,
        "image": {"url": "attachment://factor_heatmap.png"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "JK Factor Inspector  ·  Green = bullish  ·  Red = bearish  ·  White = neutral"},
    }

    fields = []

    if interpretation:
        embed["description"] = f"**{interpretation.get('headline', '')}**"

        # Top signals
        for sig in interpretation.get("top_signals", [])[:4]:
            ticker   = sig.get("ticker", "?")
            signal   = sig.get("signal", "?")
            dir_emoji = "📈" if signal == "LONG" else "📉"
            field_val = (
                f"{sig.get('layman', '')}\n"
                f"**Trade:** {sig.get('trade', '')}\n"
                f"`{signal}  ·  {sig.get('factor', '')}`"
            )
            fields.append({
                "name": f"{dir_emoji} **{ticker}** — {sig.get('factor', '')}",
                "value": field_val[:1020],
                "inline": False,
            })

        if interpretation.get("market_regime"):
            fields.append({
                "name": "🧭 Market Regime",
                "value": interpretation["market_regime"],
                "inline": False,
            })

        if interpretation.get("watch"):
            fields.append({
                "name": "👀 Watch For",
                "value": interpretation["watch"],
                "inline": False,
            })

    # Top 3 tickers by composite score
    composite = z_scores.mean(axis=1).sort_values(ascending=False)
    top3  = "  ".join(f"**{t}** ({v:+.1f})" for t, v in composite.head(3).items())
    bot3  = "  ".join(f"**{t}** ({v:+.1f})" for t, v in composite.tail(3).items())
    fields.append({"name": "🏆 Strongest", "value": top3,  "inline": True})
    fields.append({"name": "⚠️ Weakest",   "value": bot3,  "inline": True})

    embed["fields"] = fields

    # Post multipart: embed JSON + image file
    try:
        resp = requests.post(
            WEBHOOK_URL,
            data={"payload_json": json.dumps({"embeds": [embed]})},
            files={"files[0]": ("factor_heatmap.png", img_bytes, "image/png")},
            timeout=30,
        )
        resp.raise_for_status()
        print(f"  [factor] Posted {session_label} heatmap to Discord ✓")
    except Exception as e:
        print(f"  [factor] Discord post failed: {e}")


# ---------------------------------------------------------------------------
# Hit-rate logging (tracks prediction accuracy for factor evolution)
# ---------------------------------------------------------------------------

def log_signals_for_accuracy(z_scores: pd.DataFrame, session_label: str):
    """Log today's factor signals so we can check accuracy on the next trading day."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "session": session_label,
        "signals": {
            ticker: {
                factor: float(z_scores.loc[ticker, factor])
                for factor in z_scores.columns
            }
            for ticker in z_scores.index
        },
    }
    with open(HITRATE_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def determine_session() -> str:
    """Determine if this is an OPEN, MIDDAY, or CLOSE scan based on ET time."""
    hour = datetime.now().hour
    minute = datetime.now().minute
    t = hour * 60 + minute
    if t < 11 * 60:
        return "OPEN"
    elif t < 14 * 60:
        return "MIDDAY"
    else:
        return "CLOSE"


def run_factor_inspector(
    session_label: Optional[str] = None,
    dry_run: bool = False,
    evolve: bool = False,
) -> bool:
    """
    Main function: fetch data, compute factors, generate heatmap + interpretation, post.
    Returns True on success.
    """
    session = session_label or determine_session()
    print(f"  [factor] Starting {session} scan — {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Step 1: Fetch data for all tickers
    print(f"  [factor] Fetching data for {len(TICKERS)} tickers...")
    data = {}
    for ticker in TICKERS:
        try:
            df = fetch_data(ticker, period=DATA_LOOKBACK)
            if df is not None and len(df) >= 63:
                data[ticker] = df
        except Exception as e:
            print(f"    {ticker}: fetch failed — {e}")
        time.sleep(0.05)

    if len(data) < 5:
        print("  [factor] Too few tickers fetched — aborting")
        return False

    print(f"  [factor] {len(data)}/{len(TICKERS)} tickers loaded")

    # Step 2: Compute factors
    raw_scores = compute_factor_scores(data)
    z_scores   = zscore_factors(raw_scores)

    # Print top readings
    composite = z_scores.mean(axis=1).sort_values(ascending=False)
    print(f"  [factor] Top tickers: {', '.join(composite.head(3).index.tolist())}")
    print(f"  [factor] Weak tickers: {', '.join(composite.tail(3).index.tolist())}")

    # Step 3: Generate heatmap image
    print(f"  [factor] Generating heatmap...")
    img_bytes = build_heatmap_image(z_scores, raw_scores, session)

    # Step 4: LLM interpretation
    print(f"  [factor] Generating interpretation...")
    interp = generate_interpretation(z_scores, raw_scores, session)
    if interp:
        print(f"  [factor] Headline: {interp.get('headline', '')[:80]}")

    # Step 5: Post to Discord
    if not dry_run:
        post_to_discord(img_bytes, interp, session, z_scores)
        log_signals_for_accuracy(z_scores, session)
    else:
        print("  [factor] DRY RUN — skipping Discord post")
        # Save image locally for inspection
        out = STATE_DIR / f"factor_heatmap_{session.lower()}.png"
        out.write_bytes(img_bytes)
        print(f"  [factor] Saved heatmap to {out}")

    # Step 6: Check if it's time to evolve factors (every 7 days or when requested)
    if evolve:
        regime = interp.get("market_regime", "") if interp else ""
        print(f"  [factor] Running factor evolution...")
        evolve_factors(regime_context=regime)

    return True


# ---------------------------------------------------------------------------
# Post factor dictionary to Discord (run once on setup)
# ---------------------------------------------------------------------------

def post_factor_dictionary():
    """Post a detailed description of every factor to Discord as a reference embed."""
    registry = load_factor_registry()
    if not WEBHOOK_URL:
        return

    fields = []
    for f in registry["factors"]:
        fields.append({
            "name": f"📐 {f['name']}",
            "value": f['description'],
            "inline": False,
        })

    payload = {"embeds": [{
        "title": "📖 Factor Inspector — Factor Dictionary",
        "description": (
            "These are the 8 factors used to score each ETF 3x daily. "
            "Factors evolve over time — this log is updated automatically when the system adds or replaces a factor.\n\n"
            "**Color guide:** 🟩 Green = bullish signal  ·  🟥 Red = bearish  ·  ⬜ White = neutral\n"
            "**Z-Score:** each factor is normalized across all 15 tickers so you can compare apples-to-apples.\n"
            "**Composite:** average z-score across all 8 factors — the overall factor health of each ETF."
        ),
        "color": 0x2ecc71,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": "JK Factor Inspector — pinned reference"},
    }]}

    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=15)
        r.raise_for_status()
        print("  [factor] Factor dictionary posted to Discord ✓")
    except Exception as e:
        print(f"  [factor] Dictionary post failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JK Factor Inspector")
    parser.add_argument("--session", choices=["OPEN", "MIDDAY", "CLOSE"],
                        help="Override session label")
    parser.add_argument("--test",   action="store_true", help="Dry run — save image locally, no Discord post")
    parser.add_argument("--evolve", action="store_true", help="Run factor evolution after scan")
    parser.add_argument("--dict",   action="store_true", help="Post factor dictionary to Discord and exit")
    args = parser.parse_args()

    if args.dict:
        post_factor_dictionary()
        sys.exit(0)

    run_factor_inspector(
        session_label=args.session,
        dry_run=args.test,
        evolve=args.evolve,
    )
