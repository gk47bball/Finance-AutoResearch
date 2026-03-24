"""
Trump Alpha Bot
===============
Polls Trump's Truth Social account every 15 seconds and uses Claude to
identify market-moving posts, extract the pure-play trade, post to Discord,
and — during market hours — automatically place Alpaca paper trades.

What gets flagged (HIGH signal):
  - Tariff announcements or threats (sector/country-specific)
  - Specific company mentions (positive → long, threat → short)
  - Trade deal language (winners and losers)
  - Fed/interest rate commentary
  - Energy policy (fossil vs. clean)
  - Sanctions / export controls
  - Government contract references
  - Military / geopolitical events (Iran, Russia, Taiwan, etc.)

Setup:
  1. Create a free Truth Social account at https://truthsocial.com
  2. Add to .env:
       TRUTHSOCIAL_USERNAME=your_username
       TRUTHSOCIAL_PASSWORD=your_password
       JK_DISCORD_TRUMP_WEBHOOK=https://discord.com/api/webhooks/...  (already set)
       ALPACA_API_KEY=...   ALPACA_API_SECRET=...   (already set)
  3. Schedule with launchd — see com.jkbot.trump-watcher.plist

Run manually:
  python -m technical_analysis.bot.trump_watcher              # single poll
  python -m technical_analysis.bot.trump_watcher --daemon     # continuous loop
  python -m technical_analysis.bot.trump_watcher --backtest 60  # simulate last 60 days
  python -m technical_analysis.bot.trump_watcher --positions  # show open paper positions
  python -m technical_analysis.bot.trump_watcher --close-all  # emergency close all positions
"""

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import requests

from technical_analysis.bot.llm_client import llm_chat_json, llm_chat_json_array


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRUMP_HANDLE = "realDonaldTrump"
POLL_INTERVAL_SECONDS = 15        # 15 seconds — worst-case lag ~17s (poll + LLM + post)
MAX_POST_AGE_MINUTES  = 10        # ignore posts older than this (catches up gracefully)
STATE_FILE = Path(__file__).parent / "state" / "trump_watcher_state.json"

# ---------------------------------------------------------------------------
# Alpaca paper trading config
# ---------------------------------------------------------------------------

ALPACA_PAPER_BASE   = "https://paper-api.alpaca.markets/v2"
_ALPACA_KEY         = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET      = os.environ.get("ALPACA_API_SECRET", "")

NOTIONAL_KNEE_JERK  = 500    # $500 per knee-jerk trade
NOTIONAL_TREND      = 1000   # $1000 per trend trade
KNEE_JERK_HOURS     = 4      # close knee-jerk after 4 hours
TREND_TRADING_DAYS  = 5      # close trend trade after 5 trading days
STOP_LOSS_PCT       = 0.02   # 2% stop loss triggers close
TAKE_PROFIT_PCT     = 0.04   # 4% take-profit triggers close (captures leveraged ETF spikes)
SLIPPAGE_PCT        = 0.002  # 0.2% per-leg estimated slippage (bid-ask on leveraged ETFs)

# Signal quality / deduplication
CONFIDENCE_THRESHOLD   = 6   # minimum LLM confidence score (1-10) to trade
SIGNAL_COOLDOWN_HOURS  = 4   # suppress duplicate (ticker, direction) signals within this window
MAX_POSITIONS_PER_TICKER = 1 # max simultaneous open positions per ticker

# Keyword pre-filter for backtest (reduces LLM calls; does NOT affect live bot)
_BACKTEST_KEYWORDS = [
    "tariff", "tax", "trade", "china", "deal", "fed", "rate",
    "oil", "energy", "drill", "iran", "military", "sanction",
    "bitcoin", "crypto", "bank", "stock", "market", "wall street",
    "nato", "russia", "ukraine", "israel", "taiwan", "chip",
    "semiconductor", "export", "import", "trillion", "billion",
    "percent", "%", "opec", "gold", "inflation", "dollar",
    "tariffs", "mexico", "canada", "eu", "europe", "nuclear",
]

# ---------------------------------------------------------------------------
# Leverage config — swap underlying signal tickers to leveraged ETF equivalents
# ---------------------------------------------------------------------------

USE_LEVERAGE = True   # set False to trade underlying ETF directly

# Maps signal ticker → {long_etf, short_etf, multiplier}
# SHORT signals automatically route to the inverse ETF (always buy, no short-selling)
LEVERAGE_MAP: dict = {
    "USO": {"long_etf": "UCO",  "short_etf": "SCO",  "multiplier": 2},
    "XLE": {"long_etf": "ERX",  "short_etf": "ERY",  "multiplier": 2},
    "TLT": {"long_etf": "TMF",  "short_etf": "TMV",  "multiplier": 3},
    "GLD": {"long_etf": "UGL",  "short_etf": "GLL",  "multiplier": 2},
    "SPY": {"long_etf": "SSO",  "short_etf": "SDS",  "multiplier": 2},
    "QQQ": {"long_etf": "QLD",  "short_etf": "QID",  "multiplier": 2},
    "IWM": {"long_etf": "UWM",  "short_etf": "TWM",  "multiplier": 2},
    "XLF": {"long_etf": "FAS",  "short_etf": "FAZ",  "multiplier": 3},
    # Pass-throughs: no clean leveraged product — trade underlying directly
    "LMT": None, "ITA": None, "XLI": None, "UUP": None,
}

# ---------------------------------------------------------------------------
# AutoResearch config
# ---------------------------------------------------------------------------

TRUMP_POST_CACHE  = Path(__file__).parent / "state" / "trump_post_cache.json"
TRUMP_PARAMS_FILE = Path(__file__).parent / "state" / "trump_best_params.json"
TRUMP_LEARN_LOG   = Path(__file__).parent / "state" / "trump_learn_log.jsonl"

TRUMP_PARAM_BOUNDS: dict = {
    "stop_loss_pct":        (0.01, 0.05),  # stop on the leveraged ETF price
    "take_profit_pct":      (0.02, 0.10),  # take-profit on the leveraged ETF price
    "cooldown_hours":       (1, 24),        # int — min hours between same (ticker, direction) signals
    "confidence_threshold": (5, 9),         # int — minimum LLM confidence score to trade
}

TRUMP_DEFAULT_PARAMS: dict = {
    "stop_loss_pct":        0.018,  # from prior AutoResearch best
    "take_profit_pct":      0.04,   # 4% take-profit on leveraged ETF
    "cooldown_hours":       4,
    "confidence_threshold": 6,
}

TRUMP_AUTOLEARN_SYSTEM_PROMPT = """You are optimizing a Trump Truth Social trade bot.
The bot places directional paper trades (via leveraged ETFs) when Trump posts market-moving content.

PARAMETERS:
  stop_loss_pct:        0.01–0.05  (fraction, e.g. 0.02 = 2%). Stop on the leveraged ETF price.
                        Tighter stops cut losses but may exit good trades early on volatility.
  take_profit_pct:      0.02–0.10  (fraction, e.g. 0.04 = 4%). Take-profit on the leveraged ETF.
                        Captures spikes on leveraged ETFs before mean-reversion. Higher = let winners run.
                        Key insight: leveraged ETFs spike then fade; taking profit early locks in gains.
  cooldown_hours:       1–24 (int). Minimum hours between two trades in the same (ticker, direction).
                        Prevents re-entering duplicate signals from clustered posts on the same topic.
                        Lower = more trades (more noise); higher = fewer, more independent bets.
  confidence_threshold: 5–9 (int). Only trade signals where LLM confidence ≥ this value.
                        Higher threshold = fewer trades but higher quality; lower = more trades.

OBJECTIVE: maximize avg_ret = average directional return across all INDEPENDENT (deduplicated) trades.
  - Stop-loss hits count as -stop_loss_pct (real cost) minus slippage
  - Take-profit hits count as +take_profit_pct (locked in early) minus slippage
  - cooldown_hours deduplicates clustered posts (same ticker/direction) within that window
  - confidence_threshold filters out ambiguous signals

RULES:
  1. Return ONLY a JSON array of exactly {n} objects.
  2. Format: [{{"changes": {{"param": value}}, "hypothesis": "one sentence"}}, ...]
  3. Each proposal must test a DIFFERENT parameter or direction than the others.
  4. Avoid repeating combinations that already failed (provided in history).
  5. Keep values within the stated bounds. cooldown_hours and confidence_threshold must be integers.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_seen_id": None, "posts_processed": 0, "alerts_sent": 0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Truth Social polling
# ---------------------------------------------------------------------------

def get_api():
    """Return an authenticated truthbrush Api instance."""
    from truthbrush.api import Api
    username = os.environ.get("TRUTHSOCIAL_USERNAME")
    password = os.environ.get("TRUTHSOCIAL_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD must be set in .env\n"
            "Create a free account at https://truthsocial.com"
        )
    return Api(username=username, password=password)


def fetch_new_posts(since_id: Optional[str], verbose: bool = True) -> list[dict]:
    """
    Fetch Trump's latest Truth Social posts.
    Returns only posts newer than since_id (or posts from last MAX_POST_AGE_MINUTES if since_id is None).
    Posts are returned in reverse-chronological order (newest first).
    """
    try:
        api = get_api()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=MAX_POST_AGE_MINUTES)

        posts = []
        for post in api.pull_statuses(
            TRUMP_HANDLE,
            replies=False,
            since_id=since_id,
        ):
            # truthbrush already stops at since_id, but double-check age on first run
            try:
                post_dt_str = post.get("created_at", "")
                from dateutil import parser as date_parse
                _parsed = date_parse.parse(post_dt_str)
                post_dt = _parsed.replace(tzinfo=timezone.utc) if _parsed.tzinfo is None else _parsed.astimezone(timezone.utc)
                if since_id is None and post_dt < cutoff:
                    break  # first run: only process recent posts
            except Exception:
                pass

            posts.append(post)

        if verbose and posts:
            print(f"  [trump] {len(posts)} new post(s) found")

        return posts

    except Exception as e:
        print(f"  [trump] Error fetching posts: {e}")
        return []


# ---------------------------------------------------------------------------
# LLM analysis
# ---------------------------------------------------------------------------

TRUMP_SYSTEM_PROMPT = """You are a veteran macro trader who has spent years trading around Trump's market-moving statements.

Your job: read a Trump Truth Social post and determine if it is market-moving. If it is, identify the best pure-play trade.

MARKET-MOVING CONTENT (flag ALL of these — when in doubt, flag it):
  - Tariff announcements/threats/rollbacks (any %, any country)
  - Specific company name + positive or negative sentiment
  - Trade deal language (new deal, deal breakdown, sanctions)
  - Federal Reserve / interest rate criticism or praise
  - Energy policy (drill/mine = fossil fuels long; attack renewables = solar/wind short)
  - Export controls, chip restrictions, technology bans
  - Government spending announcements or cuts
  - Country-specific economic actions (China, EU, Canada, Mexico, etc.)
  - MILITARY / GEOPOLITICAL — pick the BEST single pure-play based on the specific mechanism:
      * Strait of Hormuz closure threat / oil supply disruption → USO (oil supply directly)
      * US military strike / direct attack on Iran → GLD (fear bid) or ITA (defense spending)
      * De-escalation / ceasefire / "winding down" → USO SHORT (oil falls), ITA SHORT
      * Defense contract announcements / troop deployments → ITA or LMT (defense prime)
      * Broader geopolitical instability without specific commodity link → GLD (safe haven)
      * Russia/Ukraine escalation → GLD LONG (safe haven), XLE LONG (energy disruption)
      * China/Taiwan tension → QQQ SHORT (tech supply chain) or GLD LONG
      * Nuclear threat language → GLD LONG (maximum fear premium)
      * DO NOT default to USO for every military post — only if oil supply is directly at risk

TICKER SELECTION GUIDE — pick the most direct link, not the catchall:
  USO  → oil supply disruption (Hormuz, OPEC, sanctions on major oil producers)
  XLE  → broad energy policy (drill permits, pipeline approvals, clean energy attacks)
  GLD  → fear/uncertainty premium (war uncertainty, nuclear threats, systemic risk)
  TLT  → interest rate signals (Fed criticism, recession signals, flight to bonds)
  ITA  → defense spending (military action, contract news, NATO commitments)
  SPY  → broad economic signal (major trade deals, sweeping tariffs, GDP impact)
  QQQ  → tech-specific (chip bans, export controls, tech company mentions)
  UUP  → dollar strength (tariff threats that strengthen USD, safe haven dollar flows)

Examples of posts that ARE market-moving (do NOT skip these):
  - "The United States has blown Iran off of the map" → GLD LONG (fear) or ITA LONG (defense)
  - "Winding down our great military effort" → USO SHORT (oil supply normalizes)
  - "If Iran doesn't open the Strait of Hormuz within 48 hours..." → USO LONG (supply disruption)
  - "We are imposing 25% tariffs on Canada" → SPY SHORT or UUP LONG
  - "Proud to announce a great deal with China" → SPY LONG or QQQ LONG
  - "The Fed must cut rates NOW" → TLT LONG
  - "Drill baby drill — we are opening federal lands" → XLE LONG

NOT MARKET-MOVING (respond with SKIP only for these):
  - Political attacks on individuals with zero policy content (Mueller, Democrats generally)
  - Cultural commentary, patriotism, sports, personal nostalgia
  - Immigration rhetoric with no specific new policy action announced
  - Reposts of others' content with no new signal from Trump himself
  - Purely legal/criminal case commentary with no market angle

OUTPUT FORMAT (JSON, for market-moving posts):
{
  "market_moving": true,
  "urgency": "KNEE_JERK" or "TREND",
  "direction": "LONG" or "SHORT" or "MIXED",
  "primary_ticker": "USO",
  "secondary_tickers": ["XLE", "GLD", "LMT"],
  "thesis": "one sentence — the exact mechanism linking the post to the trade",
  "risk": "one sentence — what could make this wrong / fade quickly",
  "category": "TARIFF" | "COMPANY_MENTION" | "TRADE_DEAL" | "FED" | "ENERGY" | "TECH_EXPORT" | "SPENDING" | "SANCTIONS" | "MILITARY" | "OTHER",
  "confidence": 7
}

Confidence guide (1-10):
  9-10: Explicit, unambiguous policy action with clear directional market impact
  7-8:  Strong signal with minor ambiguity (e.g. threat not yet confirmed)
  5-6:  Moderate signal — market impact likely but size/duration uncertain
  3-4:  Weak signal — speculative or easily reversible
  1-2:  Very speculative — mostly noise

Urgency guide:
  KNEE_JERK = act within 30-60 minutes; headline-driven, typically fades within a day
  TREND     = structural impact; may take days-weeks to fully price in

For posts with NO market impact: respond with exactly: SKIP

Default to flagging, not skipping. Missing a real trade is more costly than one false positive."""


def analyze_post(post_text: str, post_url: str) -> Optional[dict]:
    """
    Use local Ollama (Qwen 3 4B) to analyze a Trump post for market impact.
    Falls back to Anthropic if Ollama is unavailable.
    Returns a dict with trade details, or None if not market-moving.
    """
    try:
        data = llm_chat_json(
            system=TRUMP_SYSTEM_PROMPT,
            user=f'Trump just posted:\n\n"{post_text}"',
            max_tokens=800,
            temperature=0.3,
        )

        if not data.get("market_moving"):
            return None

        # Ensure confidence is an int 1-10 (default 7 if missing)
        try:
            data["confidence"] = max(1, min(10, int(data.get("confidence", 7))))
        except (TypeError, ValueError):
            data["confidence"] = 7
        return data

    except Exception as e:
        print(f"  [trump] LLM error: {e}")
        return None


# ---------------------------------------------------------------------------
# Discord formatting
# ---------------------------------------------------------------------------

CATEGORY_EMOJI = {
    "TARIFF":          "🚢",
    "COMPANY_MENTION": "🏢",
    "TRADE_DEAL":      "🤝",
    "FED":             "🏦",
    "ENERGY":          "⚡",
    "TECH_EXPORT":     "💾",
    "SPENDING":        "💰",
    "SANCTIONS":       "🚫",
    "MILITARY":        "🪖",
    "OTHER":           "📢",
}

URGENCY_EMOJI = {
    "KNEE_JERK": "⚡ KNEE-JERK",
    "TREND":     "📈 TREND",
}

DIRECTION_COLOR = {
    "LONG":  0x2ecc71,   # green
    "SHORT": 0xe74c3c,   # red
    "MIXED": 0xf39c12,   # orange
}


def format_trump_embed(post: dict, analysis: dict) -> dict:
    """Format a Trump post + trade analysis as a Discord embed."""
    post_text = post.get("content", "")
    # Strip HTML tags from Truth Social content
    import re
    post_text = re.sub(r"<[^>]+>", "", post_text).strip()

    post_time = post.get("created_at", "")
    post_url  = f"https://truthsocial.com/@realDonaldTrump/{post['id']}"

    category  = analysis.get("category", "OTHER")
    urgency   = analysis.get("urgency", "TREND")
    direction = analysis.get("direction", "LONG")
    primary   = analysis.get("primary_ticker", "?")
    secondary = analysis.get("secondary_tickers", [])
    thesis    = analysis.get("thesis", "")
    risk      = analysis.get("risk", "")

    cat_emoji     = CATEGORY_EMOJI.get(category, "📢")
    urgency_label = URGENCY_EMOJI.get(urgency, urgency)
    dir_emoji     = {"LONG": "📈", "SHORT": "📉", "MIXED": "↕️"}.get(direction, "")
    color         = DIRECTION_COLOR.get(direction, 0x95a5a6)

    tickers_str = f"**{primary}**"
    if secondary:
        tickers_str += "  |  " + "  ".join(secondary)

    # Resolve leveraged ETF for display
    trade_ticker, multiplier, always_long = _resolve_leveraged_ticker(primary, direction)
    lev_line = ""
    if multiplier > 1:
        inv_note  = " (inverse ETF)" if always_long else ""
        lev_line  = f"\n⚡ Trading **{trade_ticker}** ({multiplier}x){inv_note}"

    fields = [
        {
            "name": f"{dir_emoji} Trade",
            "value": f"{tickers_str}\n`{direction}  ·  {urgency_label}`{lev_line}",
            "inline": False,
        },
        {
            "name": "📌 Thesis",
            "value": thesis,
            "inline": False,
        },
        {
            "name": "⚠️ Risk",
            "value": risk,
            "inline": False,
        },
        {
            "name": "🇺🇸 Post",
            "value": f"> {post_text[:400]}{'...' if len(post_text) > 400 else ''}\n[View on Truth Social]({post_url})",
            "inline": False,
        },
    ]

    return {"embeds": [{
        "title": f"{cat_emoji} Trump Alert — {category.replace('_', ' ')}",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": f"JK Trump Alpha  ·  {post_time[:19].replace('T', ' ')} UTC"},
    }]}


def post_to_discord(payload: dict, url: str):
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Alpaca paper trading
# ---------------------------------------------------------------------------

def _alpaca_headers() -> dict:
    return {
        "APCA-API-KEY-ID":     _ALPACA_KEY,
        "APCA-API-SECRET-KEY": _ALPACA_SECRET,
    }


def _alpaca_available() -> bool:
    return bool(_ALPACA_KEY and _ALPACA_SECRET and _ALPACA_KEY != "your_key_here")


def _resolve_leveraged_ticker(signal_ticker: str, direction: str):
    """
    Returns (trade_ticker, multiplier, always_long).

    trade_ticker:  the ETF to actually buy (e.g. "UCO" for USO LONG, "SCO" for USO SHORT)
    multiplier:    leverage factor (1 = no leverage, 2 = 2x, 3 = 3x)
    always_long:   True when a SHORT signal was converted to a long buy of the inverse ETF
    """
    if not USE_LEVERAGE:
        return signal_ticker, 1, False
    entry = LEVERAGE_MAP.get(signal_ticker.upper())
    if entry is None:   # pass-through (unknown or explicitly unmapped)
        return signal_ticker, 1, False
    if direction == "LONG":
        return entry["long_etf"], entry["multiplier"], False
    else:  # SHORT → buy inverse ETF
        return entry["short_etf"], entry["multiplier"], True


def is_market_hours() -> bool:
    """Return True if NYSE is currently open (9:30–16:00 ET, Mon–Fri)."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        # Fallback: approximate EDT (Mar–Nov) vs EST (Nov–Mar)
        now_utc = datetime.now(timezone.utc)
        month = now_utc.month
        is_edt = 3 <= month <= 10  # rough EDT range (exact DST dates vary)
        et = timezone(timedelta(hours=-4)) if is_edt else timezone(timedelta(hours=-5))
    now_et = datetime.now(et)
    if now_et.weekday() >= 5:         # Saturday=5, Sunday=6
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et < market_close


def place_paper_trade(analysis: dict, post_id: str, verbose: bool = True) -> Optional[dict]:
    """
    Place a market paper trade on Alpaca based on Trump alert analysis.
    Only called during market hours. Returns position info dict or None.
    """
    if not _alpaca_available():
        if verbose:
            print("  [alpaca] No API keys configured — skipping paper trade")
        return None

    ticker    = analysis.get("primary_ticker", "").upper()
    direction = analysis.get("direction", "")
    urgency   = analysis.get("urgency", "TREND")

    if not ticker or direction == "MIXED":
        if verbose:
            print(f"  [alpaca] Skipping paper trade — ticker={ticker!r}, direction={direction!r}")
        return None

    # Resolve to leveraged ETF (SHORT → buy inverse ETF so side is always "buy")
    trade_ticker, multiplier, always_long = _resolve_leveraged_ticker(ticker, direction)
    base_notional = NOTIONAL_KNEE_JERK if urgency == "KNEE_JERK" else NOTIONAL_TREND
    # ── Confidence-based position sizing ─────────────────────────────────
    # Scale notional by confidence / 7.0, clamped to 0.5x – 1.5x
    confidence     = analysis.get("confidence", 7)
    conf_scale     = max(0.5, min(1.5, confidence / 7.0))
    notional       = round(base_notional * conf_scale, 2)

    # ── Regime awareness: consult Four Pillars SPY regime ────────────────
    # BEAR regime: skip TREND signals entirely (market headwind)
    # CHOP regime: reduce TREND notional by 50%
    if urgency == "TREND":
        try:
            from technical_analysis.bot.pillars import FourPillarsEngine
            _spy_engine = FourPillarsEngine()
            _spy_snap   = _spy_engine.compute("SPY")
            spy_regime  = getattr(_spy_snap, "regime", "CHOP")
            if spy_regime == "BEAR":
                if verbose:
                    print(f"  [alpaca] Skipping TREND trade — SPY regime is BEAR")
                return None
            elif spy_regime == "CHOP":
                notional = round(notional * 0.5, 2)
                if verbose:
                    print(f"  [alpaca] SPY regime CHOP — halving TREND notional to ${notional:.0f}")
        except Exception as _regime_err:
            if verbose:
                print(f"  [alpaca] Regime check failed ({_regime_err}) — proceeding without adjustment")

    def _place_order(symbol: str, use_notional: bool) -> dict:
        body = {
            "symbol":        symbol,
            "side":          "buy",
            "type":          "market",
            "time_in_force": "day",
        }
        if use_notional:
            body["notional"] = str(notional)
        else:
            body["qty"] = "1"  # fallback for non-fractional ETFs
        return requests.post(
            f"{ALPACA_PAPER_BASE}/orders",
            headers=_alpaca_headers(),
            json=body,
            timeout=10,
        )

    try:
        resp = _place_order(trade_ticker, use_notional=True)
        # Alpaca returns 422 when notional orders aren't supported for the symbol
        if resp.status_code in (400, 422):
            resp = _place_order(trade_ticker, use_notional=False)
        resp.raise_for_status()
        order_data = resp.json()

        lev_label = f"({multiplier}x)" if multiplier > 1 else ""
        position_info = {
            "ticker":              trade_ticker,   # what was actually ordered
            "signal_ticker":       ticker,         # original signal (e.g. "USO")
            "leverage_multiplier": multiplier,
            "direction":           direction,      # original signal direction preserved
            "urgency":             urgency,
            "order_id":            order_data.get("id"),
            "entry_time":          datetime.now(timezone.utc).isoformat(),
            "notional":            notional,
            "confidence":          confidence,
            "post_id":             post_id,
            "thesis":              analysis.get("thesis", "")[:120],
        }

        if verbose:
            oid = (order_data.get("id") or "?")[:8]
            print(f"  [alpaca] Paper trade placed: BUY ${notional} {trade_ticker}{lev_label}  "
                  f"({'inverse ETF for ' if always_long else ''}{direction} {ticker})  (order {oid})")

        return position_info

    except Exception as e:
        if verbose:
            print(f"  [alpaca] Order failed for {trade_ticker}: {e}")
        return None


def get_position_pnl(ticker: str) -> Optional[float]:
    """Fetch unrealized P&L % for a position (-0.03 = -3%). Returns None on error."""
    try:
        resp = requests.get(
            f"{ALPACA_PAPER_BASE}/positions/{ticker}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if resp.status_code == 404:
            return None  # already closed
        resp.raise_for_status()
        return float(resp.json().get("unrealized_plpc", 0))
    except Exception:
        return None


def close_position(ticker: str, verbose: bool = True) -> bool:
    """Close an Alpaca paper position. Returns True on success."""
    try:
        resp = requests.delete(
            f"{ALPACA_PAPER_BASE}/positions/{ticker}",
            headers=_alpaca_headers(),
            timeout=10,
        )
        if resp.status_code in (200, 204):
            if verbose:
                print(f"  [alpaca] Closed position: {ticker}")
            return True
        return False
    except Exception as e:
        if verbose:
            print(f"  [alpaca] Failed to close {ticker}: {e}")
        return False


def check_and_close_positions(verbose: bool = True) -> list:
    """
    Review all open paper positions. Close any that have hit their time stop
    or the 2% stop loss. Returns list of closed position dicts.
    """
    if not _alpaca_available():
        return []

    state          = load_state()
    open_positions = state.get("paper_positions", [])
    if not open_positions:
        return []

    now_utc  = datetime.now(timezone.utc)
    closed   = []
    still_open = []

    for pos in open_positions:
        try:
            entry_time   = datetime.fromisoformat(pos["entry_time"])
        except Exception:
            still_open.append(pos)
            continue

        ticker         = pos["ticker"]
        direction      = pos["direction"]
        urgency        = pos.get("urgency", "TREND")

        # Time stop
        elapsed_hours  = (now_utc - entry_time).total_seconds() / 3600
        limit_hours    = KNEE_JERK_HOURS if urgency == "KNEE_JERK" else TREND_TRADING_DAYS * 24
        time_expired   = elapsed_hours >= limit_hours

        # P&L stop
        pnl_pct  = get_position_pnl(ticker)
        if pnl_pct is None:
            # Position no longer exists on Alpaca — mark as externally closed
            closed.append({**pos, "close_reason": "already_closed", "pnl_pct": None})
            continue

        stop_hit   = pnl_pct <= -STOP_LOSS_PCT
        profit_hit = pnl_pct >= TAKE_PROFIT_PCT

        if time_expired or stop_hit or profit_hit:
            if profit_hit:
                reason = "take_profit"
            elif time_expired:
                reason = "time_stop"
            else:
                reason = "stop_loss"
            if close_position(ticker, verbose=verbose):
                if verbose:
                    pct_str = f"{pnl_pct*100:+.1f}%" if pnl_pct is not None else "?"
                    print(f"  [alpaca] {ticker} closed ({reason})  P&L: {pct_str}")
                closed_pos = {**pos, "close_reason": reason, "pnl_pct": pnl_pct}
                closed.append(closed_pos)
                # Post close notification to Discord
                _post_position_close_embed(closed_pos, pnl_pct, reason, entry_time)
            else:
                still_open.append(pos)
        else:
            still_open.append(pos)

    if closed:
        state["paper_positions"]  = still_open
        all_closed = state.get("closed_positions", []) + closed
        # Cap to last 200 entries to prevent unbounded state growth
        if len(all_closed) > 200:
            all_closed = all_closed[-200:]
        state["closed_positions"] = all_closed

        # Update cumulative realized P&L
        realized_pnl = sum(
            p["pnl_pct"] * p.get("notional", 0)
            for p in all_closed
            if p.get("pnl_pct") is not None
        )
        state["cumulative_realized_pnl"] = realized_pnl
        save_state(state)

    return closed


def _post_position_close_embed(pos: dict, pnl_pct: Optional[float],
                               reason: str, entry_time: datetime):
    """Post a Discord embed when a paper position is closed."""
    webhook_url = os.environ.get("JK_DISCORD_TRUMP_WEBHOOK")
    if not webhook_url:
        return

    ticker    = pos.get("ticker", "?")
    signal    = pos.get("signal_ticker", ticker)
    direction = pos.get("direction", "?")
    urgency   = pos.get("urgency", "TREND")
    notional  = pos.get("notional", 0)
    thesis    = pos.get("thesis", "")[:120]

    now_utc   = datetime.now(timezone.utc)
    hold_h    = (now_utc - entry_time).total_seconds() / 3600

    pnl_str   = f"{pnl_pct*100:+.1f}%" if pnl_pct is not None else "?"
    pnl_dollar = f"${pnl_pct * notional:+.2f}" if pnl_pct is not None else "?"

    reason_labels = {
        "take_profit": "✅ Take-Profit Hit",
        "stop_loss":   "🛑 Stop-Loss Hit",
        "time_stop":   "⏱️ Time Stop",
    }
    reason_label = reason_labels.get(reason, reason)

    color = (0x2ecc71 if pnl_pct is not None and pnl_pct > 0
             else (0xe74c3c if pnl_pct is not None and pnl_pct < 0 else 0xf39c12))

    embed = {
        "title":  f"🔔 Position Closed — {ticker}  ({reason_label})",
        "color":  color,
        "fields": [
            {
                "name":   "Trade",
                "value":  f"**{ticker}** · {direction} · {urgency}  (signal: {signal})",
                "inline": True,
            },
            {
                "name":   "P&L",
                "value":  f"`{pnl_str}`  ({pnl_dollar} on ${notional:.0f})",
                "inline": True,
            },
            {
                "name":   "Hold Duration",
                "value":  f"{hold_h:.1f} hours",
                "inline": True,
            },
            {
                "name":   "Thesis",
                "value":  thesis or "—",
                "inline": False,
            },
        ],
        "timestamp": now_utc.isoformat(),
        "footer":    {"text": "JK Trump Alpha · Paper trade closed"},
    }

    try:
        requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
    except Exception:
        pass   # non-critical — don't interrupt the main flow


def show_open_positions(verbose: bool = True):
    """Print current open paper positions with live P&L."""
    if not _alpaca_available():
        print("  [alpaca] API keys not configured")
        return

    state     = load_state()
    positions = state.get("paper_positions", [])
    if not positions:
        print("  [alpaca] No open paper positions")
        return

    now_utc = datetime.now(timezone.utc)
    print(f"\n{'TICKER':<8} {'DIR':<6} {'URGENCY':<10} {'NOTIONAL':>9} {'UNREAL P&L':>11} {'AGE':>10}  THESIS")
    print("-" * 95)
    total_unrealized = 0.0
    unrealized_notional = 0.0
    for pos in positions:
        try:
            entry_time = datetime.fromisoformat(pos["entry_time"])
            age_h      = (now_utc - entry_time).total_seconds() / 3600
            age_str    = f"{age_h:.1f}h"
        except Exception:
            age_str    = "?"

        pnl      = get_position_pnl(pos["ticker"])
        notional = pos.get("notional", 0)
        pstr     = f"{pnl*100:+.1f}%" if pnl is not None else "?"
        if pnl is not None:
            total_unrealized   += pnl * notional
            unrealized_notional += notional
        print(f"  {pos['ticker']:<6} {pos['direction']:<6} {pos.get('urgency','?'):<10} "
              f"${notional:>7.0f} {pstr:>11} {age_str:>10}  {pos.get('thesis','')[:50]}")

    # Cumulative P&L summary
    print("-" * 95)
    realized_pnl = state.get("cumulative_realized_pnl", 0.0)
    n_closed     = len(state.get("closed_positions", []))
    print(f"\n  Unrealized (open):  ${total_unrealized:+.2f}  (on ${unrealized_notional:.0f} notional)")
    print(f"  Realized (closed):  ${realized_pnl:+.2f}  ({n_closed} closed positions)")
    print(f"  Total P&L:          ${total_unrealized + realized_pnl:+.2f}")


def close_all_positions(verbose: bool = True):
    """Emergency: close every open paper position immediately."""
    if not _alpaca_available():
        print("  [alpaca] API keys not configured")
        return
    try:
        resp = requests.delete(
            f"{ALPACA_PAPER_BASE}/positions",
            headers=_alpaca_headers(),
            timeout=15,
        )
        if verbose:
            print(f"  [alpaca] Close-all response: {resp.status_code}")
    except Exception as e:
        print(f"  [alpaca] close-all failed: {e}")

    # Clear state
    state = load_state()
    closed = state.get("paper_positions", [])
    for pos in closed:
        pos["close_reason"] = "manual_close_all"
    state["closed_positions"] = state.get("closed_positions", []) + closed
    state["paper_positions"]  = []
    save_state(state)
    if verbose:
        print(f"  [alpaca] Cleared {len(closed)} position(s) from state")


# ---------------------------------------------------------------------------
# Historical backtest
# ---------------------------------------------------------------------------

TRUMP_ACCOUNT_ID = "107780257626128497"   # @realDonaldTrump on Truth Social


def fetch_posts_public(days: int = 60, verbose: bool = True) -> list:
    """
    Fetch Trump's Truth Social posts using the public Mastodon-compatible API
    (no OAuth required — uses curl_cffi browser impersonation to bypass HTML redirect).

    Paginates backwards using max_id until posts are older than `days` ago.
    Returns list of dicts: {"post": raw_dict, "text": clean_text, "dt": datetime}
    """
    import re
    import curl_cffi.requests as cffi_req
    from dateutil import parser as date_parse

    cutoff  = datetime.now(timezone.utc) - timedelta(days=days)
    base    = f"https://truthsocial.com/api/v1/accounts/{TRUMP_ACCOUNT_ID}/statuses"
    headers = {
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control":   "no-cache",
        "Referer":         "https://truthsocial.com/@realDonaldTrump",
    }

    all_posts = []
    max_id    = None
    page      = 0

    while True:
        params = {
            "limit":            40,
            "exclude_reblogs":  "true",
        }
        if max_id:
            params["max_id"] = max_id

        # Retry loop — handles 429 rate limits with exponential backoff
        batch      = None
        last_err   = None
        max_retry  = 6
        for attempt in range(max_retry):
            try:
                resp = cffi_req.get(
                    base,
                    impersonate="chrome110",
                    headers=headers,
                    params=params,
                    timeout=20,
                )
                if resp.status_code == 429:
                    wait = 8 * (attempt + 1)  # 8s, 16s, 24s, 32s
                    print(f"  [public_api] Rate limited (429) — sleeping {wait}s "
                          f"(attempt {attempt+1}/{max_retry})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                batch = resp.json()
                break
            except Exception as e:
                last_err = e
                if attempt < max_retry - 1:
                    time.sleep(3)

        if batch is None:
            print(f"  [public_api] Fetch error on page {page+1}: {last_err}")
            break

        if not batch:
            break

        page += 1
        stop  = False

        for post in batch:
            try:
                _parsed_dt = date_parse.parse(post.get("created_at", ""))
                post_dt = _parsed_dt.replace(tzinfo=timezone.utc) if _parsed_dt.tzinfo is None else _parsed_dt.astimezone(timezone.utc)
            except Exception:
                continue

            if post_dt < cutoff:
                stop = True
                break

            text = re.sub(r"<[^>]+>", "", post.get("content", "")).strip()
            all_posts.append({"post": post, "text": text, "dt": post_dt})

        if verbose:
            print(f"  [public_api] Page {page}: fetched {len(batch)} posts, total={len(all_posts)}")

        if stop:
            break

        # Paginate — use the ID of the last post in the batch as max_id
        max_id = batch[-1]["id"]
        time.sleep(2.0)   # rate-limit buffer (0.5s caused 429s on deeper pagination)

    return all_posts


def _keyword_hit(text: str) -> bool:
    """Return True if the post contains at least one market-relevant keyword."""
    lower = text.lower()
    return any(kw in lower for kw in _BACKTEST_KEYWORDS)


def _fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch OHLCV bars for backtest. Tries Alpaca first, falls back to yfinance."""
    if _alpaca_available():
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from technical_analysis.backtest.signal_tester import fetch_data_alpaca
            df = fetch_data_alpaca(ticker, start=start_date, end=end_date)
            if df is not None and len(df) >= 3:
                return df
        except Exception:
            pass

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        if len(df) >= 3:
            return df
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# AutoResearch helpers
# ---------------------------------------------------------------------------

def load_trump_params() -> dict:
    """Load best Trump bot params, merging saved values over defaults.
    This ensures new params (e.g. take_profit_pct) always have a value
    even if the saved file predates their addition."""
    merged = TRUMP_DEFAULT_PARAMS.copy()
    if TRUMP_PARAMS_FILE.exists():
        try:
            with open(TRUMP_PARAMS_FILE) as f:
                saved = json.load(f)
            merged.update(saved)
        except Exception:
            pass
    return merged


def save_trump_params(params: dict):
    TRUMP_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUMP_PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def cache_analyzed_posts(days: int = 60, force: bool = False, verbose: bool = True) -> list:
    """
    Fetch and LLM-analyze Trump posts, caching results for 12 hours.
    Returns list of tradeable-signal dicts (ticker, direction, urgency, etc.)

    force=True always re-fetches (used by --backtest which needs fresh data).
    """
    import time as _time

    # Check cache freshness
    if not force and TRUMP_POST_CACHE.exists():
        try:
            age_hours = (_time.time() - TRUMP_POST_CACHE.stat().st_mtime) / 3600
            if age_hours < 12:
                with open(TRUMP_POST_CACHE) as f:
                    data = json.load(f)
                if data.get("days") == days:
                    posts = data.get("posts", [])
                    # Re-parse post_dt strings back to datetime
                    for p in posts:
                        if isinstance(p.get("post_dt"), str):
                            from dateutil import parser as dp
                            _pdt = dp.parse(p["post_dt"])
                            p["post_dt"] = _pdt.replace(tzinfo=timezone.utc) if _pdt.tzinfo is None else _pdt.astimezone(timezone.utc)
                    if verbose:
                        print(f"  [cache] Loaded {len(posts)} analyzed posts from cache "
                              f"({age_hours:.1f}h old)")
                    return posts
        except Exception:
            pass

    # Fetch + analyze
    if verbose:
        print("  Step 1/4: Pulling Trump posts via public Truth Social API...")
    all_posts = fetch_posts_public(days=days, verbose=verbose)
    if verbose:
        print(f"  Found {len(all_posts)} posts in last {days} days")
        print("  Step 2/4: Running LLM analysis (keyword-filtered)...")

    skipped_kw = 0
    skipped_rt = 0
    flagged    = 0
    trades     = []

    for i, item in enumerate(all_posts):
        text    = item["text"]
        post_dt = item["dt"]
        post    = item["post"]

        # Filter out retweets/reposts — not new information from Trump himself
        if text.startswith("RT ") or text.startswith('"RT '):
            skipped_rt += 1
            continue

        if not _keyword_hit(text):
            skipped_kw += 1
            continue

        if verbose:
            print(f"    [{i+1}/{len(all_posts)}] {text[:70]}...")

        analysis = analyze_post(text, post.get("id", ""))
        if analysis is None:
            continue

        flagged += 1
        ticker     = analysis.get("primary_ticker", "").upper()
        direction  = analysis.get("direction", "")
        urgency    = analysis.get("urgency", "TREND")
        confidence = analysis.get("confidence", 7)

        if not ticker or direction == "MIXED":
            continue

        trades.append({
            "post_id":    post.get("id", ""),
            "post_dt":    post_dt,
            "post_text":  text[:200],
            "ticker":     ticker,
            "direction":  direction,
            "urgency":    urgency,
            "confidence": confidence,
            "thesis":     analysis.get("thesis", ""),
            "category":   analysis.get("category", "OTHER"),
        })

    if verbose:
        print(f"  RT-filtered: {skipped_rt} reposts skipped")
        print(f"  Keyword-filtered: {skipped_kw} skipped, "
              f"{len(all_posts)-skipped_kw-skipped_rt} analyzed")
        print(f"  Market-moving: {flagged} posts flagged → {len(trades)} tradeable signals")

    # Save cache (post_dt as ISO string for JSON serialization)
    cache_data = {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "days":      days,
        "posts":     [
            {**t, "post_dt": t["post_dt"].isoformat()} for t in trades
        ],
    }
    TRUMP_POST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUMP_POST_CACHE, "w") as f:
        json.dump(cache_data, f, indent=2)

    return trades


def _prefetch_ohlcv_for_trades(trades: list, start_str: str, end_str: str,
                                verbose: bool = True) -> dict:
    """
    Fetch OHLCV for all unique tickers in trades.
    When USE_LEVERAGE=True, resolves each signal ticker to its leveraged ETF variants
    (both long and short ETF, since different trades may go in different directions).
    Returns price_cache keyed by the ETF/ticker that was fetched.
    """
    # Collect all fetch tickers (ETF symbols, not signal symbols)
    fetch_tickers: set = set()
    for trade in trades:
        sig   = trade["ticker"]
        direc = trade["direction"]
        etf, _, _  = _resolve_leveraged_ticker(sig, "LONG")    # long variant
        etf2, _, _ = _resolve_leveraged_ticker(sig, "SHORT")   # short variant
        fetch_tickers.add(etf)
        fetch_tickers.add(etf2)

    price_cache: dict = {}
    for tk in sorted(fetch_tickers):
        df = _fetch_ohlcv(tk, start_str, end_str)
        price_cache[tk] = df
        if verbose:
            status = f"{len(df)} bars" if df is not None else "FAILED"
            print(f"    [{tk}] {status}")

    return price_cache


def simulate_with_params(trades: list, price_cache: dict, params: dict) -> dict:
    """
    Pure function: simulate Trump trades given pre-analyzed posts + pre-fetched prices.
    Uses params for stop_loss_pct, cooldown_hours (deduplication), and confidence_threshold.
    No API calls. Returns {"trades": [...], "metrics": {...}}.

    Stop-loss: checks if intraday Low hit stop before exit day.
    Deduplication: within cooldown_hours, suppress duplicate (ticker, direction) signals.
    """
    stop_loss            = params.get("stop_loss_pct", 0.018)
    take_profit          = params.get("take_profit_pct", 0.04)
    slippage             = SLIPPAGE_PCT * 2   # round-trip (entry + exit)
    cooldown_hours       = int(params.get("cooldown_hours", 4))
    confidence_threshold = int(params.get("confidence_threshold", 6))

    # Always use 1d KNEE_JERK horizon, 5d TREND horizon (structural choice, not search param)
    trend_horizon = 5

    raw_signal_count = len(trades)

    # ── Step 1: Filter by confidence threshold ────────────────────────────
    conf_filtered = [
        t for t in trades
        if t.get("confidence", 7) >= confidence_threshold
    ]

    # ── Step 2: Deduplicate by (ticker, direction) within cooldown window ─
    # Sort by post_dt ascending so we process chronologically
    sorted_trades = sorted(conf_filtered, key=lambda t: t["post_dt"])
    last_signal: dict = {}   # key: (ticker, direction) → last post_dt
    deduped_trades = []
    for t in sorted_trades:
        key = (t["ticker"], t["direction"])
        post_dt = t["post_dt"]
        last_dt = last_signal.get(key)
        if last_dt is not None:
            elapsed = (post_dt - last_dt).total_seconds() / 3600
            if elapsed < cooldown_hours:
                continue   # duplicate within cooldown window — skip
        last_signal[key] = post_dt
        deduped_trades.append(t)

    results = []

    for trade in deduped_trades:
        signal_ticker = trade["ticker"]
        direction     = trade["direction"]
        urgency       = trade["urgency"]
        post_dt       = trade["post_dt"]
        post_date     = post_dt.date() if hasattr(post_dt, "date") else post_dt

        # Resolve leveraged ticker for this trade
        trade_ticker, multiplier, always_long = _resolve_leveraged_ticker(
            signal_ticker, direction
        )

        df = price_cache.get(trade_ticker)
        if df is None or len(df) < 2:
            continue

        try:
            # Normalize index to date objects
            if hasattr(df.index, "date"):
                idx_dates = [d.date() if hasattr(d, "date") else d for d in df.index]
            else:
                idx_dates = list(df.index)

            # Find next trading day after post
            future_idx = [i for i, d in enumerate(idx_dates) if d > post_date]
            if len(future_idx) < 1:
                continue

            entry_i     = future_idx[0]
            entry_price = float(df.iloc[entry_i]["Open"])

            # Determine exit horizon
            horizon = 1 if urgency == "KNEE_JERK" else trend_horizon

            ret = {}
            for h in [1, 3, 5]:
                if entry_i + h >= len(df):
                    ret[h] = None
                    continue

                # Check stop-loss AND take-profit intraday (Low/High per day)
                # For LONG / inverse ETF (always_long): stop at Low, profit at High
                # For direct SHORT (no leverage map): stop at High (price up = loss), profit at Low (price down = gain)
                is_long_side = always_long or direction == "LONG"
                if is_long_side:
                    stop_price   = entry_price * (1 - stop_loss)
                    profit_price = entry_price * (1 + take_profit)
                else:
                    # Direct SHORT: losing when price goes UP, profiting when price goes DOWN
                    stop_price   = entry_price * (1 + stop_loss)
                    profit_price = entry_price * (1 - take_profit)

                stopped_out  = False
                took_profit  = False
                for day_i in range(entry_i + 1, entry_i + h + 1):
                    if day_i >= len(df):
                        break
                    day_low  = float(df.iloc[day_i]["Low"])
                    day_high = float(df.iloc[day_i]["High"])
                    day_open = float(df.iloc[day_i]["Open"])

                    if is_long_side:
                        stop_hit   = day_low <= stop_price
                        profit_hit = day_high >= profit_price
                    else:
                        stop_hit   = day_high >= stop_price
                        profit_hit = day_low <= profit_price

                    if stop_hit and profit_hit:
                        # Both triggers on same bar — use proximity to Open as tiebreaker
                        dist_to_stop   = abs(day_open - stop_price)
                        dist_to_profit = abs(day_open - profit_price)
                        if dist_to_stop <= dist_to_profit:
                            stopped_out = True
                        else:
                            took_profit = True
                        break
                    elif stop_hit:
                        stopped_out = True
                        break
                    elif profit_hit:
                        took_profit = True
                        break

                if stopped_out:
                    ret[h] = -stop_loss - slippage   # stopped out at -stop_loss_pct
                elif took_profit:
                    ret[h] = take_profit - slippage  # take-profit hit
                else:
                    exit_price  = float(df.iloc[entry_i + h]["Close"])
                    raw_ret     = (exit_price - entry_price) / entry_price
                    # When always_long (inverse ETF), raw_ret already encodes direction
                    directional = raw_ret if (always_long or direction == "LONG") else -raw_ret
                    ret[h] = directional - slippage

            results.append({
                **trade,
                "trade_ticker":        trade_ticker,
                "leverage_multiplier": multiplier,
                "entry_date":          str(idx_dates[entry_i]),
                "entry_price":         entry_price,
                "returns":             ret,
            })

        except Exception:
            continue

    # Compute metrics
    def _s(xs):
        clean = [x for x in xs if x is not None]
        if not clean:
            return {"n": 0, "win_rate": 0.0, "avg_ret": 0.0}
        wins = sum(1 for x in clean if x > 0)
        return {"n": len(clean), "win_rate": wins / len(clean),
                "avg_ret": sum(clean) / len(clean)}

    m1 = _s([r["returns"].get(1) for r in results])

    # Primary objective: avg return at the urgency-appropriate horizon
    primary_returns = []
    for r in results:
        h = 1 if r.get("urgency") == "KNEE_JERK" else trend_horizon
        v = r["returns"].get(h)
        if v is not None:
            primary_returns.append(v)

    avg_ret  = sum(primary_returns) / len(primary_returns) if primary_returns else 0.0
    win_rate = sum(1 for v in primary_returns if v > 0) / len(primary_returns) \
               if primary_returns else 0.0

    metrics = {
        "n_trades":          len(results),
        "raw_signals":       raw_signal_count,
        "independent_bets":  len(deduped_trades),
        "win_rate_1d":       m1["win_rate"],
        "avg_1d_ret":        m1["avg_ret"],
        "avg_ret":           avg_ret,       # primary objective
        "win_rate":          win_rate,
        "trend_horizon_used": trend_horizon,
    }

    return {"trades": results, "metrics": metrics}


def _log_trump_experiment(entry: dict):
    TRUMP_LEARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRUMP_LEARN_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _load_trump_history() -> list:
    if not TRUMP_LEARN_LOG.exists():
        return []
    out = []
    with open(TRUMP_LEARN_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def propose_trump_tweaks(current_params: dict, history: list, n: int = 3) -> list:
    """
    Use local Ollama to propose N distinct parameter tweaks for Trump bot.
    Returns list of validated {"changes": {...}, "hypothesis": "..."} dicts.
    """
    history_lines = []
    for h in history[-20:]:  # last 20 experiments for context
        improved = "✓" if h.get("improved") else "✗"
        history_lines.append(
            f"  {improved} params={h.get('params_tested')}  "
            f"obj={h.get('objective', 0):.4f}  [{h.get('hypothesis', '')}]"
        )

    user_msg = (
        f"Current best params: {json.dumps(current_params)}\n"
        f"Current best objective (avg_ret): {current_params.get('_objective', 'unknown')}\n\n"
        f"Recent experiment history:\n" + ("\n".join(history_lines) or "  (none yet)") + "\n\n"
        f"Propose {n} distinct parameter tweaks to maximize avg_ret."
    )

    try:
        proposals = llm_chat_json_array(
            system=TRUMP_AUTOLEARN_SYSTEM_PROMPT.replace("{n}", str(n)),
            user=user_msg,
            max_tokens=600,
            temperature=0.7,
        )
    except Exception as e:
        print(f"  [trump_learn] LLM proposal failed: {e}")
        return []

    validated = []
    for prop in proposals:
        changes = prop.get("changes", {})
        clean   = {}
        for param, value in changes.items():
            if param not in TRUMP_PARAM_BOUNDS:
                continue
            lo, hi = TRUMP_PARAM_BOUNDS[param]
            if param in ("cooldown_hours", "confidence_threshold"):
                value = int(round(float(value)))
            else:
                value = float(value)
            value = max(lo, min(hi, value))
            if abs(value - current_params.get(param, 0)) > 1e-6:
                clean[param] = value
        if clean:
            validated.append({"changes": clean, "hypothesis": prop.get("hypothesis", "")})

    return validated[:n]


def run_trump_autolearn(n_rounds: int = 10, days: int = 60, verbose: bool = True):
    """
    Karpathy-style AutoResearch loop for Trump bot parameters.

    Two-phase design:
      Phase 1 (once): Fetch posts + LLM analysis + price data
      Phase 2 (N rounds): Vary params, simulate cheaply in parallel, keep best
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    webhook_url = os.environ.get("JK_DISCORD_TRUMP_WEBHOOK")
    cutoff      = datetime.now(timezone.utc) - timedelta(days=days)
    start_str   = cutoff.strftime("%Y-%m-%d")
    end_str     = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  Trump Alpha AutoResearch — {n_rounds} rounds, {days}-day window")
    print(f"{'='*60}\n")

    # ── Phase 1: Fetch data (cache for reuse across rounds) ───────────────
    trades = cache_analyzed_posts(days=days, force=False, verbose=verbose)
    if not trades:
        print("  No tradeable signals — cannot run AutoResearch")
        return

    print("  Fetching price data (leveraged ETFs)...")
    price_cache = _prefetch_ohlcv_for_trades(trades, start_str, end_str, verbose=verbose)

    # ── Phase 2: Optimization loop ────────────────────────────────────────
    current_params = load_trump_params()
    baseline       = simulate_with_params(trades, price_cache, current_params)
    best_obj       = baseline["metrics"]["avg_ret"]
    current_params["_objective"] = best_obj

    history = _load_trump_history()

    print(f"\n  Baseline: avg_ret={best_obj*100:+.3f}%  "
          f"win_rate={baseline['metrics']['win_rate']*100:.0f}%  "
          f"n={baseline['metrics']['n_trades']}\n")

    for rnd in range(1, n_rounds + 1):
        print(f"  Round {rnd}/{n_rounds}")
        proposals = propose_trump_tweaks(current_params, history, n=3)
        if not proposals:
            print("    No valid proposals — skipping round")
            continue

        # Run 3 simulations in parallel (pure computation, no API calls)
        def _run_one(prop):
            test_p = {**current_params, **prop["changes"]}
            test_p.pop("_objective", None)
            result = simulate_with_params(trades, price_cache, test_p)
            return prop, test_p, result

        futures_map = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures_map = {pool.submit(_run_one, p): p for p in proposals}
            outcomes = []
            for fut in as_completed(futures_map):
                try:
                    outcomes.append(fut.result())
                except Exception as e:
                    print(f"    Simulation error: {e}")

        # Find best of this round
        best_this_round = max(outcomes, key=lambda x: x[2]["metrics"]["avg_ret"],
                              default=None)

        for prop, test_p, result in outcomes:
            obj       = result["metrics"]["avg_ret"]
            improved  = (obj > best_obj and prop["changes"] == best_this_round[0]["changes"]) if best_this_round else False

            entry = {
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "round":         rnd,
                "params_tested": {k: v for k, v in test_p.items() if k != "_objective"},
                "objective":     obj,
                "baseline":      best_obj,
                "improved":      improved,
                "hypothesis":    prop.get("hypothesis", ""),
                "changes":       prop["changes"],
                "metrics":       result["metrics"],
            }
            _log_trump_experiment(entry)
            history.append(entry)

            mark = "✓ IMPROVED" if improved else "✗"
            print(f"    {mark}  obj={obj*100:+.3f}%  "
                  f"[{prop['hypothesis'][:60]}]")

        # Keep improvement if any
        if best_this_round:
            prop, test_p, result = best_this_round
            obj = result["metrics"]["avg_ret"]
            if obj > best_obj:
                best_obj       = obj
                current_params = {**test_p, "_objective": best_obj}
                print(f"    → New best: {best_obj*100:+.3f}%")

    # Save best params and post summary
    save_params = {k: v for k, v in current_params.items() if k != "_objective"}
    save_trump_params(save_params)
    print(f"\n  Best params saved: {json.dumps(save_params)}")
    print(f"  Final avg_ret: {best_obj*100:+.3f}%")

    # Post summary to Discord
    if webhook_url:
        _post_autolearn_summary(save_params, best_obj, baseline["metrics"]["avg_ret"],
                                n_rounds, days, webhook_url)


def _post_autolearn_summary(best_params: dict, best_obj: float, baseline_obj: float,
                             n_rounds: int, days: int, webhook_url: str):
    """Post AutoResearch completion summary to Discord."""
    delta     = best_obj - baseline_obj
    delta_str = f"{delta*100:+.3f}%"
    color     = 0x2ecc71 if delta > 0 else (0xe74c3c if delta < -0.001 else 0xf39c12)

    param_lines = "\n".join(
        f"  `{k}`: **{v}**" for k, v in best_params.items()
    )

    embed = {
        "title":  f"🧠 Trump Alpha AutoResearch — {n_rounds} rounds complete",
        "color":  color,
        "fields": [
            {
                "name":  "📈 Objective Change",
                "value": f"Baseline avg_ret: `{baseline_obj*100:+.3f}%`\n"
                         f"Best avg_ret: `{best_obj*100:+.3f}%`\n"
                         f"Delta: **{delta_str}**",
                "inline": False,
            },
            {
                "name":  "⚙️ Best Params",
                "value": param_lines,
                "inline": False,
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer":    {"text": f"Trump AutoResearch · {days}-day window · "
                              f"{n_rounds} rounds × 3 parallel experiments"},
    }

    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=15)
        resp.raise_for_status()
        print("  ✅ AutoResearch summary posted to Discord")
    except Exception as e:
        print(f"  ❌ Discord post failed: {e}")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

def run_backtest(days: int = 60, verbose: bool = True):
    """
    Pull last `days` days of Trump Truth Social posts, run LLM analysis on
    candidate posts, simulate paper trades using historical OHLCV, and post
    a results summary to #trump-alpha Discord channel.

    Entry: next-day open after post (conservative — catches both AH and pre-mkt posts)
    Exit:  1d / 3d / 5d close (stop-loss-aware using daily Low)
    Win:   directional return > 0 (LONG → price up, SHORT → price down)
    """
    webhook_url = os.environ.get("JK_DISCORD_TRUMP_WEBHOOK")
    cutoff      = datetime.now(timezone.utc) - timedelta(days=days)
    start_str   = cutoff.strftime("%Y-%m-%d")
    end_str     = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")

    lev_note = " · Leveraged ETFs (2x–3x)" if USE_LEVERAGE else ""
    print(f"\n{'='*60}")
    print(f"  Trump Alpha — Historical Backtest ({days} days){lev_note}")
    print(f"  Period: {start_str} → {end_str}")
    print(f"{'='*60}\n")

    # ── Steps 1+2: fetch posts + LLM analysis (force=True for fresh data) ─
    trades = cache_analyzed_posts(days=days, force=True, verbose=verbose)
    if not trades:
        print("  No tradeable signals found — nothing to post")
        return

    # ── Step 3: fetch leveraged ETF prices ────────────────────────────────
    print("  Step 3/4: Fetching historical prices...")
    price_cache = _prefetch_ohlcv_for_trades(trades, start_str, end_str, verbose=verbose)

    # ── Simulate with current best params ─────────────────────────────────
    params = load_trump_params()
    sim    = simulate_with_params(trades, price_cache, params)
    results = sim["trades"]
    metrics = sim["metrics"]

    if verbose:
        for r in results:
            tk = r.get("trade_ticker", r["ticker"])
            sig = r["ticker"]
            lev_tag = f" ({r['leverage_multiplier']}x)" if r.get("leverage_multiplier", 1) > 1 else ""
            r1  = f"{r['returns'][1]*100:+.1f}%" if r["returns"].get(1) is not None else "?"
            print(f"    {r['direction']:<5} {tk}{lev_tag}  [{sig}]  "
                  f"entry={r['entry_price']:.2f}  1d={r1}  "
                  f"thesis: {r['thesis'][:45]}")

    print(f"\n  Simulated {len(results)} trades "
          f"({metrics.get('raw_signals', len(results))} raw signals → "
          f"{metrics.get('independent_bets', len(results))} independent bets)")

    # ── Step 4: post to Discord ───────────────────────────────────────────
    print("  Step 4/4: Computing stats and posting to Discord...")
    _post_backtest_results(results, days, webhook_url, verbose, metrics=metrics)


def _post_backtest_results(results: list, days: int, webhook_url: Optional[str],
                           verbose: bool = True, metrics: Optional[dict] = None):
    """Format backtest results and post summary embed to Discord."""
    if not results:
        print("  No results to post")
        return

    def _stats(ret_list):
        clean = [r for r in ret_list if r is not None]
        if not clean:
            return {"n": 0, "win_rate": 0, "avg_ret": 0, "total_pnl": 0}
        wins = sum(1 for r in clean if r > 0)
        return {
            "n":        len(clean),
            "win_rate": wins / len(clean),
            "avg_ret":  sum(clean) / len(clean),
            "total_pnl": sum(clean),
        }

    # Overall + per-horizon stats
    by_h = {h: _stats([r["returns"].get(h) for r in results]) for h in [1, 3, 5]}

    # By urgency tier
    kj   = [r for r in results if r.get("urgency") == "KNEE_JERK"]
    tr   = [r for r in results if r.get("urgency") == "TREND"]
    kj_s = {h: _stats([r["returns"].get(h) for r in kj]) for h in [1, 3, 5]}
    tr_s = {h: _stats([r["returns"].get(h) for r in tr]) for h in [1, 3, 5]}

    # Format numbers
    def _pct(v): return f"{v*100:+.1f}%"
    def _wr(v):  return f"{v*100:.0f}%"

    # Summary table text
    lines = [
        f"```",
        f"Horizon │ n   │ Win Rate │ Avg Ret │ Total P&L",
        f"────────┼─────┼──────────┼─────────┼──────────",
    ]
    for h in [1, 3, 5]:
        s = by_h[h]
        lines.append(f"  {h}d    │ {s['n']:<4}│  {_wr(s['win_rate']):<8}│ {_pct(s['avg_ret']):<8}│ {_pct(s['total_pnl'])}")
    lines.append("```")
    table_str = "\n".join(lines)

    # Top 5 best trades (by 1d directional return)
    top_trades = sorted(
        [r for r in results if r["returns"].get(1) is not None],
        key=lambda x: x["returns"][1], reverse=True
    )[:5]

    def _trade_label(r):
        sig = r["ticker"]
        etf = r.get("trade_ticker", sig)
        mult = r.get("leverage_multiplier", 1)
        lev  = f"({mult}x)" if mult > 1 else ""
        return f"{etf}{lev}" if etf != sig else sig

    top_lines = []
    for r in top_trades:
        emoji = "📈" if r["direction"] == "LONG" else "📉"
        ret1d = _pct(r["returns"][1])
        top_lines.append(f"{emoji} **{_trade_label(r)}** {r['direction']} `{ret1d}` — {r['thesis'][:65]}")

    # Worst 3
    bot_trades = sorted(
        [r for r in results if r["returns"].get(1) is not None],
        key=lambda x: x["returns"][1]
    )[:3]
    bot_lines = []
    for r in bot_trades:
        ret1d = _pct(r["returns"][1])
        bot_lines.append(f"🔴 **{_trade_label(r)}** {r['direction']} `{ret1d}` — {r['thesis'][:65]}")

    # KNEE_JERK vs TREND section
    urgency_parts = []
    for label, s_map in [("⚡ KNEE-JERK", kj_s), ("📈 TREND", tr_s)]:
        n = s_map[1]["n"]
        if n == 0:
            continue
        wr = _wr(s_map[1]["win_rate"])
        ar = _pct(s_map[1]["avg_ret"])
        urgency_parts.append(f"{label} ({n} trades) — Win rate: {wr} · Avg 1d: {ar}")

    raw_signals    = metrics.get("raw_signals", len(results)) if metrics else len(results)
    indep_bets     = metrics.get("independent_bets", len(results)) if metrics else len(results)
    dedup_note     = f" · {raw_signals} raw → {indep_bets} independent bets" if raw_signals != indep_bets else ""

    fields = [
        {
            "name": f"📊 Results Summary — last {days} days  ({len(results)} simulated{dedup_note})",
            "value": table_str,
            "inline": False,
        },
    ]
    if urgency_parts:
        fields.append({
            "name": "By Urgency Tier",
            "value": "\n".join(urgency_parts),
            "inline": False,
        })
    if top_lines:
        fields.append({
            "name": "🏆 Best Trades (1d)",
            "value": "\n".join(top_lines),
            "inline": False,
        })
    if bot_lines:
        fields.append({
            "name": "💀 Worst Trades (1d)",
            "value": "\n".join(bot_lines),
            "inline": False,
        })

    # Overall 1d win rate for embed color
    wr_1d = by_h[1]["win_rate"]
    color = 0x2ecc71 if wr_1d >= 0.55 else (0xe74c3c if wr_1d < 0.45 else 0xf39c12)

    embed = {
        "title":     f"🇺🇸 Trump Alpha Backtest — {days}-Day Historical Simulation",
        "color":     color,
        "fields":    fields,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer":    {"text": f"Entry: next-day open · Exit: 1d/3d/5d close · SL/TP intraday · {SLIPPAGE_PCT*200:.1f}% RT slippage"
                              + (" · Leveraged ETFs (2x–3x)" if USE_LEVERAGE else "")},
    }

    payload = {"embeds": [embed]}

    if verbose:
        print(f"\n  Overall 1d — n={by_h[1]['n']}, win={_wr(wr_1d)}, avg={_pct(by_h[1]['avg_ret'])}")

    if webhook_url:
        try:
            resp = requests.post(webhook_url, json=payload, timeout=15)
            resp.raise_for_status()
            print(f"  ✅ Backtest results posted to Discord")
        except Exception as e:
            print(f"  ❌ Discord post failed: {e}")
    else:
        print("  (no webhook URL — would post to Discord)")
        import pprint
        pprint.pprint(payload)


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------

def run_once(verbose: bool = True) -> int:
    """
    Single poll cycle: fetch new posts, analyze, alert, optionally paper-trade.
    Returns number of alerts sent.
    """
    webhook_url = os.environ.get("JK_DISCORD_TRUMP_WEBHOOK")
    if not webhook_url:
        print("  [trump] WARNING: JK_DISCORD_TRUMP_WEBHOOK not set — alerts will print but not post")

    # Check + close stale paper positions on every cycle (cheap)
    if _alpaca_available():
        check_and_close_positions(verbose=False)

    state     = load_state()
    last_id   = state.get("last_seen_id")
    new_max_id = last_id

    posts = fetch_new_posts(since_id=last_id, verbose=verbose)

    if not posts:
        if verbose:
            print("  [trump] No new posts")
        return 0

    alerts_sent = 0
    import re as _re_once
    now_utc = datetime.now(timezone.utc)

    for post in posts:
        post_id = post.get("id", "")

        # Track the newest post ID we've seen
        if new_max_id is None or int(post_id) > int(new_max_id):
            new_max_id = post_id

        # Strip HTML for display / analysis
        raw_content = post.get("content", "")
        text = _re_once.sub(r"<[^>]+>", "", raw_content).strip()

        if verbose:
            print(f"  [trump] Post {post_id}: {text[:80]}...")

        # Filter out retweets — not new information from Trump himself
        if text.startswith("RT ") or text.startswith('"RT '):
            if verbose:
                print(f"    → SKIP (retweet)")
            continue

        # LLM analysis
        analysis = analyze_post(text, post_id)

        if analysis is None:
            if verbose:
                print(f"    → SKIP (not market-moving)")
            continue

        # Market-moving — check quality gates
        ticker     = analysis.get("primary_ticker", "?")
        urgency    = analysis.get("urgency", "?")
        direction  = analysis.get("direction", "?")
        confidence = analysis.get("confidence", 7)

        # Confidence threshold gate
        if confidence < CONFIDENCE_THRESHOLD:
            if verbose:
                print(f"    → SKIP (confidence {confidence} < threshold {CONFIDENCE_THRESHOLD})")
            continue

        # Cooldown gate — suppress same (ticker, direction) within SIGNAL_COOLDOWN_HOURS
        recent_signals = state.get("recent_signals", {})
        cooldown_key   = f"{ticker}_{direction}"
        last_signal_ts = recent_signals.get(cooldown_key)
        if last_signal_ts:
            try:
                last_dt  = datetime.fromisoformat(last_signal_ts)
                elapsed  = (now_utc - last_dt).total_seconds() / 3600
                if elapsed < SIGNAL_COOLDOWN_HOURS:
                    if verbose:
                        print(f"    → SKIP (cooldown: {elapsed:.1f}h < {SIGNAL_COOLDOWN_HOURS}h for {ticker} {direction})")
                    continue
            except Exception:
                pass

        print(f"    ⚡ ALERT: {direction} {ticker} [{urgency}] conf={confidence} — {analysis.get('thesis','')[:80]}")

        # Discord alert
        if webhook_url:
            payload = format_trump_embed(post, analysis)
            try:
                post_to_discord(payload, webhook_url)
                alerts_sent += 1
            except Exception as e:
                print(f"    Discord post failed: {e}")
        else:
            alerts_sent += 1

        # Update cooldown state
        state = load_state()
        recent_signals = state.get("recent_signals", {})
        recent_signals[cooldown_key] = now_utc.isoformat()
        # Prune entries older than 24 hours to keep state small
        cutoff_prune = now_utc - timedelta(hours=24)
        pruned = {}
        for k, v in recent_signals.items():
            try:
                if datetime.fromisoformat(v) > cutoff_prune:
                    pruned[k] = v
            except Exception:
                pruned[k] = v
        state["recent_signals"] = pruned

        # Alpaca paper trade (only during market hours)
        if _alpaca_available():
            if is_market_hours():
                open_positions = state.get("paper_positions", [])
                ticker_positions = [
                    p for p in open_positions
                    if p.get("signal_ticker", "").upper() == ticker.upper()
                ]

                # Conflicting position check — skip if opposite direction already open on this ticker
                opposite_positions = [
                    p for p in ticker_positions
                    if p.get("direction", "") != direction
                ]
                if opposite_positions:
                    if verbose:
                        print(f"    [alpaca] Skipping — conflicting {opposite_positions[0].get('direction')} position already open on {ticker}")
                # Max positions per ticker gate
                elif len(ticker_positions) >= MAX_POSITIONS_PER_TICKER:
                    if verbose:
                        print(f"    [alpaca] Skipping — already {len(ticker_positions)} open position(s) for {ticker}")
                else:
                    position_info = place_paper_trade(analysis, post_id, verbose=verbose)
                    if position_info:
                        open_positions.append(position_info)
                        state["paper_positions"] = open_positions
            else:
                if verbose:
                    print(f"    [alpaca] Market closed — paper trade not placed (would be {direction} {ticker})")

        save_state(state)

    # Update last-seen ID + counters (reload state in case loop already modified it)
    state = load_state()
    if new_max_id and new_max_id != last_id:
        state["last_seen_id"]    = new_max_id
        state["posts_processed"] = state.get("posts_processed", 0) + len(posts)
        state["alerts_sent"]     = state.get("alerts_sent", 0) + alerts_sent
        save_state(state)

    return alerts_sent


def run_daemon(verbose: bool = True):
    """Continuous polling loop — runs forever, polling every POLL_INTERVAL_SECONDS."""
    print(f"  [trump] Starting daemon — polling every {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            run_once(verbose=verbose)
        except KeyboardInterrupt:
            print("\n  [trump] Stopped by user")
            break
        except Exception as e:
            print(f"  [trump] Unhandled error: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JK Trump Alpha — Truth Social trade alert bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Single poll then exit (default when called by launchd)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously, polling every 15 seconds",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear seen-state and reprocess the last 10 minutes of posts",
    )
    parser.add_argument(
        "--backtest",
        type=int,
        metavar="DAYS",
        default=0,
        help="Pull last N days of posts, simulate paper trades, post results to Discord  (e.g. --backtest 60)",
    )
    parser.add_argument(
        "--positions",
        action="store_true",
        help="Show current open Alpaca paper positions",
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        dest="close_all",
        help="Emergency close ALL open Alpaca paper positions",
    )
    parser.add_argument(
        "--learn",
        type=int,
        metavar="ROUNDS",
        default=0,
        help="Run AutoResearch loop for N rounds (e.g. --learn 10)",
    )
    parser.add_argument(
        "--learn-days",
        type=int,
        default=60,
        dest="learn_days",
        help="Days of posts to use for AutoResearch backtest window (default: 60)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
    )
    args = parser.parse_args()

    if args.reset:
        state = load_state()
        state["last_seen_id"] = None
        save_state(state)
        print("  [trump] State reset — will reprocess recent posts")

    if args.backtest > 0:
        run_backtest(days=args.backtest, verbose=not args.quiet)
    elif args.learn > 0:
        run_trump_autolearn(n_rounds=args.learn, days=args.learn_days,
                            verbose=not args.quiet)
    elif args.positions:
        show_open_positions()
    elif args.close_all:
        close_all_positions(verbose=not args.quiet)
    elif args.daemon:
        run_daemon(verbose=not args.quiet)
    else:
        # Default: single poll (used by launchd)
        run_once(verbose=not args.quiet)
