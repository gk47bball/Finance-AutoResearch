"""
Intraday Day-Trading Simulator
================================
Simulates the Four Pillars strategy applied to intraday price data for a
specific historical trading day.

How it works:
  1. Pulls daily data via EODHD (fallback: yfinance) to establish the morning
     REGIME (BULL / CHOP / BEAR) using the Four Pillars daily read.
  2. Fetches intraday bars (5m for recent dates, 1h for older) plus 15 trading
     days of WARMUP before the target date so RSI and EMA indicators are
     fully initialized when the target day begins.
  3. Applies simplified intraday signals on the target day's bars only:
       Entry:  close < EMA20 * (1 - 0.003)  AND  RSI < 43
               (only in BULL or CHOP regime)
       Exit:   close > EMA20 * (1 + 0.002)  OR  RSI > 60  OR  -1.5% stop
       Force close on last bar.
  4. Returns a full trade log with entry/exit times, prices, P&L.

Data:
  - Daily context: EODHD EOD API (basic subscription) → yfinance fallback
  - Intraday bars: yfinance only (EODHD intraday requires paid upgrade)
  - 5m interval: yfinance covers ~60 calendar days back
  - 1h interval: yfinance covers ~730 calendar days back

Usage:
  from technical_analysis.bot.intraday_sim import simulate_day
  results = simulate_day("XLK", "2026-03-18", interval="5m")
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from technical_analysis.bot.data_providers import get_eod, get_intraday_with_warmup


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Daily regime context via EODHD / yfinance
# ---------------------------------------------------------------------------

def get_daily_regime(ticker: str, before_date: str, n_days: int = 50) -> dict:
    """
    Fetch daily bars ending on before_date and compute the Four Pillars
    simplified regime (BULL / CHOP / BEAR) plus key levels.
    Uses EODHD for data quality, falls back to yfinance.
    """
    end = pd.Timestamp(before_date)
    start = (end - timedelta(days=int(n_days * 1.8))).strftime("%Y-%m-%d")

    try:
        df = get_eod(ticker, start=start, end=end.strftime("%Y-%m-%d"))
    except Exception as e:
        return {"regime": "chop", "trend_score": 0,
                "description": f"Data error: {e}", "error": True}

    df = df.tail(n_days)
    if len(df) < 20:
        return {"regime": "chop", "trend_score": 0,
                "description": "Insufficient daily history"}

    closes = df["close"]
    sma20 = closes.rolling(20).mean()
    sma50 = closes.rolling(50).mean() if len(closes) >= 50 else sma20

    last_close = float(closes.iloc[-1])
    last_sma20 = float(sma20.iloc[-1])
    last_sma50 = float(sma50.iloc[-1])

    # Simplified trend score
    score = 0
    if last_close > last_sma20:
        score += 1
    if last_close > last_sma50:
        score += 1
    if last_sma20 > last_sma50:
        score += 1

    # 5-day momentum
    if len(closes) >= 6:
        mom5 = (last_close - float(closes.iloc[-6])) / float(closes.iloc[-6])
        if mom5 > 0.005:
            score += 1
        elif mom5 < -0.010:
            score -= 2

    if score >= 3:
        regime = "bull"
    elif score <= 0:
        regime = "bear"
    else:
        regime = "chop"

    # Daily ATR for context
    atr = compute_atr(df).iloc[-1]
    atr_pct = atr / last_close

    return {
        "regime": regime,
        "trend_score": score,
        "last_close": round(last_close, 2),
        "sma20": round(last_sma20, 2),
        "sma50": round(last_sma50, 2),
        "atr_pct": round(atr_pct * 100, 3),
        "description": (
            f"trend_score={score} | close={last_close:.2f} vs "
            f"SMA20={last_sma20:.2f} SMA50={last_sma50:.2f} | "
            f"ATR={atr_pct:.2%}"
        ),
    }


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_day(ticker: str, date: str, interval: str = "5m") -> dict:
    """
    Simulate intraday day-trading for ticker on a specific historical date.

    Args:
        ticker:   Stock symbol (e.g. "SPY", "XLK")
        date:     Trading date as "YYYY-MM-DD" — must be a weekday (market open)
        interval: "5m" (up to ~55 days back) or "1h" (up to ~730 days back)

    Returns:
        dict with regime_context, trades list, summary stats, and error (if any)
    """
    result = {
        "ticker": ticker,
        "date": date,
        "interval": interval,
        "regime_context": {},
        "trades": [],
        "summary": {},
        "error": None,
    }

    # 1. Daily regime from EODHD (day BEFORE target so we only use prior info)
    try:
        target_ts = pd.Timestamp(date)
        prev_day = (target_ts - timedelta(days=3)).strftime("%Y-%m-%d")
        regime_info = get_daily_regime(ticker, before_date=date, n_days=50)
        result["regime_context"] = regime_info
    except Exception as e:
        result["error"] = f"Regime context failed: {e}"
        return result

    # 2. Intraday bars WITH multi-day warmup (solves the cold-start problem)
    try:
        all_bars = get_intraday_with_warmup(
            ticker, target_date=date, interval=interval, warmup_trading_days=15
        )
    except Exception as e:
        result["error"] = f"Intraday fetch failed: {e}"
        return result

    if all_bars.empty or len(all_bars) < 20:
        result["error"] = f"No intraday data for {ticker} on {date} ({interval})"
        return result

    # 3. Compute indicators across ALL bars (warmup + target day)
    closes = all_bars["close"]
    ema20 = compute_ema(closes, span=20)
    rsi = compute_rsi(closes, period=14)

    # Volume: ratio vs 30-bar rolling average (optional, loosened)
    vol_avg = all_bars["volume"].rolling(30, min_periods=5).mean()
    vol_ratio = all_bars["volume"] / vol_avg.replace(0, np.nan)

    # 4. Filter to target trading day only (regular hours 09:30 – 16:00 ET)
    target_date_obj = pd.Timestamp(date).date()
    day_mask = all_bars.index.date == target_date_obj
    day_df = all_bars[day_mask].between_time("09:30", "15:55")

    if len(day_df) < 3:
        result["error"] = (
            f"No regular-hours bars for {ticker} on {date}. "
            f"Market may have been closed or data unavailable."
        )
        return result

    day_ema20 = ema20[day_df.index]
    day_rsi = rsi[day_df.index]
    day_vol_ratio = vol_ratio[day_df.index]

    # 5. Trade simulation
    regime = regime_info.get("regime", "chop")
    tradeable = regime in ("bull", "chop")

    position = 0.0
    entry_price = None
    entry_time = None
    entry_reason = None
    trades = []
    total_pnl_pct = 0.0
    n_bars = len(day_df)

    for i in range(1, n_bars):
        bar_time = day_df.index[i]
        price = float(day_df["close"].iloc[i])
        curr_ema = float(day_ema20.iloc[i]) if not pd.isna(day_ema20.iloc[i]) else price
        curr_rsi = float(day_rsi.iloc[i]) if not pd.isna(day_rsi.iloc[i]) else 50.0
        curr_vol = float(day_vol_ratio.iloc[i]) if not pd.isna(day_vol_ratio.iloc[i]) else 1.0
        is_last = (i == n_bars - 1)

        if position == 0.0:
            # --- ENTRY ---
            # Thresholds calibrated for 5m/1h ETF intraday bars:
            # ETFs rarely deviate >0.3% from EMA20 in bull markets, so use 0.15%
            if tradeable and not is_last:
                below_ema = price < curr_ema * 0.9985       # 0.15% below 20-bar EMA
                rsi_oversold = curr_rsi < 45                 # RSI below 45
                if below_ema and rsi_oversold:
                    position = 1.0
                    entry_price = price
                    entry_time = bar_time
                    entry_reason = (
                        f"price={price:.2f} ({(price/curr_ema-1)*100:+.2f}% vs EMA20), "
                        f"RSI={curr_rsi:.0f}"
                    )
        else:
            # --- EXIT ---
            pnl = (price - entry_price) / entry_price
            hold_bars = i - list(day_df.index).index(entry_time)

            above_ema = price > curr_ema * 1.001             # mean-reversion complete (+0.1%)
            rsi_overbought = curr_rsi > 58
            stop_hit = pnl <= -0.015                         # -1.5% stop

            exit_reason = None
            if is_last:
                exit_reason = f"FORCE_CLOSE"
            elif stop_hit:
                exit_reason = f"STOP_LOSS ({pnl:+.1%})"
            elif above_ema:
                exit_reason = f"MEAN_REV_DONE (RSI={curr_rsi:.0f})"
            elif rsi_overbought:
                exit_reason = f"RSI_OVERBOUGHT ({curr_rsi:.0f})"

            if exit_reason:
                trades.append({
                    "entry_time": str(bar_time.strftime("%H:%M")),
                    "exit_time": str(bar_time.strftime("%H:%M")),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(price, 4),
                    "pnl_pct": round(pnl * 100, 3),
                    "hold_bars": hold_bars,
                    "entry_reason": entry_reason,
                    "exit_reason": exit_reason,
                })
                total_pnl_pct += pnl * 100
                position = 0.0
                entry_price = None

    # 6. Summary stats
    open_price = float(day_df["close"].iloc[0])
    close_price = float(day_df["close"].iloc[-1])
    day_return = round((close_price - open_price) / open_price * 100, 3)

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    result["trades"] = trades
    result["summary"] = {
        "n_trades": len(trades),
        "total_pnl_pct": round(total_pnl_pct, 3),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins), 3) if wins else 0,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses), 3) if losses else 0,
        "day_open": round(open_price, 2),
        "day_close": round(close_price, 2),
        "day_return_pct": day_return,
        "n_bars": n_bars,
        "regime": regime,
        "trend_score": regime_info.get("trend_score", 0),
    }

    return result


# ---------------------------------------------------------------------------
# Multi-ticker / multi-date batch runner
# ---------------------------------------------------------------------------

def run_intraday_batch(tickers: list, dates_config: list) -> list:
    """
    Run intraday simulations for all ticker × date combinations.

    dates_config: list of (date_str, interval, label) e.g.:
        [("2026-03-18", "5m", "Last Week"), ("2026-01-09", "1h", "~2 Months Ago")]
    """
    all_results = []
    for date_str, interval, label in dates_config:
        day_results = {
            "date": date_str,
            "label": label,
            "interval": interval,
            "ticker_results": [],
        }
        for ticker in tickers:
            print(f"  Simulating {ticker} on {date_str} ({interval})...")
            r = simulate_day(ticker, date_str, interval)
            day_results["ticker_results"].append(r)
        all_results.append(day_results)
    return all_results


# ---------------------------------------------------------------------------
# Discord embed formatting
# ---------------------------------------------------------------------------

def format_intraday_embeds(day_batch: dict) -> list:
    """
    Format one day's intraday results as a list of Discord embed dicts.
    Returns a summary embed + one detail embed per ticker that had trades.
    """
    date = day_batch["date"]
    label = day_batch["label"]
    interval = day_batch["interval"]
    ticker_results = day_batch["ticker_results"]

    embeds = []

    # Header / summary embed
    lines = []
    for r in ticker_results:
        if r.get("error"):
            lines.append(f"**{r['ticker']}** ⚠️ `{r['error'][:70]}`")
            continue
        s = r["summary"]
        regime_emoji = {"bull": "🟢", "chop": "🟡", "bear": "🔴"}.get(s["regime"], "⚪")
        if s["n_trades"] == 0:
            lines.append(
                f"{regime_emoji} **{r['ticker']}** | Day: {s['day_return_pct']:+.2f}% | "
                f"Bot: ➖ No trades (regime: {s['regime'].upper()})"
            )
        else:
            pnl_emoji = "📈" if s["total_pnl_pct"] > 0 else "📉"
            lines.append(
                f"{regime_emoji} **{r['ticker']}** | Day: {s['day_return_pct']:+.2f}% | "
                f"Bot: {pnl_emoji} {s['total_pnl_pct']:+.2f}% | "
                f"{s['n_trades']} trades | {s['win_rate']:.0f}% win"
            )

    header = {
        "title": f"📊 Intraday Sim — {label} ({date})  [{interval} bars]",
        "description": "\n".join(lines) if lines else "No results",
        "color": 0x9b59b6,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {
            "text": (
                "Entry: price < EMA20 − 0.15%, RSI < 45 | "
                "Exit: price > EMA20 + 0.1%, RSI > 58, or −1.5% stop | "
                "Regime from prior-day EODHD daily bars"
            )
        },
    }
    embeds.append(header)

    # Per-ticker detail embeds (only for tickers with trades)
    for r in ticker_results:
        if r.get("error") or not r.get("trades"):
            continue
        s = r["summary"]
        color = 0x2ecc71 if s["total_pnl_pct"] > 0 else 0xe74c3c

        trade_lines = []
        for t in r["trades"]:
            sign = "+" if t["pnl_pct"] >= 0 else ""
            trade_lines.append(
                f"  {t['entry_time']}→{t['exit_time']} | "
                f"{sign}{t['pnl_pct']:.2f}% ({t['hold_bars']}bars) | "
                f"{t['exit_reason']}"
            )

        rc = r["regime_context"]
        detail = {
            "title": f"{r['ticker']} — {date}  [{interval}]",
            "color": color,
            "fields": [
                {
                    "name": "Daily Context (EODHD)",
                    "value": rc.get("description", "N/A"),
                    "inline": False,
                },
                {
                    "name": "Regime",
                    "value": f"{s['regime'].upper()} (score={s['trend_score']})",
                    "inline": True,
                },
                {
                    "name": "Day Return",
                    "value": f"{s['day_return_pct']:+.2f}%",
                    "inline": True,
                },
                {
                    "name": "Bot P&L",
                    "value": f"{s['total_pnl_pct']:+.2f}%",
                    "inline": True,
                },
                {
                    "name": "Trades",
                    "value": (
                        f"{s['n_trades']} | "
                        f"Win: {s['win_rate']:.0f}% | "
                        f"Avg+{s['avg_win_pct']:.2f}% / {s['avg_loss_pct']:.2f}%"
                    ),
                    "inline": True,
                },
                {
                    "name": "Bars",
                    "value": f"{s['n_bars']} {interval} bars",
                    "inline": True,
                },
                {
                    "name": "Trade Log",
                    "value": (
                        "```\n" + "\n".join(trade_lines) + "\n```"
                        if trade_lines else "No trades"
                    ),
                    "inline": False,
                },
            ],
            "footer": {
                "text": f"JK Four Pillars — Intraday Simulation  |  Data: yfinance {interval}"
            },
        }
        embeds.append(detail)

    return embeds
