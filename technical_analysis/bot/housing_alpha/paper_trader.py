"""
Housing Alpha Paper Trader
============================
Executes housing alpha signals via Alpaca paper trading.
Monthly rebalance — checks signal and adjusts XHB/ITB positions.

Uses the same Alpaca client as the Four Pillars system.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

from technical_analysis.bot.housing_alpha.engine import (
    HousingAlphaEngine,
    HousingSignal,
    HOUSING_TICKERS,
    load_params,
)
from technical_analysis.bot import alpaca_client


STATE_FILE = Path(__file__).parent / "state" / "portfolio.json"

DEFAULT_CASH = 50_000  # $50k starting capital for housing alpha


def _load_state() -> dict:
    """Load housing portfolio state."""
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        "cash": DEFAULT_CASH,
        "positions": {},
        "trade_log": [],
        "last_rebalance": None,
        "created": datetime.now().isoformat(),
    }


def _save_state(state: dict):
    """Save housing portfolio state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _get_price(ticker: str) -> Optional[float]:
    """Get current price for a ticker."""
    try:
        from technical_analysis.backtest.signal_tester import fetch_data
        df = fetch_data(ticker, "5d")
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def _nav(state: dict) -> float:
    """Compute current NAV."""
    nav = state["cash"]
    for ticker, pos in state["positions"].items():
        price = _get_price(ticker) or pos.get("last_price", 0)
        nav += pos["shares"] * price
    return nav


def run_monthly_rebalance(
    tickers: Optional[list[str]] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the monthly housing alpha rebalance.

    1. Compute current housing signals
    2. Compare target allocation vs current
    3. Place Alpaca orders to rebalance
    4. Log everything

    Returns list of trades executed.
    """
    if tickers is None:
        tickers = ["XHB", "ITB"]

    state = _load_state()
    engine = HousingAlphaEngine()
    signals = engine.compute_signals(tickers=tickers)

    if not signals:
        if verbose:
            print("  [housing-trader] No signals available — skipping")
        return []

    nav = _nav(state)
    trades = []

    if verbose:
        print(f"\n  Housing Alpha Paper Trader")
        print(f"  NAV: ${nav:,.0f} | Cash: ${state['cash']:,.0f}")
        print(f"  Regime: {signals[0].regime} | Composite Z: {signals[0].composite_z:+.2f}")
        print(f"  {'─' * 50}")

    # Equal capital allocation across tickers
    per_ticker_capital = nav / len(tickers)

    for signal in signals:
        ticker = signal.ticker
        if ticker not in tickers:
            continue

        price = _get_price(ticker)
        if price is None:
            if verbose:
                print(f"  [{ticker}] No price data — skipping")
            continue

        # Target position
        target_pct = signal.target_position
        target_value = per_ticker_capital * target_pct
        target_shares = target_value / price

        # Current position
        current_pos = state["positions"].get(ticker, {})
        current_shares = current_pos.get("shares", 0)
        current_value = current_shares * price

        # Compute delta
        delta_shares = target_shares - current_shares
        delta_value = abs(delta_shares * price)

        # Minimum rebalance: 5% of per-ticker capital
        min_rebalance = per_ticker_capital * 0.05
        if delta_value < min_rebalance:
            if verbose:
                print(f"  [{ticker}] ${price:.2f} | current {current_shares:.1f} sh (${current_value:,.0f}) "
                      f"→ target {target_shares:.1f} sh (${target_value:,.0f}) | delta ${delta_value:,.0f} < min — SKIP")
            continue

        # Determine action
        if delta_shares > 0:
            action = "BUY"
            side = "buy"
            notional = delta_shares * price
        else:
            action = "SELL"
            side = "sell"
            notional = abs(delta_shares) * price

        if verbose:
            print(f"  [{ticker}] ${price:.2f} | {action} {abs(delta_shares):.1f} shares "
                  f"(${notional:,.0f}) → target {target_pct:.0%} ({target_shares:.1f} sh)")

        # Execute on Alpaca
        alpaca_order = None
        if alpaca_client.is_available():
            if action == "BUY":
                alpaca_order = alpaca_client.place_market_order(
                    ticker, side="buy", notional=round(notional, 2), verbose=verbose,
                )
            elif action == "SELL":
                if target_shares <= 0.01:
                    # Close entire position
                    alpaca_client.close_position(ticker, verbose=verbose)
                else:
                    alpaca_order = alpaca_client.place_market_order(
                        ticker, side="sell", qty=round(abs(delta_shares), 4), verbose=verbose,
                    )

        # Update local state
        if action == "BUY":
            state["cash"] -= notional
            if ticker in state["positions"]:
                old = state["positions"][ticker]
                total_shares = old["shares"] + delta_shares
                # Weighted average entry price
                old_cost = old["shares"] * old["entry_price"]
                new_cost = delta_shares * price
                avg_entry = (old_cost + new_cost) / total_shares if total_shares > 0 else price
                state["positions"][ticker] = {
                    "shares": round(total_shares, 4),
                    "entry_price": round(avg_entry, 2),
                    "last_price": price,
                    "target_pct": target_pct,
                    "regime": signal.regime,
                    "updated": datetime.now().isoformat(),
                }
            else:
                state["positions"][ticker] = {
                    "shares": round(delta_shares, 4),
                    "entry_price": price,
                    "last_price": price,
                    "target_pct": target_pct,
                    "regime": signal.regime,
                    "updated": datetime.now().isoformat(),
                }
        elif action == "SELL":
            state["cash"] += notional
            remaining = current_shares + delta_shares  # delta_shares is negative
            if remaining <= 0.01:
                # Closed out
                entry = current_pos.get("entry_price", price)
                pnl = (price - entry) * current_shares
                state["positions"].pop(ticker, None)
                if verbose:
                    print(f"    Closed {ticker} | P&L: ${pnl:+,.0f} ({(price/entry - 1)*100:+.1f}%)")
            else:
                state["positions"][ticker]["shares"] = round(remaining, 4)
                state["positions"][ticker]["last_price"] = price
                state["positions"][ticker]["target_pct"] = target_pct
                state["positions"][ticker]["updated"] = datetime.now().isoformat()

        # Log trade
        trade = {
            "ticker": ticker,
            "action": action,
            "shares": round(abs(delta_shares), 4),
            "price": price,
            "notional": round(notional, 2),
            "target_pct": target_pct,
            "regime": signal.regime,
            "composite_z": signal.composite_z,
            "timestamp": datetime.now().isoformat(),
            "alpaca_order_id": alpaca_order.get("id", "")[:8] if alpaca_order else None,
        }
        trades.append(trade)
        state["trade_log"].append(trade)

    state["last_rebalance"] = datetime.now().isoformat()
    _save_state(state)

    # Print summary
    if verbose:
        updated_nav = _nav(state)
        print(f"  {'─' * 50}")
        print(f"  Post-rebalance NAV: ${updated_nav:,.0f} | Cash: ${state['cash']:,.0f}")
        print(f"  Positions:")
        for ticker, pos in state["positions"].items():
            price = _get_price(ticker) or pos["last_price"]
            value = pos["shares"] * price
            pnl_pct = (price / pos["entry_price"] - 1) * 100
            print(f"    {ticker}: {pos['shares']:.1f} sh @ ${pos['entry_price']:.2f} "
                  f"→ ${value:,.0f} ({pnl_pct:+.1f}%)")
        if not state["positions"]:
            print(f"    (no positions)")
        print(f"  Trades this cycle: {len(trades)}")

    return trades


def get_status() -> dict:
    """Get current housing portfolio status."""
    state = _load_state()
    nav = _nav(state)
    positions = {}
    for ticker, pos in state["positions"].items():
        price = _get_price(ticker) or pos.get("last_price", 0)
        value = pos["shares"] * price
        pnl_pct = (price / pos["entry_price"] - 1) * 100 if pos["entry_price"] > 0 else 0
        positions[ticker] = {
            "shares": pos["shares"],
            "entry_price": pos["entry_price"],
            "current_price": round(price, 2),
            "value": round(value, 2),
            "pnl_pct": round(pnl_pct, 2),
            "target_pct": pos.get("target_pct", 0),
            "regime": pos.get("regime", "?"),
        }
    return {
        "nav": round(nav, 2),
        "cash": round(state["cash"], 2),
        "positions": positions,
        "last_rebalance": state.get("last_rebalance"),
        "trade_count": len(state.get("trade_log", [])),
    }
