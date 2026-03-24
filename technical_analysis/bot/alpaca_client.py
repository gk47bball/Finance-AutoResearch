"""
Shared Alpaca Paper Trading Client
====================================
HTTP-based client for Alpaca paper trading API.
Used by both the Four Pillars paper trader and the Trump watcher.
"""

import os
import requests
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets/v2"
_ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_API_SECRET", "")


def _headers() -> dict:
    return {
        "APCA-API-KEY-ID": _ALPACA_KEY,
        "APCA-API-SECRET-KEY": _ALPACA_SECRET,
    }


def is_available() -> bool:
    """Check if Alpaca credentials are configured."""
    return bool(_ALPACA_KEY and _ALPACA_SECRET and _ALPACA_KEY != "your_key_here")


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

def get_account() -> Optional[dict]:
    """Get Alpaca paper account info."""
    if not is_available():
        return None
    try:
        r = requests.get(f"{ALPACA_PAPER_BASE}/account", headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  [alpaca] Account fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def place_market_order(
    ticker: str,
    side: str = "buy",
    notional: Optional[float] = None,
    qty: Optional[float] = None,
    verbose: bool = True,
) -> Optional[dict]:
    """
    Place a market order on Alpaca paper account.

    Args:
        ticker: Symbol to trade
        side: "buy" or "sell"
        notional: Dollar amount (mutually exclusive with qty)
        qty: Number of shares (mutually exclusive with notional)
        verbose: Print debug info

    Returns:
        Order response dict, or None on failure
    """
    if not is_available():
        if verbose:
            print("  [alpaca] No API keys configured — skipping order")
        return None

    body = {
        "symbol": ticker,
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if notional is not None:
        body["notional"] = str(round(notional, 2))
    elif qty is not None:
        body["qty"] = str(round(qty, 4))
    else:
        if verbose:
            print("  [alpaca] Must specify notional or qty")
        return None

    try:
        r = requests.post(
            f"{ALPACA_PAPER_BASE}/orders",
            headers=_headers(),
            json=body,
            timeout=10,
        )
        if r.status_code == 422 and notional is not None:
            # Notional not supported for this symbol — fall back to 1 share
            if verbose:
                print(f"  [alpaca] Notional rejected for {ticker} — falling back to qty=1")
            body.pop("notional", None)
            body["qty"] = "1"
            r = requests.post(
                f"{ALPACA_PAPER_BASE}/orders",
                headers=_headers(),
                json=body,
                timeout=10,
            )
        r.raise_for_status()
        order = r.json()
        if verbose:
            print(f"  [alpaca] Order placed: {side.upper()} {ticker} | order_id={order.get('id', '?')[:8]}")
        return order
    except Exception as e:
        if verbose:
            print(f"  [alpaca] Order failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def get_positions() -> list:
    """Get all open positions from Alpaca."""
    if not is_available():
        return []
    try:
        r = requests.get(f"{ALPACA_PAPER_BASE}/positions", headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def get_position(ticker: str) -> Optional[dict]:
    """Get a specific position from Alpaca."""
    if not is_available():
        return None
    try:
        r = requests.get(f"{ALPACA_PAPER_BASE}/positions/{ticker}", headers=_headers(), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_position_pnl(ticker: str) -> Optional[float]:
    """Get unrealized P&L percentage for a position."""
    pos = get_position(ticker)
    if pos:
        try:
            return float(pos.get("unrealized_plpc", 0))
        except (TypeError, ValueError):
            return None
    return None


def close_position(ticker: str, verbose: bool = True) -> bool:
    """Close an entire position on Alpaca."""
    if not is_available():
        return False
    try:
        r = requests.delete(
            f"{ALPACA_PAPER_BASE}/positions/{ticker}",
            headers=_headers(),
            timeout=10,
        )
        if r.status_code in (200, 204):
            if verbose:
                print(f"  [alpaca] Closed position: {ticker}")
            return True
        if verbose:
            print(f"  [alpaca] Close failed for {ticker}: {r.status_code} {r.text[:200]}")
        return False
    except Exception as e:
        if verbose:
            print(f"  [alpaca] Close error for {ticker}: {e}")
        return False


def close_all_positions(verbose: bool = True) -> int:
    """Close all open positions. Returns count of positions closed."""
    positions = get_positions()
    closed = 0
    for pos in positions:
        ticker = pos.get("symbol", "")
        if close_position(ticker, verbose=verbose):
            closed += 1
    return closed
