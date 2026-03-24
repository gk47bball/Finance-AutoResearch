"""
Housing Alpha Engine
=====================
Generates monthly trading signals for housing-exposed ETFs based on
composite housing indicators from FRED + Zillow data.

Signal flow:
  1. Build housing dataset (monthly frequency)
  2. Compute composite housing signal (5 sub-indicators)
  3. Determine regime: HOUSING_BULL, HOUSING_BEAR, HOUSING_NEUTRAL
  4. Map regime to position sizing for each tradeable ticker
  5. Apply rate-regime override (rising rates → reduce exposure)

Tradeable tickers:
  XHB  — SPDR S&P Homebuilders ETF (D.R. Horton, Lennar, NVR...)
  ITB  — iShares Home Construction ETF
  XLRE — Real Estate Select Sector SPDR
  VNQ  — Vanguard Real Estate ETF
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from technical_analysis.bot.housing_alpha.data_fetcher import (
    build_housing_dataset,
    get_housing_snapshot,
)
from technical_analysis.bot.housing_alpha.indicators import (
    compute_housing_composite,
)


PARAMS_FILE = Path(__file__).parent / "state" / "best_params.json"

# ---------------------------------------------------------------------------
# Default parameters (mutable genome for AutoResearch)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # Composite indicator weights
    "weight_activity": 0.30,
    "weight_affordability": 0.20,
    "weight_supply_demand": 0.15,
    "weight_price_momentum": 0.15,
    "weight_rate_regime": 0.20,

    # Indicator computation
    "activity_mom_window": 3,       # months for activity momentum
    "activity_zscore_window": 36,   # months for activity z-score
    "afford_zscore_window": 36,
    "supply_zscore_window": 36,
    "price_mom_window": 6,
    "price_zscore_window": 36,
    "rate_lookback": 6,
    "rate_zscore_window": 36,
    "composite_zscore_window": 36,

    # Regime thresholds (on composite z-score)
    "bull_threshold": 0.5,          # composite z > this → HOUSING_BULL
    "bear_threshold": -0.5,         # composite z < this → HOUSING_BEAR

    # Position sizing
    "bull_position": 0.80,          # target allocation in HOUSING_BULL
    "neutral_position": 0.40,       # target allocation in HOUSING_NEUTRAL
    "bear_position": 0.10,          # target allocation in HOUSING_BEAR (small, not zero — contrarian)

    # Risk management
    "max_position": 1.0,            # max position per ticker
    "rebalance_threshold": 0.10,    # min position change to trigger rebalance

    # Rate override: if mortgage rates rising fast, reduce exposure
    "rate_override_threshold": 1.5, # rate_regime z-score below -this → force reduce
    "rate_override_reduction": 0.50,# multiply position by this when rate override active
}


def load_params() -> dict:
    """Load current best params or defaults."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            saved = json.load(f)
        # Merge with defaults (in case new params added)
        merged = DEFAULT_PARAMS.copy()
        merged.update(saved)
        return merged
    return DEFAULT_PARAMS.copy()


def save_params(params: dict):
    """Save optimized params."""
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class HousingSignal:
    """Signal output for a single ticker at a point in time."""
    ticker: str
    date: str
    regime: str                          # HOUSING_BULL, HOUSING_NEUTRAL, HOUSING_BEAR
    composite_z: float = 0.0             # composite z-score
    activity_z: float = 0.0
    affordability_z: float = 0.0
    supply_demand_z: float = 0.0
    price_momentum_z: float = 0.0
    rate_regime_z: float = 0.0
    target_position: float = 0.0         # recommended allocation (0-1)
    rate_override_active: bool = False
    action: str = "HOLD"                 # BUY, SELL, HOLD, REDUCE
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Which tickers respond to housing signals, and how
HOUSING_TICKERS = {
    "XHB": {"sensitivity": 1.0, "name": "SPDR Homebuilders"},
    "ITB": {"sensitivity": 1.0, "name": "iShares Home Construction"},
    "XLRE": {"sensitivity": 0.7, "name": "Real Estate Select"},
    "VNQ": {"sensitivity": 0.6, "name": "Vanguard Real Estate"},
}


class HousingAlphaEngine:
    """
    Main engine: computes housing indicators and generates signals.

    Usage:
        engine = HousingAlphaEngine()
        signals = engine.compute_signals()  # returns list of HousingSignal
        # or for backtesting:
        positions = engine.compute_historical()  # returns DataFrame
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or load_params()
        self._housing_df: Optional[pd.DataFrame] = None
        self._indicators: Optional[pd.DataFrame] = None

    def _ensure_data(self, start: str = "2000-01-01"):
        """Lazy-load housing data."""
        if self._housing_df is None:
            self._housing_df = build_housing_dataset(start=start)
            self._indicators = compute_housing_composite(self._housing_df, self.params)

    def get_regime(self, composite_z: float) -> str:
        """Map composite z-score to regime."""
        if composite_z >= self.params["bull_threshold"]:
            return "HOUSING_BULL"
        elif composite_z <= self.params["bear_threshold"]:
            return "HOUSING_BEAR"
        return "HOUSING_NEUTRAL"

    def get_target_position(self, regime: str, ticker: str) -> float:
        """Get target position for a ticker given the regime."""
        sensitivity = HOUSING_TICKERS.get(ticker, {}).get("sensitivity", 0.5)

        if regime == "HOUSING_BULL":
            base = self.params["bull_position"]
        elif regime == "HOUSING_BEAR":
            base = self.params["bear_position"]
        else:
            base = self.params["neutral_position"]

        return min(base * sensitivity, self.params["max_position"])

    def compute_signals(
        self,
        tickers: Optional[list[str]] = None,
    ) -> list[HousingSignal]:
        """
        Compute current housing signals for all tickers.
        Returns list of HousingSignal.
        """
        self._ensure_data()

        if tickers is None:
            tickers = list(HOUSING_TICKERS.keys())

        ind = self._indicators
        if ind is None or ind.empty:
            return []

        # Get latest values
        latest = ind.iloc[-1]
        composite_z = float(latest.get("composite_signal", 0))
        regime = self.get_regime(composite_z)

        # Check rate override
        rate_z = float(latest.get("rate_regime", 0))
        rate_override = rate_z < -self.params["rate_override_threshold"]

        signals = []
        for ticker in tickers:
            if ticker not in HOUSING_TICKERS:
                continue

            target = self.get_target_position(regime, ticker)

            # Apply rate override
            if rate_override and regime != "HOUSING_BEAR":
                target *= self.params["rate_override_reduction"]

            signal = HousingSignal(
                ticker=ticker,
                date=ind.index[-1].strftime("%Y-%m-%d"),
                regime=regime,
                composite_z=round(composite_z, 3),
                activity_z=round(float(latest.get("activity_momentum", 0)), 3),
                affordability_z=round(float(latest.get("affordability_index", 0)), 3),
                supply_demand_z=round(float(latest.get("supply_demand", 0)), 3),
                price_momentum_z=round(float(latest.get("price_momentum", 0)), 3),
                rate_regime_z=round(float(latest.get("rate_regime", 0)), 3),
                target_position=round(target, 3),
                rate_override_active=rate_override,
                action="HOLD",  # actual action determined by comparing to current position
                reason=f"{regime} (composite z={composite_z:+.2f})"
                       + (f" | RATE OVERRIDE: rate z={rate_z:+.2f}" if rate_override else ""),
            )
            signals.append(signal)

        return signals

    def compute_historical(
        self,
        ticker: str = "XHB",
        start: str = "2005-01-01",
    ) -> pd.DataFrame:
        """
        Compute historical positions for backtesting.
        Returns DataFrame with columns: composite_z, regime, position, and all sub-indicators.
        Monthly frequency, aligned to the housing data calendar.
        """
        self._ensure_data(start=start)

        ind = self._indicators
        if ind is None or ind.empty:
            return pd.DataFrame()

        result = ind.copy()
        sensitivity = HOUSING_TICKERS.get(ticker, {}).get("sensitivity", 0.5)

        # Compute regime
        bull_t = self.params["bull_threshold"]
        bear_t = self.params["bear_threshold"]

        composite = result["composite_signal"].fillna(0)

        result["regime"] = "HOUSING_NEUTRAL"
        result.loc[composite >= bull_t, "regime"] = "HOUSING_BULL"
        result.loc[composite <= bear_t, "regime"] = "HOUSING_BEAR"

        # Compute position
        result["position"] = self.params["neutral_position"] * sensitivity
        result.loc[result["regime"] == "HOUSING_BULL", "position"] = (
            self.params["bull_position"] * sensitivity
        )
        result.loc[result["regime"] == "HOUSING_BEAR", "position"] = (
            self.params["bear_position"] * sensitivity
        )

        # Rate override
        rate_z = result.get("rate_regime", pd.Series(0, index=result.index))
        override_mask = (
            (rate_z < -self.params["rate_override_threshold"])
            & (result["regime"] != "HOUSING_BEAR")
        )
        result.loc[override_mask, "position"] *= self.params["rate_override_reduction"]

        # Clip
        result["position"] = result["position"].clip(0, self.params["max_position"])

        return result

    def print_dashboard(self):
        """Print a text dashboard of current housing signals."""
        signals = self.compute_signals()
        if not signals:
            print("  No housing data available.")
            return

        s = signals[0]  # all tickers share the same underlying signal

        print()
        print("  ╔══════════════════════════════════════════════════════════════╗")
        print("  ║               HOUSING ALPHA DASHBOARD                       ║")
        print(f"  ║  Date: {s.date}                                          ║")
        print(f"  ║  Regime: {s.regime:<20s}  Composite Z: {s.composite_z:+.2f}          ║")
        print("  ╠══════════════════════════════════════════════════════════════╣")
        print("  ║  Sub-Indicators:                                            ║")
        print(f"  ║    Activity Momentum:    {s.activity_z:+.2f}                           ║")
        print(f"  ║    Affordability (inv):   {s.affordability_z:+.2f}                           ║")
        print(f"  ║    Supply/Demand (inv):   {s.supply_demand_z:+.2f}                           ║")
        print(f"  ║    Price Momentum:        {s.price_momentum_z:+.2f}                           ║")
        print(f"  ║    Rate Regime:           {s.rate_regime_z:+.2f}                           ║")
        if s.rate_override_active:
            print("  ║    ⚠️  RATE OVERRIDE ACTIVE — positions reduced              ║")
        print("  ╠══════════════════════════════════════════════════════════════╣")
        print("  ║  Ticker    Target Pos    Sensitivity                        ║")
        print("  ║  ──────    ──────────    ───────────                        ║")
        for sig in signals:
            sens = HOUSING_TICKERS.get(sig.ticker, {}).get("sensitivity", 0.5)
            name = HOUSING_TICKERS.get(sig.ticker, {}).get("name", "")
            print(f"  ║  {sig.ticker:<8s}  {sig.target_position:>6.0%}         {sens:.1f}x  {name:<20s} ║")
        print("  ╚══════════════════════════════════════════════════════════════╝")
        print()
