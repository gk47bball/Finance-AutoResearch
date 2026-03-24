"""
Paper Trading Engine
=====================
Simulates live trading with the Four Pillars strategy.
Tracks positions, P&L, trade log, and portfolio state.
Persists state to disk so the bot can resume after restarts.

Alpaca Integration:
  When Alpaca credentials are configured, all BUY/SELL/REDUCE signals are
  also executed as real paper trades on the Alpaca paper account. The local
  portfolio.json remains the source of truth for signals/tracking; Alpaca
  is the execution venue.
"""

import json
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import pandas as pd

from technical_analysis.bot.pillars import FourPillarsEngine, PillarSnapshot, TradeSignal
from technical_analysis.bot import alpaca_client


STATE_DIR = Path(__file__).parent / "state"


@dataclass
class Position:
    """A single open position."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: str              # ISO format
    current_price: float
    high_water_mark: float
    position_pct: float          # target allocation %
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """A completed trade."""
    ticker: str
    action: str
    shares: float
    price: float
    timestamp: str
    reason: str
    position_pct: float
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class PortfolioState:
    """Full portfolio state, persistable to JSON."""
    cash: float = 100_000.0
    initial_capital: float = 100_000.0
    positions: dict = field(default_factory=dict)   # ticker → Position dict
    trade_log: list = field(default_factory=list)    # list of Trade dicts
    daily_nav: list = field(default_factory=list)    # list of {date, nav}
    created_at: str = ""
    updated_at: str = ""
    # Circuit breaker state
    peak_nav: float = 0.0
    circuit_breaker_active: bool = False


class PaperTrader:
    """
    Paper trading engine.

    Usage:
        trader = PaperTrader(initial_capital=100000)
        trader.run_daily()  # Call once per trading day
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        tickers: list[str] = None,
        state_file: str = "portfolio.json",
    ):
        self.tickers = tickers or ["SPY", "DIA"]
        self.engine = FourPillarsEngine(period="2y")
        self.state_file = STATE_DIR / state_file

        # Load or create state
        if self.state_file.exists():
            self.state = self._load_state()
        else:
            self.state = PortfolioState(
                cash=initial_capital,
                initial_capital=initial_capital,
                created_at=datetime.now().isoformat(),
            )

    def _load_state(self) -> PortfolioState:
        with open(self.state_file) as f:
            data = json.load(f)
        # Handle new fields gracefully for existing state files
        valid_fields = {f.name for f in PortfolioState.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return PortfolioState(**filtered)

    def _save_state(self):
        self.state.updated_at = datetime.now().isoformat()
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f, indent=2, default=str)

    @property
    def nav(self) -> float:
        """Current net asset value."""
        positions_value = sum(
            p["current_price"] * p["shares"]
            for p in self.state.positions.values()
        )
        return self.state.cash + positions_value

    @property
    def total_return_pct(self) -> float:
        return (self.nav / self.state.initial_capital - 1) * 100

    def run_daily(self, verbose: bool = True) -> list[TradeSignal]:
        """
        Run the daily trading cycle with cross-sectional laggard weighting
        and portfolio-level circuit breaker.
        """
        # --- Circuit breaker check ---
        if self._check_circuit_breaker(verbose):
            self._save_state()
            if verbose:
                self._print_summary()
            return []

        signals = []
        snapshots = []

        # Phase 1: Compute all snapshots
        for ticker in self.tickers:
            try:
                snapshot = self.engine.compute(ticker)
                snapshots.append(snapshot)
            except Exception as e:
                if verbose:
                    print(f"  [{ticker}] Error computing snapshot: {e}")

        # Phase 2: Rank cross-sectionally (laggards get priority)
        if snapshots:
            snapshots = FourPillarsEngine.rank_snapshots(snapshots)

        # Phase 3: Compute allocation weights (laggards get more capital)
        n = len(snapshots)
        if n > 1:
            raw_weights = {s.ticker: (n - (s.laggard_rank - 1)) for s in snapshots}
            total_w = sum(raw_weights.values())
            alloc_weights = {t: w / total_w for t, w in raw_weights.items()}
        else:
            alloc_weights = {s.ticker: 1.0 for s in snapshots}

        # Phase 4: Process each ticker with its cross-sectional weight
        for snapshot in snapshots:
            try:
                signal = self._process_ticker_weighted(
                    snapshot,
                    alloc_weights.get(snapshot.ticker, 1.0 / max(1, len(self.tickers))),
                    verbose,
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                if verbose:
                    print(f"  [{snapshot.ticker}] Error: {e}")

        # Update NAV
        self._update_positions()
        self.state.daily_nav.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "nav": round(self.nav, 2),
        })

        self._save_state()

        if verbose:
            self._print_summary()

        return signals

    def _check_circuit_breaker(self, verbose: bool) -> bool:
        """Portfolio-level max drawdown circuit breaker.
        Returns True if circuit breaker is active (skip all entries)."""
        current_nav = self.nav

        # Track peak NAV
        if current_nav > self.state.peak_nav:
            self.state.peak_nav = current_nav

        if self.state.peak_nav <= 0:
            return False

        drawdown = (current_nav - self.state.peak_nav) / self.state.peak_nav

        # Activate at -8% drawdown
        if drawdown <= -0.08 and not self.state.circuit_breaker_active:
            self.state.circuit_breaker_active = True
            if verbose:
                print(f"\n  *** CIRCUIT BREAKER ACTIVATED: drawdown {drawdown:.1%} exceeds -8% ***")
                print(f"  Closing all positions. Will re-enter when drawdown recovers to -4%")
            # Close all positions
            for ticker in list(self.state.positions.keys()):
                pos = self.state.positions[ticker]
                shares = pos["shares"]
                price = pos["current_price"]
                proceeds = shares * price
                self.state.cash += proceeds
                pnl = (price - pos["entry_price"]) * shares
                self.state.trade_log.append({
                    "ticker": ticker, "action": "CIRCUIT_BREAKER",
                    "shares": round(shares, 4), "price": round(price, 2),
                    "timestamp": datetime.now().isoformat(),
                    "reason": f"CIRCUIT BREAKER: portfolio drawdown {drawdown:.1%}",
                    "position_pct": 0.0,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((price / pos["entry_price"] - 1) * 100, 2),
                })
                self._alpaca_close(ticker, verbose)
            self.state.positions.clear()
            return True

        # Reset at -4% recovery
        if self.state.circuit_breaker_active and drawdown > -0.04:
            self.state.circuit_breaker_active = False
            if verbose:
                print(f"\n  CIRCUIT BREAKER RESET: drawdown recovered to {drawdown:.1%}")
            return False

        return self.state.circuit_breaker_active

    def _correlation_adjustment(self, ticker: str, target_pct: float) -> float:
        """Reduce allocation when highly correlated positions are already large."""
        CORR_PAIRS = {
            ("DIA", "SPY"): 0.95, ("QQQ", "SPY"): 0.85,
            ("DIA", "QQQ"): 0.80, ("IWM", "SPY"): 0.80,
            ("IWM", "QQQ"): 0.75, ("DIA", "IWM"): 0.75,
        }

        total_correlated_exposure = 0.0
        for other_ticker, pos in self.state.positions.items():
            if other_ticker == ticker:
                continue
            pair = tuple(sorted([ticker, other_ticker]))
            corr = CORR_PAIRS.get(pair, 0.5)
            if corr > 0.8:
                other_exposure = pos["position_pct"]
                total_correlated_exposure += other_exposure * corr

        max_effective = 1.5  # max combined effective exposure
        if total_correlated_exposure + target_pct > max_effective:
            adjusted = max(0.25, max_effective - total_correlated_exposure)
            return min(target_pct, adjusted)

        return target_pct

    def _process_ticker_weighted(self, snapshot: PillarSnapshot, weight: float, verbose: bool) -> Optional[TradeSignal]:
        """Process a ticker with cross-sectional weight applied to allocation."""
        ticker = snapshot.ticker

        # Current position state
        pos = self.state.positions.get(ticker)
        current_pct = pos["position_pct"] if pos else 0.0
        entry_price = pos["entry_price"] if pos else None
        entry_date = pd.Timestamp(pos["entry_date"]) if pos else None
        hwm = pos["high_water_mark"] if pos else None

        # Get current price
        from technical_analysis.backtest.signal_tester import fetch_data
        df = fetch_data(ticker, "5d")
        current_price = float(df["Close"].iloc[-1])

        # Generate signal
        signal = self.engine.generate_signal(
            snapshot=snapshot,
            current_position=current_pct,
            entry_price=entry_price,
            entry_date=entry_date,
            high_water_mark=hwm,
            current_price=current_price,
        )

        if verbose:
            self._print_snapshot(snapshot, current_price, current_pct)

        # Execute signal with cross-sectional weight
        if signal.action in ("BUY", "REDUCE", "SELL"):
            self._execute(signal, current_price, verbose, weight=weight)

        return signal

    def _process_ticker(self, ticker: str, verbose: bool) -> Optional[TradeSignal]:
        """Process a single ticker (backward-compatible, equal-weight)."""
        snapshot = self.engine.compute(ticker)
        return self._process_ticker_weighted(
            snapshot, 1.0 / len(self.tickers), verbose
        )

    def _execute(self, signal: TradeSignal, price: float, verbose: bool, weight: float = None):
        """Execute a trade signal — updates local portfolio AND places Alpaca paper order."""
        ticker = signal.ticker
        target_pct = signal.position_pct

        # Correlation-aware adjustment
        target_pct = self._correlation_adjustment(ticker, target_pct)

        portfolio_value = self.nav

        # Target dollar allocation (cross-sectional weight or equal-weight)
        alloc_fraction = weight if weight is not None else (1.0 / len(self.tickers))
        target_value = portfolio_value * target_pct * alloc_fraction
        current_pos = self.state.positions.get(ticker)
        current_value = current_pos["shares"] * price if current_pos else 0.0
        delta_value = target_value - current_value

        # Minimum rebalance threshold: 2% of NAV (prevents churn)
        min_rebalance = portfolio_value * 0.02
        if abs(delta_value) < min_rebalance:
            return

        shares_delta = delta_value / price

        if signal.action == "BUY" or (signal.action == "REDUCE" and delta_value > 0):
            # Buying
            cost = abs(delta_value)
            if cost > self.state.cash:
                shares_delta = self.state.cash / price
                cost = self.state.cash

            self.state.cash -= cost

            if current_pos:
                # Add to position
                old_shares = current_pos["shares"]
                new_shares = old_shares + shares_delta
                # Weighted average entry
                current_pos["entry_price"] = (
                    (old_shares * current_pos["entry_price"] + shares_delta * price) / new_shares
                )
                current_pos["shares"] = new_shares
                current_pos["position_pct"] = target_pct
                current_pos["current_price"] = price
                current_pos["high_water_mark"] = max(current_pos["high_water_mark"], price)
            else:
                self.state.positions[ticker] = {
                    "ticker": ticker,
                    "shares": shares_delta,
                    "entry_price": price,
                    "entry_date": datetime.now().isoformat(),
                    "current_price": price,
                    "high_water_mark": price,
                    "position_pct": target_pct,
                    "unrealized_pnl": 0.0,
                    "unrealized_pnl_pct": 0.0,
                }

            # ── Alpaca: place buy order ──────────────────────────────
            self._alpaca_buy(ticker, abs(shares_delta), cost, verbose)

        elif signal.action in ("SELL", "REDUCE"):
            # Selling
            if current_pos:
                shares_to_sell = min(abs(shares_delta), current_pos["shares"])
                proceeds = shares_to_sell * price
                self.state.cash += proceeds

                pnl = (price - current_pos["entry_price"]) * shares_to_sell
                pnl_pct = (price / current_pos["entry_price"] - 1) * 100

                current_pos["shares"] -= shares_to_sell
                fully_closed = current_pos["shares"] < 0.01

                if fully_closed:
                    del self.state.positions[ticker]
                else:
                    current_pos["position_pct"] = target_pct
                    current_pos["current_price"] = price

                # ── Alpaca: sell shares ──────────────────────────────
                if fully_closed:
                    self._alpaca_close(ticker, verbose)
                else:
                    self._alpaca_sell(ticker, shares_to_sell, verbose)

                # Log trade
                self.state.trade_log.append({
                    "ticker": ticker,
                    "action": signal.action,
                    "shares": round(shares_to_sell, 4),
                    "price": round(price, 2),
                    "timestamp": datetime.now().isoformat(),
                    "reason": signal.reason,
                    "position_pct": target_pct,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                })

                if verbose:
                    emoji_pnl = "+" if pnl >= 0 else ""
                    print(f"    EXECUTED: {signal.action} {shares_to_sell:.1f} shares @ ${price:.2f} | P&L: {emoji_pnl}${pnl:.2f} ({emoji_pnl}{pnl_pct:.1f}%)")
                return

        # Log buy trade
        self.state.trade_log.append({
            "ticker": ticker,
            "action": signal.action,
            "shares": round(abs(shares_delta), 4),
            "price": round(price, 2),
            "timestamp": datetime.now().isoformat(),
            "reason": signal.reason,
            "position_pct": target_pct,
        })

        if verbose:
            print(f"    EXECUTED: {signal.action} {abs(shares_delta):.1f} shares @ ${price:.2f}")

    # ------------------------------------------------------------------
    # Alpaca execution helpers
    # ------------------------------------------------------------------

    def _alpaca_buy(self, ticker: str, shares: float, notional: float, verbose: bool):
        """Place an Alpaca paper buy order."""
        if not alpaca_client.is_available():
            return
        order = alpaca_client.place_market_order(
            ticker=ticker,
            side="buy",
            notional=notional,
            verbose=verbose,
        )
        if order and verbose:
            print(f"    [alpaca] BUY {ticker} ${notional:.0f} notional | order_id={order.get('id', '?')[:8]}")

    def _alpaca_sell(self, ticker: str, shares: float, verbose: bool):
        """Place an Alpaca paper sell order for a partial position."""
        if not alpaca_client.is_available():
            return
        order = alpaca_client.place_market_order(
            ticker=ticker,
            side="sell",
            qty=shares,
            verbose=verbose,
        )
        if order and verbose:
            print(f"    [alpaca] SELL {ticker} {shares:.1f} shares | order_id={order.get('id', '?')[:8]}")

    def _alpaca_close(self, ticker: str, verbose: bool):
        """Close an entire Alpaca paper position."""
        if not alpaca_client.is_available():
            return
        closed = alpaca_client.close_position(ticker, verbose=verbose)
        if closed and verbose:
            print(f"    [alpaca] CLOSED {ticker} — full position")

    def _update_positions(self):
        """Update current prices and P&L for all positions."""
        from technical_analysis.backtest.signal_tester import fetch_data

        for ticker, pos in list(self.state.positions.items()):
            try:
                df = fetch_data(ticker, "5d")
                price = float(df["Close"].iloc[-1])
                pos["current_price"] = price
                pos["high_water_mark"] = max(pos["high_water_mark"], price)
                pos["unrealized_pnl"] = round((price - pos["entry_price"]) * pos["shares"], 2)
                pos["unrealized_pnl_pct"] = round((price / pos["entry_price"] - 1) * 100, 2)
            except Exception:
                pass

    def _print_snapshot(self, snap: PillarSnapshot, price: float, current_pct: float):
        """Print pillar readings for a ticker."""
        print(f"\n  [{snap.ticker}] ${price:.2f} — {snap.signal_label}")
        print(f"    P1 Regime:   trend_score={snap.trend_score_raw:+.0f} → {snap.regime.upper()}")
        print(f"    P2 Timing:   z_hybrid z={snap.z_hybrid_zscore:+.2f} → {snap.timing_signal}")
        print(f"    P3 Momentum: hybrid_osc={snap.hybrid_osc_raw:+.3f} (slope={snap.hybrid_osc_slope:+.4f}) → {'CONFIRMING' if snap.momentum_confirming else 'not confirming'}")
        print(f"    P4 Volume:   ve_rsi={snap.ve_rsi_raw:.1f} vol_ratio={snap.volume_ratio:.2f}x → {'CONFIRMING' if snap.volume_confirming else 'not confirming'}")
        print(f"    P5 X-Sect:   mmrsi={snap.multimac_rsi_score:.2f} → {'CONFIRMING' if snap.cross_sectional_confirming else 'not confirming'}")
        print(f"    VIX z={snap.vix_zscore:+.2f} | RRF={snap.rrf_value:.1f} | VolRatio={snap.vol_ratio:.2f} | Baseline={snap.adaptive_baseline:.0%}")
        print(f"    Target: {snap.position_pct:.0%} (current: {current_pct:.0%}) | Confidence: {snap.confidence:.0%} ({snap.pillars_confirming}/5 pillars)")

    def _print_summary(self):
        """Print portfolio summary."""
        print(f"\n  === PORTFOLIO ===")
        print(f"  NAV: ${self.nav:,.2f} | Cash: ${self.state.cash:,.2f} | Return: {self.total_return_pct:+.2f}%")
        if self.state.positions:
            print(f"  Positions:")
            for ticker, pos in self.state.positions.items():
                pnl_str = f"{'+'if pos['unrealized_pnl']>=0 else ''}${pos['unrealized_pnl']:,.2f} ({'+'if pos['unrealized_pnl_pct']>=0 else ''}{pos['unrealized_pnl_pct']:.1f}%)"
                print(f"    {ticker}: {pos['shares']:.1f} shares @ ${pos['entry_price']:.2f} → ${pos['current_price']:.2f} | {pnl_str}")
        print(f"  Trades today: {sum(1 for t in self.state.trade_log if t['timestamp'][:10] == datetime.now().strftime('%Y-%m-%d'))}")
        print(f"  Total trades: {len(self.state.trade_log)}")
