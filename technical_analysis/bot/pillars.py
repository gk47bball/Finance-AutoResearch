"""
Four Pillars Signal Engine
===========================
Computes the four orthogonal JK pillars and generates position signals.

Pillar 1: REGIME     — trend_score (-5 to +5) → bull / chop / bear
Pillar 2: TIMING     — z_hybrid z-score → oversold / neutral / overbought
Pillar 3: MOMENTUM   — hybrid_osc vs signal line → confirming / not
Pillar 4: VOLUME     — ve_rsi level + volume ratio → confirming / not
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from technical_analysis.indicators.jk_indicators import INDICATOR_REGISTRY
from technical_analysis.backtest.ta_backtest import _normalize_signal
from technical_analysis.backtest.signal_tester import fetch_data


# ---------------------------------------------------------------------------
# Pillar outputs
# ---------------------------------------------------------------------------

@dataclass
class PillarSnapshot:
    """Point-in-time reading of all four pillars."""
    timestamp: pd.Timestamp
    ticker: str

    # Pillar 1: Regime
    trend_score_raw: float       # -5 to +5
    regime: str                  # "bull", "chop", "bear"

    # Pillar 2: Timing
    z_hybrid_raw: float
    z_hybrid_zscore: float
    timing_signal: str           # "deep_oversold", "oversold", "neutral", "overbought"

    # Pillar 3: Momentum
    hybrid_osc_raw: float
    hybrid_osc_signal: float     # signal line value
    hybrid_osc_slope: float      # 3-bar slope of hybrid_osc
    momentum_confirming: bool

    # Pillar 4: Volume
    ve_rsi_raw: float
    ve_rsi_zscore: float
    volume_ratio: float          # current vol / 65-day avg
    volume_confirming: bool

    # Derived
    position_pct: float          # 0.0, 0.25, 0.50, 0.75, 1.0
    signal_label: str            # "STRONG_BUY", "BUY", "HOLD", "REDUCE", "FLAT"
    confidence: float            # 0-1
    pillars_confirming: int      # 0-4

    # Cross-sectional laggard score (from 2007 paper: "Leaders vs Laggards")
    # multimac_rsi combines multi-period EMA convergence + asymmetric RSI(5).
    # More negative = more "washed out" relative to its own trend history.
    # Used for ranking across tickers: the most negative score = highest priority buy.
    multimac_rsi_score: float = 0.0
    laggard_rank: Optional[int] = None   # rank within a multi-ticker scan (1=most lagging)

    # --- New fields (improvements March 2026) ---
    cross_sectional_confirming: bool = False   # multimac_rsi deeply lagging → higher conviction
    vix_zscore: float = 0.0                    # VIX z-score (>2 = fear spike)
    rrf_value: float = 12.5                    # Retracement/Reversal Factor
    vol_ratio: float = 1.0                     # realized vol / 252d median vol
    adaptive_baseline: float = 0.50            # volatility-adjusted baseline used


@dataclass
class TradeSignal:
    """Actionable trade signal from the Four Pillars engine."""
    timestamp: pd.Timestamp
    ticker: str
    action: str                  # "BUY", "SELL", "HOLD", "STOP_LOSS", "TRAIL_STOP", "TIME_STOP"
    position_pct: float          # target position 0-1
    entry_price: Optional[float]
    stop_price: Optional[float]
    trail_price: Optional[float]
    reason: str
    snapshot: PillarSnapshot


# ---------------------------------------------------------------------------
# Pillar Engine
# ---------------------------------------------------------------------------

class FourPillarsEngine:
    """
    Computes the Four Pillars strategy signals.

    Usage:
        engine = FourPillarsEngine()
        snapshot = engine.compute(ticker="SPY")
        signal = engine.generate_signal(snapshot, current_position)
    """

    # Regime thresholds
    BULL_THRESHOLD = 2
    BEAR_THRESHOLD = -2

    # Timing thresholds (z-score of z_hybrid)
    # NOTE: These class-level defaults are intentionally kept as the conservative
    # originals. The engine auto-loads state/best_params.json on init, which
    # overrides these with the empirically optimized values (OVERSOLD=-1.0,
    # OVERBOUGHT=2.5, CHOP_BASELINE=0.5). Class-level defaults are only used
    # if best_params.json doesn't exist (first run / fresh install).
    DEEP_OVERSOLD = -1.5
    OVERSOLD = -1.0
    OVERBOUGHT = 2.5

    # Exit thresholds
    STOP_LOSS_PCT = 0.05
    TRAIL_STOP_PCT = 0.02
    TRAIL_ACTIVATE_PCT = 0.03
    TIME_STOP_DAYS = 60

    # Position sizing baselines per regime
    BULL_BASELINE = 0.50
    CHOP_BASELINE = 0.50
    BEAR_BASELINE = 0.0

    # Lookback for z-score normalization
    ZSCORE_LOOKBACK = 63

    # Path to best_params.json (relative to this file)
    _PARAMS_FILE = Path(__file__).parent / "state" / "best_params.json"

    def __init__(self, period: str = "2y"):
        """
        Args:
            period: yfinance period for data fetch. Use "2y" for live signals
                    (enough history for z-score normalization but fast to fetch).

        Auto-loads state/best_params.json so every engine instance — whether
        used for live scanning, Discord commands, or backtests — uses the
        empirically optimized parameters rather than stale class defaults.
        """
        self.period = period
        self._load_best_params()

    def _load_best_params(self):
        """Apply optimized params from best_params.json as instance attributes."""
        if not self._PARAMS_FILE.exists():
            return
        try:
            with open(self._PARAMS_FILE) as f:
                params = json.load(f)
            for key, val in params.items():
                if hasattr(type(self), key):
                    setattr(self, key, val)
        except Exception:
            pass  # Silently fall back to class defaults on any error

    def compute(self, ticker: str = "SPY") -> PillarSnapshot:
        """Compute current Four Pillars snapshot for a ticker."""
        df = fetch_data(ticker, self.period)
        if len(df) < 200:
            raise ValueError(f"Insufficient data for {ticker}: {len(df)} bars")

        ts = df.index[-1]

        # --- Adaptive Volatility Baseline ---
        daily_returns = df["Close"].pct_change()
        realized_vol_21 = daily_returns.rolling(21).std() * np.sqrt(252)
        vol_median_252 = realized_vol_21.rolling(252).median()
        vol_ratio_val = float(realized_vol_21.iloc[-1] / vol_median_252.iloc[-1]) if (
            not np.isnan(vol_median_252.iloc[-1]) and vol_median_252.iloc[-1] > 0
        ) else 1.0

        # Adaptive baselines: low-vol → increase exposure; high-vol → decrease
        adaptive_bull = max(self.BULL_BASELINE, min(0.75,
            self.BULL_BASELINE + 0.25 * max(0, 1 - vol_ratio_val)))
        adaptive_chop = max(self.CHOP_BASELINE * 0.8, min(self.CHOP_BASELINE,
            self.CHOP_BASELINE * (1 - 0.2 * (vol_ratio_val - 1)))) if vol_ratio_val > 1 else self.CHOP_BASELINE

        # --- Pillar 1: Regime (trend_score) ---
        ts_fn = INDICATOR_REGISTRY["trend_score"]["fn"]
        ts_df = ts_fn(df, len1=13, len2=21, len3=34, len4=55)
        trend_raw = float(ts_df["trend_score"].iloc[-1])
        if trend_raw >= self.BULL_THRESHOLD:
            regime = "bull"
        elif trend_raw <= self.BEAR_THRESHOLD:
            regime = "bear"
        else:
            regime = "chop"

        # --- Pillar 2: Timing (z_hybrid) ---
        zh_fn = INDICATOR_REGISTRY["z_hybrid"]["fn"]
        zh_df = zh_fn(df, fast_len=21, slow_len=34)
        z_hybrid_raw = float(zh_df["z_hybrid"].iloc[-1])
        z_hybrid_norm = _normalize_signal(zh_df["z_hybrid"], self.ZSCORE_LOOKBACK)
        z_hybrid_z = float(z_hybrid_norm.iloc[-1]) if not np.isnan(z_hybrid_norm.iloc[-1]) else 0.0

        if z_hybrid_z <= self.DEEP_OVERSOLD:
            timing = "deep_oversold"
        elif z_hybrid_z <= self.OVERSOLD:
            timing = "oversold"
        elif z_hybrid_z >= self.OVERBOUGHT:
            timing = "overbought"
        else:
            timing = "neutral"

        # --- Pillar 3: Momentum (hybrid_osc) ---
        ho_fn = INDICATOR_REGISTRY["hybrid_osc"]["fn"]
        ho_df = ho_fn(df, length1=34, length2=55, ma_len=8, scale=2.7)
        hybrid_raw = float(ho_df["hybrid_osc"].iloc[-1])

        # Signal line = EMA(8) of hybrid_osc
        ho_series = ho_df["hybrid_osc"]
        ho_signal = ho_series.ewm(span=8, adjust=False).mean()
        ho_signal_val = float(ho_signal.iloc[-1])

        # 3-bar slope
        if len(ho_series) >= 4:
            ho_slope = float(ho_series.iloc[-1] - ho_series.iloc[-4]) / 3
        else:
            ho_slope = 0.0

        # Confirming = below signal line BUT turning up (momentum shifting)
        momentum_confirming = (hybrid_raw < ho_signal_val) and (ho_slope > 0)

        # --- Laggard Score (cross-sectional) ---
        mmrsi_fn = INDICATOR_REGISTRY["multimac_rsi"]["fn"]
        mmrsi_df = mmrsi_fn(df)
        multimac_rsi_score = float(mmrsi_df["multimac_rsi"].iloc[-1]) if not np.isnan(mmrsi_df["multimac_rsi"].iloc[-1]) else 0.0

        # Cross-sectional confirmation: deeply negative = washed out = higher conviction
        cross_sectional_confirming = multimac_rsi_score < -3.0

        # --- Pillar 4: Volume (ve_rsi) ---
        vr_fn = INDICATOR_REGISTRY["ve_rsi"]["fn"]
        vr_df = vr_fn(df, length=14)
        ve_rsi_raw = float(vr_df["ve_rsi"].iloc[-1])
        ve_rsi_norm = _normalize_signal(vr_df["ve_rsi"], self.ZSCORE_LOOKBACK)
        ve_rsi_z = float(ve_rsi_norm.iloc[-1]) if not np.isnan(ve_rsi_norm.iloc[-1]) else 0.0

        # Volume ratio
        vol = df["Volume"]
        vol_avg = vol.rolling(65).mean()
        volume_ratio = float(vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

        # Confirming = ve_rsi oversold (<35) OR ve_rsi divergence
        volume_confirming = ve_rsi_raw < 35 or (ve_rsi_z < -1.5 and volume_ratio > 1.2)

        # --- VIX z-score enhancement ---
        try:
            vix_df = fetch_data("^VIX", self.period)
            vix_close = vix_df["Close"]
            vix_current = float(vix_close.iloc[-1])
            vix_mean = float(vix_close.rolling(63).mean().iloc[-1])
            vix_std = float(vix_close.rolling(63).std().iloc[-1])
            vix_z = (vix_current - vix_mean) / vix_std if vix_std > 0 else 0.0
        except Exception:
            vix_z = 0.0

        # --- RRF (Retracement/Reversal Factor) ---
        try:
            rrf_fn = INDICATOR_REGISTRY["rrf"]["fn"]
            rrf_df = rrf_fn(df)
            rrf_val = float(rrf_df["rrf"].iloc[-1]) if not np.isnan(rrf_df["rrf"].iloc[-1]) else 12.5
        except Exception:
            rrf_val = 12.5

        # --- Position sizing via pillar matrix ---
        # Now 5 confirmations: momentum + volume + cross-sectional + timing + regime
        confirmations = sum([momentum_confirming, volume_confirming, cross_sectional_confirming])
        pillars_confirming = confirmations + (1 if timing in ("oversold", "deep_oversold") else 0) + (1 if regime == "bull" else 0)

        if regime == "bull":
            if timing == "deep_oversold" and confirmations >= 1:
                position_pct = 1.0
                label = "STRONG_BUY"
            elif timing == "oversold" and confirmations >= 1:
                position_pct = 1.0
                label = "BUY"
            elif timing in ("deep_oversold", "oversold"):
                position_pct = 0.75
                label = "BUY"
            elif timing == "overbought":
                position_pct = max(0.0, adaptive_bull - 0.25)
                label = "REDUCE"
            else:
                position_pct = adaptive_bull
                label = "HOLD"
        elif regime == "chop":
            if timing == "deep_oversold" and confirmations >= 1:
                position_pct = 0.75
                label = "BUY"
            elif timing == "oversold":
                position_pct = adaptive_chop
                label = "BUY"
            elif timing == "overbought":
                position_pct = 0.0
                label = "FLAT"
            else:
                position_pct = max(0.0, adaptive_chop - 0.25)
                label = "HOLD"
        else:  # bear
            if timing == "deep_oversold" and confirmations == 2:
                position_pct = 0.50
                label = "BUY"
            elif timing == "deep_oversold" and confirmations >= 1:
                position_pct = 0.25
                label = "BUY"
            else:
                position_pct = 0.0
                label = "FLAT"

        # --- VIX fear spike boost ---
        if vix_z > 2.0 and timing in ("oversold", "deep_oversold"):
            position_pct = min(1.0, position_pct + 0.25)
            if label not in ("STRONG_BUY",):
                label = "STRONG_BUY"

        # --- RRF pre-filter: low RRF = smooth trend, cap timing-driven additions ---
        if rrf_val < 8.0 and regime == "bull" and position_pct > adaptive_bull:
            position_pct = min(position_pct, adaptive_bull + 0.25)

        # --- Momentum-regime override: strong low-vol uptrend → boost baseline ---
        momentum_50d = float(df["Close"].iloc[-1] / df["Close"].iloc[-50] - 1) if len(df) >= 50 else 0
        if regime == "bull" and timing == "neutral" and vol_ratio_val < 1.0 and momentum_50d > 0.15:
            position_pct = max(position_pct, 0.75)
            label = "TREND_HOLD"

        confidence = min(1.0, pillars_confirming / 5.0)

        return PillarSnapshot(
            timestamp=ts,
            ticker=ticker,
            trend_score_raw=trend_raw,
            regime=regime,
            z_hybrid_raw=z_hybrid_raw,
            z_hybrid_zscore=z_hybrid_z,
            timing_signal=timing,
            hybrid_osc_raw=hybrid_raw,
            hybrid_osc_signal=ho_signal_val,
            hybrid_osc_slope=ho_slope,
            momentum_confirming=momentum_confirming,
            ve_rsi_raw=ve_rsi_raw,
            ve_rsi_zscore=ve_rsi_z,
            volume_ratio=volume_ratio,
            volume_confirming=volume_confirming,
            position_pct=position_pct,
            signal_label=label,
            confidence=confidence,
            pillars_confirming=pillars_confirming,
            multimac_rsi_score=multimac_rsi_score,
            cross_sectional_confirming=cross_sectional_confirming,
            vix_zscore=vix_z,
            rrf_value=rrf_val,
            vol_ratio=vol_ratio_val,
            adaptive_baseline=adaptive_bull if regime == "bull" else adaptive_chop,
        )

    def generate_signal(
        self,
        snapshot: PillarSnapshot,
        current_position: float,
        entry_price: Optional[float] = None,
        entry_date: Optional[pd.Timestamp] = None,
        high_water_mark: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> TradeSignal:
        """
        Generate actionable trade signal given current state and position.
        """
        if current_price is None:
            df = fetch_data(snapshot.ticker, "5d")
            current_price = float(df["Close"].iloc[-1])

        # --- Regime baselines for reduce-to-baseline exits ---
        _baselines = {"bull": self.BULL_BASELINE, "chop": self.CHOP_BASELINE, "bear": 0.0}
        regime_baseline = _baselines.get(snapshot.regime, 0.0)

        # --- Check exits first (if we have a position) ---
        if current_position > 0 and entry_price is not None:
            pnl_pct = (current_price - entry_price) / entry_price
            days_held = (snapshot.timestamp - entry_date).days if entry_date else 0

            # Stop loss → reduce to baseline (not full exit in bull/chop)
            if pnl_pct <= -self.STOP_LOSS_PCT:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL" if regime_baseline == 0 else "REDUCE",
                    position_pct=regime_baseline,
                    entry_price=entry_price, stop_price=entry_price * (1 - self.STOP_LOSS_PCT),
                    trail_price=None,
                    reason=f"STOP LOSS hit: {pnl_pct:+.1%} (limit: -{self.STOP_LOSS_PCT:.0%}), reducing to {regime_baseline:.0%} baseline",
                    snapshot=snapshot,
                )

            # Adaptive trailing stop: wider trail for big winners
            if high_water_mark and pnl_pct > self.TRAIL_ACTIVATE_PCT:
                effective_trail = self.TRAIL_STOP_PCT
                if pnl_pct > 0.08:
                    effective_trail = self.TRAIL_STOP_PCT * 1.5   # wider trail for big winners
                elif pnl_pct > 0.05:
                    effective_trail = self.TRAIL_STOP_PCT * 1.25
                trail_level = high_water_mark * (1 - effective_trail)
                if current_price < trail_level:
                    return TradeSignal(
                        timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                        action="SELL" if regime_baseline == 0 else "REDUCE",
                        position_pct=regime_baseline,
                        entry_price=entry_price, stop_price=None,
                        trail_price=trail_level,
                        reason=f"TRAIL STOP: price {current_price:.2f} < trail {trail_level:.2f} (HWM {high_water_mark:.2f}, trail={effective_trail:.1%})",
                        snapshot=snapshot,
                    )

            # Profit decay timer: held 30+ days with no movement → reduce to baseline
            if entry_date and days_held >= 30 and abs(pnl_pct) < 0.01:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL" if regime_baseline == 0 else "REDUCE",
                    position_pct=regime_baseline,
                    entry_price=entry_price, stop_price=None, trail_price=None,
                    reason=f"PROFIT DECAY: held {days_held} days with only {pnl_pct:+.1%} — dead money, reducing to {regime_baseline:.0%} baseline",
                    snapshot=snapshot,
                )

            # Hard time stop as absolute backstop → reduce to baseline
            if entry_date and days_held >= self.TIME_STOP_DAYS:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL" if regime_baseline == 0 else "REDUCE",
                    position_pct=regime_baseline,
                    entry_price=entry_price, stop_price=None, trail_price=None,
                    reason=f"TIME STOP: held {days_held} days (limit: {self.TIME_STOP_DAYS}), reducing to {regime_baseline:.0%} baseline",
                    snapshot=snapshot,
                )

            # Primary exit: z_hybrid crosses above 0 → reduce to baseline (NOT full exit)
            if snapshot.z_hybrid_zscore > 0 and snapshot.hybrid_osc_slope <= 0:
                target = regime_baseline
                if target < current_position:
                    return TradeSignal(
                        timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                        action="REDUCE" if target > 0 else "SELL",
                        position_pct=target,
                        entry_price=entry_price, stop_price=None, trail_price=None,
                        reason=f"SIGNAL EXIT: z_hybrid z-score crossed above 0 ({snapshot.z_hybrid_zscore:+.2f}), reducing to {target:.0%} baseline",
                        snapshot=snapshot,
                    )

        # --- Check entries ---
        target = snapshot.position_pct

        if target > current_position:
            stop = current_price * (1 - self.STOP_LOSS_PCT)
            return TradeSignal(
                timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                action="BUY", position_pct=target,
                entry_price=current_price, stop_price=stop, trail_price=None,
                reason=f"{snapshot.signal_label}: regime={snapshot.regime}, timing={snapshot.timing_signal}, "
                       f"momentum={'YES' if snapshot.momentum_confirming else 'NO'}, "
                       f"volume={'YES' if snapshot.volume_confirming else 'NO'} "
                       f"({snapshot.pillars_confirming}/4 pillars)",
                snapshot=snapshot,
            )

        if target < current_position and current_position > 0:
            return TradeSignal(
                timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                action="REDUCE", position_pct=target,
                entry_price=entry_price, stop_price=None, trail_price=None,
                reason=f"Reducing position: {current_position:.0%} → {target:.0%} ({snapshot.signal_label})",
                snapshot=snapshot,
            )

        return TradeSignal(
            timestamp=snapshot.timestamp, ticker=snapshot.ticker,
            action="HOLD", position_pct=current_position,
            entry_price=entry_price, stop_price=None, trail_price=None,
            reason=f"No action: {snapshot.signal_label} ({snapshot.pillars_confirming}/4 pillars)",
            snapshot=snapshot,
        )

    @staticmethod
    def rank_snapshots(snapshots: list) -> list:
        """
        Rank a list of PillarSnapshots cross-sectionally by multimac_rsi_score.

        Based on J. Kornblatt's 2007 study "Leaders vs Laggards": among a universe
        of tickers, the most extreme laggards (lowest multimac_rsi) outperformed
        the leaders in 10/12 periods with 0.03 correlation to market returns.

        Assigns .laggard_rank: 1 = most washed-out (highest buy priority),
        N = most overbought/trending (lowest priority).

        Args:
            snapshots: list of PillarSnapshot objects

        Returns:
            Same list with .laggard_rank populated, sorted by rank (1 = best laggard).
        """
        if not snapshots:
            return snapshots
        # Sort ascending by multimac_rsi (most negative = rank 1)
        ranked = sorted(snapshots, key=lambda s: s.multimac_rsi_score)
        for i, snap in enumerate(ranked):
            snap.laggard_rank = i + 1
        return ranked

    def compute_historical(
        self,
        ticker: str = "SPY",
        period: str = "10y",
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """
        Compute historical pillar readings for backtesting.
        Returns a DataFrame with pillar values, positions, and signals per day.

        Args:
            start: ISO date "YYYY-MM-DD". If provided, overrides period.
            end:   ISO date "YYYY-MM-DD". Optional upper bound.
        """
        df = fetch_data(ticker, period, start=start, end=end)
        if len(df) < 200:
            raise ValueError(f"Insufficient data for {ticker}: {len(df)} bars")

        # Compute all indicators
        ts_df = INDICATOR_REGISTRY["trend_score"]["fn"](df, len1=13, len2=21, len3=34, len4=55)
        zh_df = INDICATOR_REGISTRY["z_hybrid"]["fn"](df, fast_len=21, slow_len=34)
        ho_df = INDICATOR_REGISTRY["hybrid_osc"]["fn"](df, length1=34, length2=55, ma_len=8, scale=2.7)
        vr_df = INDICATOR_REGISTRY["ve_rsi"]["fn"](df, length=14)

        result = pd.DataFrame(index=df.index)
        result["close"] = df["Close"]
        result["volume"] = df["Volume"]

        # --- Adaptive Volatility Baseline (vectorized) ---
        daily_ret = df["Close"].pct_change()
        realized_vol_21 = daily_ret.rolling(21).std() * np.sqrt(252)
        vol_median_252 = realized_vol_21.rolling(252).median()
        vol_ratio_series = (realized_vol_21 / vol_median_252).fillna(1.0).replace([np.inf, -np.inf], 1.0)
        result["vol_ratio_adaptive"] = vol_ratio_series

        adaptive_bull_series = (self.BULL_BASELINE + 0.25 * (1 - vol_ratio_series).clip(lower=0)).clip(
            lower=self.BULL_BASELINE, upper=0.75)
        adaptive_chop_series = pd.Series(self.CHOP_BASELINE, index=df.index)
        high_vol = vol_ratio_series > 1
        adaptive_chop_series[high_vol] = (
            self.CHOP_BASELINE * (1 - 0.2 * (vol_ratio_series[high_vol] - 1))
        ).clip(lower=self.CHOP_BASELINE * 0.8, upper=self.CHOP_BASELINE)

        # Pillar 1
        result["trend_score"] = ts_df["trend_score"]
        result["regime"] = "chop"
        result.loc[result["trend_score"] >= self.BULL_THRESHOLD, "regime"] = "bull"
        result.loc[result["trend_score"] <= self.BEAR_THRESHOLD, "regime"] = "bear"

        # Pillar 2
        result["z_hybrid"] = zh_df["z_hybrid"]
        result["z_hybrid_z"] = _normalize_signal(zh_df["z_hybrid"], self.ZSCORE_LOOKBACK)

        # Pillar 3
        result["hybrid_osc"] = ho_df["hybrid_osc"]
        ho_signal = ho_df["hybrid_osc"].ewm(span=8, adjust=False).mean()
        result["hybrid_osc_signal"] = ho_signal
        result["hybrid_osc_slope"] = ho_df["hybrid_osc"].diff(3) / 3
        result["momentum_confirming"] = (result["hybrid_osc"] < result["hybrid_osc_signal"]) & (result["hybrid_osc_slope"] > 0)

        # Pillar 4
        result["ve_rsi"] = vr_df["ve_rsi"]
        result["ve_rsi_z"] = _normalize_signal(vr_df["ve_rsi"], self.ZSCORE_LOOKBACK)
        vol_avg = df["Volume"].rolling(65).mean()
        result["vol_ratio"] = df["Volume"] / vol_avg
        result["volume_confirming"] = (result["ve_rsi"] < 35) | ((result["ve_rsi_z"] < -1.5) & (result["vol_ratio"] > 1.2))

        # --- Cross-sectional confirmation (multimac_rsi) ---
        mmrsi_df = INDICATOR_REGISTRY["multimac_rsi"]["fn"](df)
        result["multimac_rsi"] = mmrsi_df["multimac_rsi"]
        result["cross_sectional_confirming"] = result["multimac_rsi"] < -3.0

        # --- VIX z-score (vectorized) ---
        try:
            vix_df = fetch_data("^VIX", period, start=start, end=end)
            vix_close = vix_df["Close"].reindex(df.index, method="ffill")
            vix_mean = vix_close.rolling(63).mean()
            vix_std = vix_close.rolling(63).std()
            result["vix_z"] = ((vix_close - vix_mean) / vix_std).fillna(0.0)
        except Exception:
            result["vix_z"] = 0.0

        # --- RRF (vectorized) ---
        try:
            rrf_fn = INDICATOR_REGISTRY["rrf"]["fn"]
            rrf_df = rrf_fn(df)
            result["rrf"] = rrf_df["rrf"].fillna(12.5)
        except Exception:
            result["rrf"] = 12.5

        # Timing signal
        result["timing"] = "neutral"
        result.loc[result["z_hybrid_z"] <= self.DEEP_OVERSOLD, "timing"] = "deep_oversold"
        result.loc[(result["z_hybrid_z"] > self.DEEP_OVERSOLD) & (result["z_hybrid_z"] <= self.OVERSOLD), "timing"] = "oversold"
        result.loc[result["z_hybrid_z"] >= self.OVERBOUGHT, "timing"] = "overbought"

        # Confirmations count (now includes cross-sectional)
        result["confirmations"] = (
            result["momentum_confirming"].astype(int)
            + result["volume_confirming"].astype(int)
            + result["cross_sectional_confirming"].astype(int)
        )

        # Position sizing
        result["position"] = 0.0

        bull = result["regime"] == "bull"
        chop = result["regime"] == "chop"
        bear = result["regime"] == "bear"
        deep_os = result["timing"] == "deep_oversold"
        os_any = result["timing"].isin(["oversold", "deep_oversold"])
        ob = result["timing"] == "overbought"

        # Bull regime: adaptive baseline, ±0.25 on signals
        result.loc[bull, "position"] = adaptive_bull_series[bull]              # adaptive baseline
        result.loc[bull & ob, "position"] = (adaptive_bull_series[bull & ob] - 0.25).clip(lower=0.0)
        result.loc[bull & os_any, "position"] = (adaptive_bull_series[bull & os_any] + 0.25).clip(upper=1.0)
        result.loc[bull & os_any & (result["confirmations"] >= 1), "position"] = 1.0

        # Chop regime: adaptive baseline, ±0.25 on signals
        result.loc[chop, "position"] = adaptive_chop_series[chop]             # adaptive baseline
        result.loc[chop & ob, "position"] = (adaptive_chop_series[chop & ob] - 0.25).clip(lower=0.0)
        result.loc[chop & os_any, "position"] = (adaptive_chop_series[chop & os_any] + 0.25).clip(upper=1.0)
        result.loc[chop & deep_os & (result["confirmations"] >= 1), "position"] = (
            adaptive_chop_series[chop & deep_os & (result["confirmations"] >= 1)] + 0.50
        ).clip(upper=1.0)

        # Bear regime: parameterized baseline, small bounces only
        _bear_deep1 = min(0.50, self.BEAR_BASELINE + 0.25)
        _bear_deep2 = min(0.75, self.BEAR_BASELINE + 0.50)
        result.loc[bear, "position"] = self.BEAR_BASELINE
        result.loc[bear & deep_os & (result["confirmations"] >= 1), "position"] = _bear_deep1
        result.loc[bear & deep_os & (result["confirmations"] == 2), "position"] = _bear_deep2

        # --- VIX fear spike boost ---
        vix_spike = result["vix_z"] > 2.0
        os_timing = result["timing"].isin(["oversold", "deep_oversold"])
        result.loc[vix_spike & os_timing, "position"] = (
            result.loc[vix_spike & os_timing, "position"] + 0.25
        ).clip(upper=1.0)

        # --- RRF pre-filter: low RRF caps timing-driven additions in bull ---
        low_rrf = result["rrf"] < 8.0
        above_baseline = result["position"] > adaptive_bull_series
        result.loc[bull & low_rrf & above_baseline, "position"] = (
            adaptive_bull_series[bull & low_rrf & above_baseline] + 0.25
        ).clip(upper=result.loc[bull & low_rrf & above_baseline, "position"])

        # --- Momentum-regime override: strong low-vol uptrend → boost ---
        momentum_50d = df["Close"].pct_change(50)
        strong_trend = (momentum_50d > 0.15) & (vol_ratio_series < 1.0)
        neutral_timing = result["timing"] == "neutral"
        result.loc[bull & neutral_timing & strong_trend, "position"] = np.maximum(
            result.loc[bull & neutral_timing & strong_trend, "position"], 0.75
        )

        return result
