"""
Four Pillars Signal Engine
===========================
Computes the four orthogonal JK pillars and generates position signals.

Pillar 1: REGIME     — trend_score (-5 to +5) → bull / chop / bear
Pillar 2: TIMING     — z_hybrid z-score → oversold / neutral / overbought
Pillar 3: MOMENTUM   — hybrid_osc vs signal line → confirming / not
Pillar 4: VOLUME     — ve_rsi level + volume ratio → confirming / not
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
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
    DEEP_OVERSOLD = -1.5
    OVERSOLD = -0.8
    OVERBOUGHT = 1.5

    # Exit thresholds
    STOP_LOSS_PCT = 0.05
    TRAIL_STOP_PCT = 0.02
    TRAIL_ACTIVATE_PCT = 0.03
    TIME_STOP_DAYS = 60

    # Position sizing baselines per regime
    BULL_BASELINE = 0.50
    CHOP_BASELINE = 0.25
    BEAR_BASELINE = 0.0

    # Lookback for z-score normalization
    ZSCORE_LOOKBACK = 63

    def __init__(self, period: str = "2y"):
        """
        Args:
            period: yfinance period for data fetch. Use "2y" for live signals
                    (enough history for z-score normalization but fast to fetch).
        """
        self.period = period

    def compute(self, ticker: str = "SPY") -> PillarSnapshot:
        """Compute current Four Pillars snapshot for a ticker."""
        df = fetch_data(ticker, self.period)
        if len(df) < 200:
            raise ValueError(f"Insufficient data for {ticker}: {len(df)} bars")

        ts = df.index[-1]

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

        # --- Pillar 4: Volume (ve_rsi) ---
        vr_fn = INDICATOR_REGISTRY["ve_rsi"]["fn"]
        vr_df = vr_fn(df, length=14)
        ve_rsi_raw = float(vr_df["ve_rsi"].iloc[-1])
        ve_rsi_norm = _normalize_signal(vr_df["ve_rsi"], self.ZSCORE_LOOKBACK)
        ve_rsi_z = float(ve_rsi_norm.iloc[-1]) if not np.isnan(ve_rsi_norm.iloc[-1]) else 0.0

        # Volume ratio
        vol = df["Volume"]
        vol_avg = vol.rolling(65).mean()
        vol_ratio = float(vol.iloc[-1] / vol_avg.iloc[-1]) if vol_avg.iloc[-1] > 0 else 1.0

        # Confirming = ve_rsi oversold (<35) OR ve_rsi divergence
        volume_confirming = ve_rsi_raw < 35 or (ve_rsi_z < -1.5 and vol_ratio > 1.2)

        # --- Position sizing via pillar matrix ---
        confirmations = sum([momentum_confirming, volume_confirming])
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
                position_pct = 0.25
                label = "REDUCE"
            else:
                # Bull baseline: stay 50% invested to capture uptrend
                position_pct = 0.50
                label = "HOLD"
        elif regime == "chop":
            if timing == "deep_oversold" and confirmations >= 1:
                position_pct = 0.75
                label = "BUY"
            elif timing == "oversold":
                position_pct = 0.50
                label = "BUY"
            elif timing == "overbought":
                position_pct = 0.0
                label = "FLAT"
            else:
                position_pct = 0.25
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

        confidence = min(1.0, pillars_confirming / 4.0)

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
            volume_ratio=vol_ratio,
            volume_confirming=volume_confirming,
            position_pct=position_pct,
            signal_label=label,
            confidence=confidence,
            pillars_confirming=pillars_confirming,
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

        # --- Check exits first (if we have a position) ---
        if current_position > 0 and entry_price is not None:
            pnl_pct = (current_price - entry_price) / entry_price

            # Stop loss
            if pnl_pct <= -self.STOP_LOSS_PCT:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL", position_pct=0.0,
                    entry_price=entry_price, stop_price=entry_price * (1 - self.STOP_LOSS_PCT),
                    trail_price=None,
                    reason=f"STOP LOSS hit: {pnl_pct:+.1%} (limit: -{self.STOP_LOSS_PCT:.0%})",
                    snapshot=snapshot,
                )

            # Trailing stop (activate after +3%, trail at -2% from HWM)
            if high_water_mark and pnl_pct > self.TRAIL_ACTIVATE_PCT:
                trail_level = high_water_mark * (1 - self.TRAIL_STOP_PCT)
                if current_price < trail_level:
                    return TradeSignal(
                        timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                        action="SELL", position_pct=0.0,
                        entry_price=entry_price, stop_price=None,
                        trail_price=trail_level,
                        reason=f"TRAIL STOP: price {current_price:.2f} < trail {trail_level:.2f} (HWM {high_water_mark:.2f})",
                        snapshot=snapshot,
                    )

            # Time stop
            if entry_date and (snapshot.timestamp - entry_date).days >= self.TIME_STOP_DAYS:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL", position_pct=0.0,
                    entry_price=entry_price, stop_price=None, trail_price=None,
                    reason=f"TIME STOP: held {(snapshot.timestamp - entry_date).days} days (limit: {self.TIME_STOP_DAYS})",
                    snapshot=snapshot,
                )

            # Primary exit: z_hybrid crosses above 0 (mean-reversion complete)
            if snapshot.z_hybrid_zscore > 0 and snapshot.hybrid_osc_slope <= 0:
                return TradeSignal(
                    timestamp=snapshot.timestamp, ticker=snapshot.ticker,
                    action="SELL", position_pct=0.0,
                    entry_price=entry_price, stop_price=None, trail_price=None,
                    reason=f"SIGNAL EXIT: z_hybrid z-score crossed above 0 ({snapshot.z_hybrid_zscore:+.2f}), momentum fading",
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

    def compute_historical(self, ticker: str = "SPY", period: str = "10y") -> pd.DataFrame:
        """
        Compute historical pillar readings for backtesting.
        Returns a DataFrame with pillar values, positions, and signals per day.
        """
        df = fetch_data(ticker, period)
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

        # Timing signal
        result["timing"] = "neutral"
        result.loc[result["z_hybrid_z"] <= self.DEEP_OVERSOLD, "timing"] = "deep_oversold"
        result.loc[(result["z_hybrid_z"] > self.DEEP_OVERSOLD) & (result["z_hybrid_z"] <= self.OVERSOLD), "timing"] = "oversold"
        result.loc[result["z_hybrid_z"] >= self.OVERBOUGHT, "timing"] = "overbought"

        # Confirmations count
        result["confirmations"] = result["momentum_confirming"].astype(int) + result["volume_confirming"].astype(int)

        # Position sizing
        result["position"] = 0.0

        bull = result["regime"] == "bull"
        chop = result["regime"] == "chop"
        bear = result["regime"] == "bear"
        deep_os = result["timing"] == "deep_oversold"
        os_any = result["timing"].isin(["oversold", "deep_oversold"])
        ob = result["timing"] == "overbought"

        # Bull regime: parameterized baseline, ±0.25 on signals
        _bull_ob = max(0.0, self.BULL_BASELINE - 0.25)
        _bull_os = min(1.0, self.BULL_BASELINE + 0.25)
        result.loc[bull, "position"] = self.BULL_BASELINE             # baseline
        result.loc[bull & ob, "position"] = _bull_ob                  # reduce on overbought
        result.loc[bull & os_any, "position"] = _bull_os              # oversold
        result.loc[bull & os_any & (result["confirmations"] >= 1), "position"] = 1.0  # confirmed oversold

        # Chop regime: parameterized baseline, ±0.25 on signals
        _chop_ob = max(0.0, self.CHOP_BASELINE - 0.25)
        _chop_os = min(1.0, self.CHOP_BASELINE + 0.25)
        _chop_deep = min(1.0, self.CHOP_BASELINE + 0.50)
        result.loc[chop, "position"] = self.CHOP_BASELINE             # baseline
        result.loc[chop & ob, "position"] = _chop_ob                  # flat/reduce on overbought
        result.loc[chop & os_any, "position"] = _chop_os              # oversold
        result.loc[chop & deep_os & (result["confirmations"] >= 1), "position"] = _chop_deep

        # Bear regime: parameterized baseline, small bounces only
        _bear_deep1 = min(0.50, self.BEAR_BASELINE + 0.25)
        _bear_deep2 = min(0.75, self.BEAR_BASELINE + 0.50)
        result.loc[bear, "position"] = self.BEAR_BASELINE
        result.loc[bear & deep_os & (result["confirmations"] >= 1), "position"] = _bear_deep1
        result.loc[bear & deep_os & (result["confirmations"] == 2), "position"] = _bear_deep2

        return result
