"""
Jonathan Kornblatt's Custom Technical Indicators
=================================================
Translated from TradeStation EasyLanguage to Python.
Original code circa 2014, used as part of a larger trading website.

Each function takes a DataFrame with OHLCV columns and returns indicator value(s).
"""

import numpy as np
import pandas as pd
from .core import ema, sma, rsi, true_range, highest, lowest


# ---------------------------------------------------------------------------
# 1. JK MultiMAC (Multi-Moving-Average Convergence)
# ---------------------------------------------------------------------------
def jk_multimac(df: pd.DataFrame,
                ma_len_a: int = 7, ma_len_b: int = 11,
                ma_len_1: int = 17, ma_len_2: int = 27,
                ma_len_3: int = 44, ma_len_4: int = 72) -> pd.DataFrame:
    """
    MultiMAC — cascading EMA differences normalized by price.
    Core concept: sum of (EMA_fast - EMA_slow) pairs across multiple timeframes.
    When all EMAs are aligned (bullish), value is high positive.

    Original: {perfTestMMac cc} and {websitejkMultimac}
    """
    c = df["Close"]
    avg_a = ema(c, ma_len_a)
    avg_b = ema(c, ma_len_b)
    avg_1 = ema(c, ma_len_1)
    avg_2 = ema(c, ma_len_2)
    avg_3 = ema(c, ma_len_3)
    avg_4 = ema(c, ma_len_4)

    avgdiff1 = avg_a - avg_b
    avgdiff2 = avg_b - avg_1
    avgdiff3 = avg_1 - avg_2
    avgdiff4 = avg_2 - avg_3
    avgdiff5 = avg_3 - avg_4

    px_avg = (avg_1 + avg_2 + avg_3 + avg_4) / 4
    multimac = 100 * (avgdiff1 + avgdiff2 + avgdiff3 + avgdiff4 + avgdiff5) / px_avg

    return pd.DataFrame({
        "multimac": multimac,
        "multimac_chg_1d": multimac - multimac.shift(1),
        "multimac_chg_5d": multimac - multimac.shift(5),
        "multimac_chg_20d": multimac - multimac.shift(20),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 2. JK MultiMAC Fibonacci Version
# ---------------------------------------------------------------------------
def jk_multimac_fib(df: pd.DataFrame,
                    ma_len_a: int = 8, ma_len_b: int = 13,
                    ma_len_1: int = 21, ma_len_2: int = 34,
                    ma_len_3: int = 55, ma_len_4: int = 89) -> pd.DataFrame:
    """
    MultiMAC with Fibonacci-based MA lengths.
    Uses percentage-based normalization (each diff / longer MA).

    Original: {webzz1multimacFibVer} and {MMACtest10a}
    """
    c = df["Close"]
    avg_a = ema(c, ma_len_a)
    avg_b = ema(c, ma_len_b)
    avg_1 = ema(c, ma_len_1)
    avg_2 = ema(c, ma_len_2)
    avg_3 = ema(c, ma_len_3)
    avg_4 = ema(c, ma_len_4)

    # Each diff normalized by the slower MA of the pair
    multimac = 100 * (
        (avg_a - avg_b) / avg_b +
        (avg_b - avg_1) / avg_1 +
        (avg_1 - avg_2) / avg_2 +
        (avg_2 - avg_3) / avg_3 +
        (avg_3 - avg_4) / avg_4
    )

    return pd.DataFrame({
        "multimac_fib": multimac,
        "multimac_fib_chg_1d": multimac - multimac.shift(1),
        "multimac_fib_chg_5d": multimac - multimac.shift(5),
        "multimac_fib_chg_20d": multimac - multimac.shift(20),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 3. JK MultiMAC r7duo (with RSI component)
# ---------------------------------------------------------------------------
def jk_multimac_rsi(df: pd.DataFrame,
                    ma_len_a: int = 7, ma_len_b: int = 11,
                    ma_len_1: int = 17, ma_len_2: int = 27,
                    ma_len_3: int = 44, ma_len_4: int = 72,
                    rsi_len: int = 5) -> pd.DataFrame:
    """
    MultiMAC with adaptive RSI component.
    RSI part is scaled asymmetrically: /5.5 when bullish+oversold, /7 otherwise.
    This creates a mean-reversion overlay on the trend signal.

    Original: {MMACr7duo}
    """
    c = df["Close"]
    avg_a = ema(c, ma_len_a)
    avg_b = ema(c, ma_len_b)
    avg_1 = ema(c, ma_len_1)
    avg_2 = ema(c, ma_len_2)
    avg_3 = ema(c, ma_len_3)
    avg_4 = ema(c, ma_len_4)

    avgdiff1 = avg_a - avg_b
    avgdiff2 = avg_b - avg_1
    avgdiff3 = avg_1 - avg_2
    avgdiff4 = avg_2 - avg_3
    avgdiff5 = avg_3 - avg_4

    tot_avgdiffs = avgdiff1 + avgdiff2 + avgdiff3 + avgdiff4 + avgdiff5
    px_avg = np.where(tot_avgdiffs > 0, avg_a, avg_4)

    rsi_val = rsi(c, rsi_len)
    rsi_part = np.where(
        (tot_avgdiffs > 0) & (rsi_val < 50),
        (rsi_val - 50) / 5.5,
        (rsi_val - 50) / 7
    )

    mmac_rsi = (100 * tot_avgdiffs / px_avg) + rsi_part

    return pd.DataFrame({
        "multimac_rsi": pd.Series(mmac_rsi, index=df.index),
        "rsi_part": pd.Series(rsi_part, index=df.index),
        "trend_only": pd.Series(100 * tot_avgdiffs / px_avg, index=df.index),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 4. JK MultiMAC r9 Dampened (capped longer-term diffs)
# ---------------------------------------------------------------------------
def jk_multimac_dampened(df: pd.DataFrame,
                         ma_len_aaa: int = 4, ma_len_a: int = 7,
                         ma_len_b: int = 11, ma_len_1: int = 17,
                         ma_len_2: int = 27, ma_len_3: int = 44,
                         ma_len_4: int = 72,
                         cap3: float = 0.021, damp3: float = 0.5,
                         cap4: float = 0.027, damp4: float = 0.2,
                         cap5: float = 0.036, damp5: float = 0.2) -> pd.DataFrame:
    """
    Dampened MultiMAC — prevents longer-term diffs from dominating.
    When a diff exceeds a cap threshold, the excess is dampened by a factor.
    This is a sophistication to prevent the indicator from being overwhelmed
    by strong long-term trends (making it more responsive to shorter-term shifts).

    Original: {MMACr9duaADJ} and {MMACr9duoADJbbb}
    """
    c = df["Close"]
    avg_aaa = ema(c, ma_len_aaa)
    avg_a = ema(c, ma_len_a)
    avg_b = ema(c, ma_len_b)
    avg_1 = ema(c, ma_len_1)
    avg_2 = ema(c, ma_len_2)
    avg_3 = ema(c, ma_len_3)
    avg_4 = ema(c, ma_len_4)

    avgdiff_aaa = (avg_aaa - avg_a) / 2
    avgdiff1 = avg_a - avg_b
    avgdiff2 = avg_b - avg_1

    # Dampened diffs (percentage-based)
    pre3 = (avg_1 - avg_2) / avg_2
    pre4 = (avg_2 - avg_3) / avg_3
    pre5 = (avg_3 - avg_4) / avg_4

    def _damp(pre, cap, damp_factor):
        result = pre.copy()
        over = pre > cap
        under = pre < -cap
        result[over] = cap + damp_factor * (pre[over] - cap)
        result[under] = -cap - damp_factor * (pre[under] + cap)
        return result

    avgdiff3 = _damp(pre3, cap3, damp3)
    avgdiff4 = _damp(pre4, cap4, damp4)
    avgdiff5 = _damp(pre5, cap5, damp5)

    tot = avgdiff1 / avg_b + avgdiff2 / avg_1 + avgdiff3 + avgdiff4 + avgdiff5

    rsi_val = rsi(c, 5)
    rsi_part = np.where(
        (tot > 0) & (rsi_val < 50),
        (rsi_val - 50) / 5.5,
        (rsi_val - 50) / 7
    )

    mmac_r9 = (100 * tot) + rsi_part

    return pd.DataFrame({
        "multimac_dampened": pd.Series(mmac_r9, index=df.index),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 5. JK Hybrid Oscillator (RSI Differential Hybrid)
# ---------------------------------------------------------------------------
def jk_hybrid_oscillator(df: pd.DataFrame,
                         length1: int = 34, length2: int = 55,
                         ma_len: int = 8,
                         scale: float = 2.7) -> pd.DataFrame:
    """
    Hybrid RSI oscillator combining:
    - RSI differential (medium vs slow RSI)
    - Smoothed RSI bias term (RSI slow smoothed - 50) / 5
    - Exponential smoothing of the combination

    This captures both momentum divergence (RSI diff) and absolute level (bias).
    The signal line (MA of hybrid) generates crossover signals.
    Bands are adaptive based on recent range of the signal line.

    Original: {rsidiffhybband1}, {webjkHybridOsc}, {webzJKHybridOsc}
    """
    c = df["Close"]
    rsi_medium = rsi(c, length1)
    rsi_slow = rsi(c, length2)
    rsi_diff = rsi_medium - rsi_slow
    rsi_slow_smooth = ema(rsi_medium, 3)

    # The core hybrid indicator
    raw = (rsi_diff + ((rsi_slow_smooth - 50) / 5)) / 2
    hybrid = scale * ema(raw, 2)

    # Signal line
    signal = ema(hybrid, ma_len)

    # Adaptive bands
    up_diff = highest(signal, 18) - lowest(signal, 7)
    up_band_amt = ema(up_diff, 2) / 6.7
    dn_diff = highest(signal, 7) - lowest(signal, 18)
    dn_band_amt = ema(dn_diff, 2) / 6.7
    stnd_diff = highest(signal, 5) - lowest(signal, 5)
    stnd_amt = ema(stnd_diff, 2) / 4

    upper_band = signal + stnd_amt + up_band_amt + 0.21
    lower_band = signal - stnd_amt - dn_band_amt - 0.21

    return pd.DataFrame({
        "hybrid_osc": hybrid,
        "hybrid_signal": signal,
        "hybrid_upper": upper_band,
        "hybrid_lower": lower_band,
        "hybrid_chg_1d": hybrid - hybrid.shift(1),
        "hybrid_chg_5d": hybrid - hybrid.shift(5),
        "hybrid_chg_20d": hybrid - hybrid.shift(20),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 6. JK OBOS (Overbought / Oversold)
# ---------------------------------------------------------------------------
def jk_obos(df: pd.DataFrame,
            ma_len: int = 17, lookback: int = 20) -> pd.DataFrame:
    """
    Overbought/Oversold indicator based on distance from moving average,
    normalized by historical maximum distance.
    Value ranges roughly -6 to +6, scaled by /15.

    Original: {WebzG1OBOS}
    """
    c = df["Close"]
    ma = sma(c, ma_len)

    dist_to_ma = (c - ma).abs()
    dist_pct = 100 * dist_to_ma / (ma + 0.00001)
    avg_dist_pct = ema(highest(dist_pct, 24), 7)

    # Average of recent max distances (smoothed)
    max_dist = (
        highest(dist_pct, 24) +
        highest(dist_pct, 24).shift(24) +
        avg_dist_pct.shift(48)
    ) / 3
    # Handle early bars where shift(48) is NaN
    max_dist = max_dist.fillna(
        (highest(dist_pct, 24) + highest(dist_pct, 24).shift(24)) / 2
    )

    cur_diff_pct = 100 * (c - sma(c, ma_len)) / sma(c, ma_len)
    raw_obos = 100 * (cur_diff_pct / (max_dist + 0.00001))
    obos = raw_obos / 15

    return pd.DataFrame({
        "obos": obos,
        "obos_chg_1d": obos - obos.shift(1),
        "obos_chg_5d": obos - obos.shift(5),
        "obos_chg_20d": obos - obos.shift(20),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 7. VE-RSI (Volume-Enhanced RSI)
# ---------------------------------------------------------------------------
def jk_ve_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    Volume-Enhanced RSI — weights up/down price changes by a volume ratio.
    The volume multiplier is V / AvgV(65), so high-volume moves count more.

    This is a simplified translation that avoids the TradeStation-specific
    JKVEweightSimple1() function by using V/AvgV as the multiplier.

    Original: {WebzG1verVE-%RSI}, {WebzverRSIpercent21}
    """
    c = df["Close"]
    v = df["Volume"]

    avg_v = sma(v, 65).replace(0, np.nan)
    # Volume multiplier: current volume / 65-day average
    multip = (v / avg_v).fillna(1.0).clip(0.1, 10.0)

    pct_chg = 100 * c.pct_change()
    up_chg = (pct_chg.clip(lower=0)) * multip
    dn_chg = (-pct_chg.clip(upper=0)) * multip

    avg_up = up_chg.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_dn = dn_chg.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_up / avg_dn.replace(0, np.nan)
    ve_rsi = 100 - 100 / (1 + rs)

    return pd.DataFrame({
        "ve_rsi": ve_rsi,
        "ve_rsi_norm": (ve_rsi - 50) / 3.6,  # Normalized version
        "ve_rsi_chg_1d": ve_rsi - ve_rsi.shift(1),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 8. Z-Factor
# ---------------------------------------------------------------------------
def jk_z_factor(df: pd.DataFrame, fast_len: int = 10, slow_len: int = 21) -> pd.DataFrame:
    """
    Z-Factor — measures where the close falls within the day's range,
    normalized by true range. High values = closing near the high (bullish).

    Z = (Close - Low) / TrueRange * 100
    Then smoothed into fast and slow averages.

    Original: {performancezfactor}
    """
    c = df["Close"]
    lo = df["Low"]
    tr = true_range(df)

    z_of_day = (c - lo + 0.001) / (tr + 0.0001) * 100
    z_fast = sma(z_of_day, fast_len)
    z_slow = sma(z_of_day, slow_len)
    z_fast_exp = ema(z_fast, 9)
    z_slow_exp = ema(z_slow, 9)

    return pd.DataFrame({
        "z_factor_fast": z_fast,
        "z_factor_slow": z_slow,
        "z_factor_fast_exp": z_fast_exp,
        "z_factor_slow_exp": z_slow_exp,
        "z_diff": z_fast - z_slow,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 9. Z-Factor Hybrid
# ---------------------------------------------------------------------------
def jk_z_hybrid(df: pd.DataFrame,
                fast_len: int = 21, slow_len: int = 34) -> pd.DataFrame:
    """
    Z-Factor Hybrid — applies the RSI Hybrid concept to Z-Factor.
    Differential between fast and slow Z + smoothed bias.

    Original: {zfacthybridvers1}
    """
    c = df["Close"]
    lo = df["Low"]
    tr = true_range(df)

    z_of_day = (c - lo + 0.0001) / (tr + 0.0001) * 100
    z_fast = sma(z_of_day, fast_len)
    z_slow = sma(z_of_day, slow_len)
    z_diff = z_fast - z_slow
    z_slow_smooth = ema(z_slow, 3)
    z_hybrid = ema((z_diff + ((z_slow_smooth - 50) / 5)) / 2, 2)

    return pd.DataFrame({
        "z_hybrid": z_hybrid,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 10. Sector Inspector / TrendScore
# ---------------------------------------------------------------------------
def jk_trend_score(df: pd.DataFrame,
                   len1: int = 13, len2: int = 21,
                   len3: int = 34, len4: int = 55) -> pd.DataFrame:
    """
    TrendScore — counts MA alignment across 4 moving averages.
    Tests 10 conditions: C vs each MA (4) + all MA pairs (6).
    Score ranges from -10 to +10, divided by 2 → -5 to +5.

    +5 = perfectly bullish alignment (C > MA1 > MA2 > MA3 > MA4)
    -5 = perfectly bearish alignment

    Original: {webzG1SectorInspector}, {websectorinspector}
    """
    c = df["Close"]
    ma1 = sma(c, len1)
    ma2 = sma(c, len2)
    ma3 = sma(c, len3)
    ma4 = sma(c, len4)

    score = pd.Series(0.0, index=df.index)

    # Price vs MAs
    for ma in [ma1, ma2, ma3, ma4]:
        score += np.where(c > ma, 1, np.where(c < ma, -1, 0))

    # MA pairs
    pairs = [(ma1, ma2), (ma1, ma3), (ma1, ma4), (ma2, ma3), (ma2, ma4), (ma3, ma4)]
    for fast, slow in pairs:
        score += np.where(fast > slow, 1, np.where(fast < slow, -1, 0))

    score = score / 2

    return pd.DataFrame({
        "trend_score": score,
        "trend_score_chg_1d": score - score.shift(1),
        "trend_score_chg_5d": score - score.shift(5),
        "trend_score_chg_20d": score - score.shift(20),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 11. MFOO (Multi-Factor Oscillator Output)
# ---------------------------------------------------------------------------
def jk_mfoo(df: pd.DataFrame,
            rsi_length: int = 14, obos_ma_len: int = 17) -> pd.DataFrame:
    """
    MFOO combines Volume-Enhanced RSI and OBOS into a single oscillator.
    MFOO = ((VE_RSI - 50) / 4 + OBOS) / 2

    This blends momentum (VE-RSI) with mean-reversion (OBOS).

    Original: {perftestingMFOO}, {perftestMFOObb}
    """
    ve = jk_ve_rsi(df, length=rsi_length)
    ob = jk_obos(df, ma_len=obos_ma_len)

    mfoo = (((ve["ve_rsi"] - 50) / 4) + ob["obos"]) / 2

    return pd.DataFrame({
        "mfoo": mfoo,
        "mfoo_chg_1d": mfoo - mfoo.shift(1),
        "mfoo_chg_5d": mfoo - mfoo.shift(5),
    }, index=df.index)


# ---------------------------------------------------------------------------
# 12. RSI Differential (simple)
# ---------------------------------------------------------------------------
def jk_rsi_diff(df: pd.DataFrame,
                length1: int = 34, length2: int = 55) -> pd.DataFrame:
    """
    Simple RSI differential — RSI(medium) - RSI(slow).
    Positive = medium-term momentum exceeding long-term.

    Original: {rsi differential}
    """
    c = df["Close"]
    rsi_med = rsi(c, length1)
    rsi_slow = rsi(c, length2)

    return pd.DataFrame({
        "rsi_diff": rsi_med - rsi_slow,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 13. Stay Power — Breakout/Breakdown Analysis
# ---------------------------------------------------------------------------
def jk_stay_power_high(df: pd.DataFrame, length: int = 44) -> pd.DataFrame:
    """
    Tracks breakouts above rolling highest high.
    Measures how far price is from the prior high, and how many months
    of new highs are being made.

    Original: {webjkstaypwerHi-1dy}
    """
    h = df["High"]
    c = df["Close"]
    upper = highest(h, length).shift(1)

    brkout_diff_pct = 100 * (h - upper) / (upper + 0.0001)
    brkout_close_pct = 100 * (c - upper) / (upper + 0.0001)

    # Score: how many monthly-ish windows are at new highs
    score = pd.Series(0, index=df.index, dtype=int)
    for months, bars in enumerate([22, 44, 67, 89, 111, 134, 156, 178, 200], 1):
        prior_high = highest(h, bars).shift(1)
        score += (h > prior_high).astype(int)

    return pd.DataFrame({
        "breakout_pct": brkout_close_pct,
        "new_high_months": score,
    }, index=df.index)


def jk_stay_power_low(df: pd.DataFrame, length: int = 44) -> pd.DataFrame:
    """
    Tracks breakdowns below rolling lowest low.
    Mirror of stay_power_high.

    Original: {webjkstaypwerLo-1dy}
    """
    lo = df["Low"]
    c = df["Close"]
    lower = lowest(lo, length).shift(1)

    brkdwn_close_pct = 100 * (c - lower) / (lower + 0.0001)

    # Score: how many monthly-ish windows are at new lows
    score = pd.Series(0, index=df.index, dtype=int)
    for months, bars in enumerate([22, 44, 67, 89, 111, 134, 156, 178, 200], 1):
        prior_low = lowest(lo, bars).shift(1)
        score += (lo < prior_low).astype(int)

    return pd.DataFrame({
        "breakdown_pct": brkdwn_close_pct,
        "new_low_months": score,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 14. Quarter-End Markdown Detector
# ---------------------------------------------------------------------------
def jk_quarter_markdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects big losers that may face quarter-end institutional selling.
    Averages 20d, 40d, 60d percentage declines and normalizes by ATR.

    Original: {xxQuaterEndMkdown}
    """
    c = df["Close"]
    atr = (sma(true_range(df), 21) + sma(true_range(df), 55)) / 2

    big_loser = (
        100 * (c - c.shift(20)) / c.shift(20) +
        100 * (c - c.shift(40)) / c.shift(40) +
        100 * (c - c.shift(60)) / c.shift(60)
    ) / 3

    avg_px = (c.shift(40) + c.shift(60)) / 2
    atr_pct = 100 * atr.shift(60) / (avg_px + 0.0001)
    vol_comp = big_loser / atr_pct.clip(lower=0.001)

    return pd.DataFrame({
        "big_loser": big_loser,
        "vol_comp_markdown": vol_comp,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 15. RSI Saul System (Mean-Reversion Trading System)
# ---------------------------------------------------------------------------
def jk_rsi_saul(df: pd.DataFrame,
                ma_len: int = 200, rsi_len: int = 2,
                buy_lev1: int = 60, buy_lev2: int = 10,
                exit_long: int = 75,
                sell_lev1: int = 40, sell_lev2: int = 90,
                exit_short: int = 25) -> pd.DataFrame:
    """
    RSI-based mean reversion system using very short RSI (2-period).
    Buy: price above 200 MA + RSI dropped 3 consecutive bars + RSI extremes.
    Sell: price below 200 MA + RSI risen 3 consecutive bars + RSI extremes.

    Original: {saul_RSIC}
    """
    c = df["Close"]
    ma = sma(c, ma_len)
    rsi_val = rsi(c, rsi_len)

    # Detect 3 consecutive bars of RSI decline
    rsi_fell = (
        (rsi_val < rsi_val.shift(1)) &
        (rsi_val.shift(1) < rsi_val.shift(2)) &
        (rsi_val.shift(2) < rsi_val.shift(3))
    )
    rsi_rose = (
        (rsi_val > rsi_val.shift(1)) &
        (rsi_val.shift(1) > rsi_val.shift(2)) &
        (rsi_val.shift(2) > rsi_val.shift(3))
    )

    buy_signal = (c > ma) & rsi_fell & (rsi_val.shift(3) < buy_lev1) & (rsi_val < buy_lev2)
    sell_signal = (c < ma) & rsi_rose & (rsi_val.shift(3) > sell_lev1) & (rsi_val > sell_lev2)

    return pd.DataFrame({
        "rsi_saul_buy": buy_signal.astype(int),
        "rsi_saul_sell": sell_signal.astype(int),
        "rsi_2": rsi_val,
    }, index=df.index)


# ---------------------------------------------------------------------------
# 16. Retracement/Reversal Factor (RRF)
# ---------------------------------------------------------------------------
def jk_rrf(df: pd.DataFrame, period: int = 125) -> pd.DataFrame:
    """
    Retracement/Reversal Factor — measures how many times greater the sum of
    a stock's daily absolute moves is compared to its net directional move.

    Introduced by J. Kornblatt in the 2007 paper "Leaders vs Laggards".
    Based on (and improves upon) Perry Kauffman's Efficiency Ratio by using
    a rolling average in the denominator to prevent near-zero distortion when
    a stock's net move approaches zero.

    Formula:
        numerator   = rolling mean of (sum of |daily moves| over `period` days)
        denominator = rolling mean of (|net move over `period` days|)
        RRF         = numerator / denominator

    Typical values (125-day):
        Median S&P 500 component:   ~12.5  (stock travels 12.5× its net distance)
        Top 50 "noisiest" stocks:   ~30+
        Bottom 50 "smoothest":      ~5

    High RRF = highly oscillating stock = more prone to mean reversion.
    Low RRF  = steadily trending stock  = mean reversion less reliable.

    A useful pre-filter: prefer candidates where RRF > median of scanned universe,
    as they have more "noise budget" available for mean reversion to exploit.

    Original concept: Perry Kauffman Efficiency Ratio (inverted + averaged).
    Implementation: J. Kornblatt (2007).
    """
    c = df["Close"]
    daily_move = c.diff().abs()

    # Numerator: rolling mean of rolling sum of |daily moves|
    rolling_sum_moves = daily_move.rolling(period).sum()
    numerator = rolling_sum_moves.rolling(period).mean()

    # Denominator: rolling mean of |net move over period|
    # Using shift(period) to get the price `period` days ago
    net_moves = (c - c.shift(period)).abs()
    denominator = net_moves.rolling(period).mean()

    rrf = numerator / denominator.replace(0, np.nan)

    return pd.DataFrame({
        "rrf": rrf,
        "rrf_smooth": rrf.rolling(21).mean(),   # 21-day smooth for display
    }, index=df.index)


# ---------------------------------------------------------------------------
# Registry of all indicators for the AutoResearch loop
# ---------------------------------------------------------------------------
INDICATOR_REGISTRY = {
    "multimac": {
        "fn": jk_multimac,
        "signal_col": "multimac",
        "params": {"ma_len_a": 7, "ma_len_b": 11, "ma_len_1": 17,
                   "ma_len_2": 27, "ma_len_3": 44, "ma_len_4": 72},
        "description": "Multi-timeframe EMA convergence (JK MultiMAC)",
    },
    "multimac_fib": {
        "fn": jk_multimac_fib,
        "signal_col": "multimac_fib",
        "params": {"ma_len_a": 8, "ma_len_b": 13, "ma_len_1": 21,
                   "ma_len_2": 34, "ma_len_3": 55, "ma_len_4": 89},
        "description": "Fibonacci-based MultiMAC with pct normalization",
    },
    "multimac_rsi": {
        "fn": jk_multimac_rsi,
        "signal_col": "multimac_rsi",
        "params": {"ma_len_a": 7, "ma_len_b": 11, "ma_len_1": 17,
                   "ma_len_2": 27, "ma_len_3": 44, "ma_len_4": 72, "rsi_len": 5},
        "description": "MultiMAC with adaptive RSI overlay",
    },
    "multimac_dampened": {
        "fn": jk_multimac_dampened,
        "signal_col": "multimac_dampened",
        "params": {"ma_len_aaa": 4, "ma_len_a": 7, "ma_len_b": 11, "ma_len_1": 17,
                   "ma_len_2": 27, "ma_len_3": 44, "ma_len_4": 72,
                   "cap3": 0.021, "damp3": 0.5, "cap4": 0.027, "damp4": 0.2,
                   "cap5": 0.036, "damp5": 0.2},
        "description": "Dampened MultiMAC (capped longer-term diffs)",
    },
    "hybrid_osc": {
        "fn": jk_hybrid_oscillator,
        "signal_col": "hybrid_osc",
        "params": {"length1": 34, "length2": 55, "ma_len": 8, "scale": 2.7},
        "description": "RSI Differential Hybrid Oscillator",
    },
    "obos": {
        "fn": jk_obos,
        "signal_col": "obos",
        "params": {"ma_len": 17, "lookback": 20},
        "description": "Overbought/Oversold distance from MA",
    },
    "ve_rsi": {
        "fn": jk_ve_rsi,
        "signal_col": "ve_rsi",
        "params": {"length": 14},
        "description": "Volume-Enhanced RSI",
    },
    "z_factor": {
        "fn": jk_z_factor,
        "signal_col": "z_factor_fast",
        "params": {"fast_len": 10, "slow_len": 21},
        "description": "Z-Factor (Close-Low)/TrueRange scoring",
    },
    "z_hybrid": {
        "fn": jk_z_hybrid,
        "signal_col": "z_hybrid",
        "params": {"fast_len": 21, "slow_len": 34},
        "description": "Z-Factor Hybrid (Z differential + bias)",
    },
    "trend_score": {
        "fn": jk_trend_score,
        "signal_col": "trend_score",
        "params": {"len1": 13, "len2": 21, "len3": 34, "len4": 55},
        "description": "MA Alignment TrendScore (-5 to +5)",
    },
    "mfoo": {
        "fn": jk_mfoo,
        "signal_col": "mfoo",
        "params": {"rsi_length": 14, "obos_ma_len": 17},
        "description": "Multi-Factor Oscillator (VE-RSI + OBOS blend)",
    },
    "rsi_diff": {
        "fn": jk_rsi_diff,
        "signal_col": "rsi_diff",
        "params": {"length1": 34, "length2": 55},
        "description": "RSI Differential (medium - slow)",
    },
    "stay_power_high": {
        "fn": jk_stay_power_high,
        "signal_col": "breakout_pct",
        "params": {"length": 44},
        "description": "Breakout Stay Power (distance from prior highs)",
    },
    "stay_power_low": {
        "fn": jk_stay_power_low,
        "signal_col": "breakdown_pct",
        "params": {"length": 44},
        "description": "Breakdown Stay Power (distance from prior lows)",
    },
    "quarter_markdown": {
        "fn": jk_quarter_markdown,
        "signal_col": "big_loser",
        "params": {},
        "description": "Quarter-End Markdown Detector",
    },
    "rsi_saul": {
        "fn": jk_rsi_saul,
        "signal_col": "rsi_saul_buy",
        "params": {"ma_len": 200, "rsi_len": 2},
        "description": "RSI Saul Mean-Reversion System",
    },
    "rrf": {
        "fn": jk_rrf,
        "signal_col": "rrf",
        "params": {"period": 125},
        "description": "Retracement/Reversal Factor (J. Kornblatt 2007) — measures oscillation relative to net move",
    },
}
