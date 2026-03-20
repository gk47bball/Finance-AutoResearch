"""
Core helper functions used by multiple indicators.
Translates TradeStation built-in functions to pandas equivalents.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average — equivalent to TradeStation Xaverage()."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average — equivalent to TradeStation Average()."""
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — equivalent to TradeStation RSI()."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    """True Range — equivalent to TradeStation TrueRange."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def highest(series: pd.Series, period: int) -> pd.Series:
    """Rolling highest — equivalent to TradeStation Highest()."""
    return series.rolling(period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """Rolling lowest — equivalent to TradeStation Lowest()."""
    return series.rolling(period).min()


def swing_high(series: pd.Series, left: int = 1, right: int = 2) -> pd.Series:
    """Detect swing highs (local maxima confirmed by `right` bars after).
    Returns the swing value where confirmed, NaN otherwise."""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left + right, len(series)):
        candidate = series.iloc[i - right]
        is_high = True
        for j in range(1, left + 1):
            if series.iloc[i - right - j] >= candidate:
                is_high = False
                break
        for j in range(1, right + 1):
            if series.iloc[i - right + j] >= candidate:
                is_high = False
                break
        if is_high:
            result.iloc[i] = candidate
    return result


def swing_low(series: pd.Series, left: int = 1, right: int = 2) -> pd.Series:
    """Detect swing lows (local minima confirmed by `right` bars after)."""
    result = pd.Series(np.nan, index=series.index)
    for i in range(left + right, len(series)):
        candidate = series.iloc[i - right]
        is_low = True
        for j in range(1, left + 1):
            if series.iloc[i - right - j] <= candidate:
                is_low = False
                break
        for j in range(1, right + 1):
            if series.iloc[i - right + j] <= candidate:
                is_low = False
                break
        if is_low:
            result.iloc[i] = candidate
    return result
