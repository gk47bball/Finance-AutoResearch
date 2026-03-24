"""
Housing Alpha Indicators
=========================
Transforms raw FRED + Zillow data into composite trading signals.

The key insight: individual housing indicators are well-known and quickly priced in.
The alpha comes from combining 10-15 series into composite signals that capture
the housing cycle's position and rate of change, then trading housing-exposed ETFs
before the market fully digests the combined picture.

Indicator categories:
  1. Momentum: rate of change in activity (starts, permits, sales)
  2. Affordability: mortgage rates × prices vs income
  3. Supply/Demand Balance: inventory, months supply, listing dynamics
  4. Price Momentum: Case-Shiller, ZHVI acceleration/deceleration
  5. Rate Regime: mortgage rate trend and level
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def zscore(series: pd.Series, window: int = 36) -> pd.Series:
    """Rolling z-score with specified lookback (default 36 months = 3 years)."""
    mean = series.rolling(window, min_periods=12).mean()
    std = series.rolling(window, min_periods=12).std()
    return (series - mean) / std.replace(0, np.nan)


def mom(series: pd.Series, periods: int = 3) -> pd.Series:
    """Month-over-month percentage change, smoothed over N periods."""
    return series.pct_change(periods) * 100


def yoy(series: pd.Series) -> pd.Series:
    """Year-over-year percentage change."""
    return series.pct_change(12) * 100


def slope(series: pd.Series, window: int = 6) -> pd.Series:
    """Linear regression slope over rolling window. Positive = accelerating."""
    def _slope(x):
        if len(x) < 3 or x.isna().sum() > len(x) // 2:
            return np.nan
        t = np.arange(len(x))
        mask = ~np.isnan(x.values)
        if mask.sum() < 3:
            return np.nan
        coef = np.polyfit(t[mask], x.values[mask], 1)
        return coef[0]
    return series.rolling(window, min_periods=3).apply(_slope, raw=False)


# ---------------------------------------------------------------------------
# Composite indicators
# ---------------------------------------------------------------------------

def compute_activity_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Housing Activity Momentum: combines starts, permits, and sales momentum.
    Positive = housing activity accelerating. Negative = decelerating.

    Parameters:
        activity_mom_window: lookback for momentum (default 3 months)
        activity_zscore_window: lookback for z-score normalization (default 36 months)
    """
    mom_window = params.get("activity_mom_window", 3)
    z_window = params.get("activity_zscore_window", 36)

    components = []
    weights = []

    # Housing starts momentum (weight: 30%)
    if "housing_starts" in df.columns:
        starts_mom = zscore(mom(df["housing_starts"], mom_window), z_window)
        components.append(starts_mom)
        weights.append(0.30)

    # Building permits momentum — most leading (weight: 35%)
    if "housing_permits" in df.columns:
        permits_mom = zscore(mom(df["housing_permits"], mom_window), z_window)
        components.append(permits_mom)
        weights.append(0.35)

    # New home sales momentum (weight: 20%)
    if "new_home_sales" in df.columns:
        sales_mom = zscore(mom(df["new_home_sales"], mom_window), z_window)
        components.append(sales_mom)
        weights.append(0.20)

    # Existing home sales momentum (weight: 15%)
    if "existing_home_sales" in df.columns:
        existing_mom = zscore(mom(df["existing_home_sales"], mom_window), z_window)
        components.append(existing_mom)
        weights.append(0.15)

    if not components:
        return pd.Series(dtype=float)

    # Normalize weights
    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return result.rename("activity_momentum")


def compute_affordability_index(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Housing Affordability Index: combines mortgage rates and prices.
    High = unaffordable (bearish for housing). Low = affordable (bullish).

    Parameters:
        afford_zscore_window: lookback for z-score (default 36 months)
    """
    z_window = params.get("afford_zscore_window", 36)

    components = []
    weights = []

    # Mortgage rate level (higher = less affordable)
    if "mortgage_30y" in df.columns:
        rate_z = zscore(df["mortgage_30y"], z_window)
        components.append(rate_z)
        weights.append(0.40)

    # Mortgage rate momentum (rising = getting worse)
    if "mortgage_30y" in df.columns:
        rate_mom = zscore(mom(df["mortgage_30y"], 3), z_window)
        components.append(rate_mom)
        weights.append(0.20)

    # Home price momentum (rising prices = less affordable)
    if "case_shiller_national" in df.columns:
        price_mom = zscore(yoy(df["case_shiller_national"]), z_window)
        components.append(price_mom)
        weights.append(0.25)

    # CPI shelter (proxy for rental inflation pressure)
    if "cpi_shelter" in df.columns:
        shelter_mom = zscore(yoy(df["cpi_shelter"]), z_window)
        components.append(shelter_mom)
        weights.append(0.15)

    if not components:
        return pd.Series(dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return result.rename("affordability_index")


def compute_supply_demand(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Supply/Demand Balance: inventory vs demand signals.
    Positive = oversupply (bearish). Negative = undersupply (bullish for prices).

    Parameters:
        supply_zscore_window: lookback for z-score (default 36 months)
    """
    z_window = params.get("supply_zscore_window", 36)

    components = []
    weights = []

    # Months supply (higher = more supply = bearish)
    if "months_supply" in df.columns:
        supply_z = zscore(df["months_supply"], z_window)
        components.append(supply_z)
        weights.append(0.40)

    # Active inventory change (rising = more supply)
    if "housing_inventory" in df.columns:
        inv_mom = zscore(mom(df["housing_inventory"], 3), z_window)
        components.append(inv_mom)
        weights.append(0.30)

    # Zillow days to pending (longer = weaker demand)
    if "days_to_pending" in df.columns:
        dtp_z = zscore(df["days_to_pending"], z_window)
        components.append(dtp_z)
        weights.append(0.30)

    if not components:
        return pd.Series(dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return result.rename("supply_demand")


def compute_price_momentum(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Price Momentum: acceleration/deceleration of home prices.
    Positive = prices accelerating. Negative = decelerating.

    Parameters:
        price_mom_window: momentum lookback (default 6 months)
        price_zscore_window: z-score lookback (default 36 months)
    """
    mom_window = params.get("price_mom_window", 6)
    z_window = params.get("price_zscore_window", 36)

    components = []
    weights = []

    if "case_shiller_national" in df.columns:
        cs_slope = zscore(slope(df["case_shiller_national"], mom_window), z_window)
        components.append(cs_slope)
        weights.append(0.50)

    if "zhvi_national" in df.columns:
        zhvi_slope = zscore(slope(df["zhvi_national"], mom_window), z_window)
        components.append(zhvi_slope)
        weights.append(0.50)

    if not components:
        # Fallback: just use Case-Shiller YoY
        if "case_shiller_national" in df.columns:
            return zscore(yoy(df["case_shiller_national"]), z_window).rename("price_momentum")
        return pd.Series(dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return result.rename("price_momentum")


def compute_rate_regime(df: pd.DataFrame, params: dict) -> pd.Series:
    """
    Rate Regime: captures the interest rate environment's impact on housing.
    Positive = rates falling (bullish). Negative = rates rising (bearish).

    Parameters:
        rate_lookback: months for rate trend (default 6)
        rate_zscore_window: z-score lookback (default 36)
    """
    lookback = params.get("rate_lookback", 6)
    z_window = params.get("rate_zscore_window", 36)

    components = []
    weights = []

    # 30Y mortgage rate change (inverted: falling rates = positive signal)
    if "mortgage_30y" in df.columns:
        rate_change = -mom(df["mortgage_30y"], lookback)
        rate_z = zscore(rate_change, z_window)
        components.append(rate_z)
        weights.append(0.40)

    # Yield curve (10Y - 2Y): steepening often precedes housing recovery
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        curve = df["treasury_10y"] - df["treasury_2y"]
        curve_z = zscore(curve, z_window)
        components.append(curve_z)
        weights.append(0.30)

    # Fed funds rate change (inverted: cuts = positive)
    if "fed_funds" in df.columns:
        ff_change = -mom(df["fed_funds"], lookback)
        ff_z = zscore(ff_change, z_window)
        components.append(ff_z)
        weights.append(0.30)

    if not components:
        return pd.Series(dtype=float)

    total_w = sum(weights)
    result = sum(c * (w / total_w) for c, w in zip(components, weights))
    return result.rename("rate_regime")


# ---------------------------------------------------------------------------
# Master composite signal
# ---------------------------------------------------------------------------

def compute_housing_composite(
    df: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """
    Compute all housing indicators and the master composite signal.

    Parameters (in params dict):
        weight_activity: weight for activity momentum (default 0.30)
        weight_affordability: weight for affordability (default 0.20, inverted)
        weight_supply_demand: weight for supply/demand (default 0.15, inverted)
        weight_price_momentum: weight for price momentum (default 0.15)
        weight_rate_regime: weight for rate regime (default 0.20)
        composite_zscore_window: final z-score normalization (default 36)

    Returns DataFrame with columns:
        activity_momentum, affordability_index, supply_demand,
        price_momentum, rate_regime, composite_signal
    """
    # Compute sub-indicators
    activity = compute_activity_momentum(df, params)
    afford = compute_affordability_index(df, params)
    supply = compute_supply_demand(df, params)
    price_mom = compute_price_momentum(df, params)
    rate = compute_rate_regime(df, params)

    # Combine into DataFrame
    indicators = pd.DataFrame(index=df.index)

    if not activity.empty:
        indicators["activity_momentum"] = activity
    if not afford.empty:
        indicators["affordability_index"] = afford
    if not supply.empty:
        indicators["supply_demand"] = supply
    if not price_mom.empty:
        indicators["price_momentum"] = price_mom
    if not rate.empty:
        indicators["rate_regime"] = rate

    # Compute weighted composite
    # NOTE: affordability and supply_demand are INVERTED (high = bearish)
    w_activity = params.get("weight_activity", 0.30)
    w_afford = params.get("weight_affordability", 0.20)
    w_supply = params.get("weight_supply_demand", 0.15)
    w_price = params.get("weight_price_momentum", 0.15)
    w_rate = params.get("weight_rate_regime", 0.20)

    composite = pd.Series(0.0, index=df.index)
    total_weight = 0

    if "activity_momentum" in indicators.columns:
        composite += w_activity * indicators["activity_momentum"].fillna(0)
        total_weight += w_activity
    if "affordability_index" in indicators.columns:
        composite -= w_afford * indicators["affordability_index"].fillna(0)  # inverted
        total_weight += w_afford
    if "supply_demand" in indicators.columns:
        composite -= w_supply * indicators["supply_demand"].fillna(0)  # inverted
        total_weight += w_supply
    if "price_momentum" in indicators.columns:
        composite += w_price * indicators["price_momentum"].fillna(0)
        total_weight += w_price
    if "rate_regime" in indicators.columns:
        composite += w_rate * indicators["rate_regime"].fillna(0)
        total_weight += w_rate

    if total_weight > 0:
        composite /= total_weight

    z_window = params.get("composite_zscore_window", 36)
    indicators["composite_signal"] = zscore(composite, z_window)
    indicators["composite_raw"] = composite

    return indicators
