"""Multi-factor percentile scoring engine."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from data.fundamentals import get_key_ratios
from data.prices import get_prices


def _compute_sub_factor(ticker: str, sub_factor_name: str, ratios: dict) -> float | None:
    """Get value for a sub-factor metric."""
    # Direct ratio lookups
    direct_map = {
        "earnings_yield": "earnings_yield",
        "fcf_yield": "fcf_yield",
        "ev_to_ebitda_inv": "ev_to_ebitda_inv",
        "roe": "roe",
        "roa": "roa",
        "gross_margin": "gross_margin",
        "operating_margin": "operating_margin",
        "profit_margin": "profit_margin",
        "debt_to_equity_inv": "debt_to_equity_inv",
        "current_ratio": "current_ratio",
        "interest_coverage": "current_ratio",  # proxy
        "revenue_growth": "revenue_growth",
        "earnings_growth": "earnings_growth",
        "revenue_growth_1y": "revenue_growth",
        "eps_growth_1y": "earnings_growth",
        "dividend_yield": "dividend_yield",
        "beta": "beta",
    }

    if sub_factor_name in direct_map:
        return ratios.get(direct_map[sub_factor_name])

    # Price-based factors
    if sub_factor_name == "return_12m_1m":
        return _compute_momentum(ticker, 12, skip_last=1)
    if sub_factor_name == "return_6m":
        return _compute_momentum(ticker, 6, skip_last=0)
    if sub_factor_name == "return_12m":
        return _compute_momentum(ticker, 12, skip_last=0)

    # Revenue growth CAGR (3yr) — approximate from recent growth
    if sub_factor_name in ("revenue_growth_3y_cagr",):
        g = ratios.get("revenue_growth")
        if g is not None:
            return g  # Use 1yr as proxy; true 3yr CAGR needs historical data
        return None

    # R&D to revenue
    if sub_factor_name == "rd_to_revenue":
        return None  # Not available from yfinance info; scored as neutral

    return ratios.get(sub_factor_name)


def _compute_momentum(ticker: str, months: int, skip_last: int = 0) -> float | None:
    """Compute price momentum over N months, optionally skipping last M months."""
    end_date = datetime.now()
    if skip_last > 0:
        end_date = end_date - timedelta(days=skip_last * 30)
    start_date = end_date - timedelta(days=months * 30)

    prices = get_prices(
        ticker,
        start_date.strftime("%Y-%m-%d"),
        (end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )
    if prices.empty or len(prices) < 10:
        return None
    return (prices["Close"].iloc[-1] / prices["Close"].iloc[0]) - 1


def score_stocks(tickers: list[str], factors: dict, ratios_cache: dict = None) -> pd.DataFrame:
    """Score tickers using a multi-factor model with percentile ranking.

    Args:
        tickers: List of tickers to score
        factors: Factor config from strategy.py (FACTORS dict)
        ratios_cache: Optional pre-fetched ratios {ticker: ratios_dict}

    Returns:
        DataFrame with columns: ticker, composite_score, and per-factor scores
    """
    if not tickers:
        return pd.DataFrame()

    # Collect raw factor values for all tickers
    records = []
    for ticker in tickers:
        ratios = (ratios_cache or {}).get(ticker) or get_key_ratios(ticker)
        if not ratios:
            continue

        row = {"ticker": ticker}
        for factor_name, factor_cfg in factors.items():
            sub_factors = factor_cfg.get("sub_factors", {})
            raw_values = {}
            for sf_name, sf_weight in sub_factors.items():
                val = _compute_sub_factor(ticker, sf_name, ratios)
                raw_values[sf_name] = val
                row[f"{factor_name}__{sf_name}"] = val
            row[f"_factor_cfg_{factor_name}"] = factor_cfg
        row["sector"] = ratios.get("sector", "Unknown")
        row["market_cap"] = ratios.get("market_cap", 0)
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Percentile rank each sub-factor cross-sectionally
    composite_scores = pd.Series(0.0, index=df.index)

    for factor_name, factor_cfg in factors.items():
        factor_weight = factor_cfg.get("weight", 0)
        sub_factors = factor_cfg.get("sub_factors", {})
        factor_score = pd.Series(0.0, index=df.index)
        total_sf_weight = 0

        for sf_name, sf_weight in sub_factors.items():
            col = f"{factor_name}__{sf_name}"
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().sum() < 3:
                continue

            # Percentile rank (0-1), higher is better
            ranked = series.rank(pct=True, na_option="bottom")
            factor_score += ranked * sf_weight
            total_sf_weight += sf_weight

        if total_sf_weight > 0:
            factor_score /= total_sf_weight  # Normalize to 0-1

        df[f"score_{factor_name}"] = factor_score
        composite_scores += factor_score * factor_weight

    # Normalize composite to 0-100
    total_weight = sum(f.get("weight", 0) for f in factors.values())
    if total_weight > 0:
        composite_scores /= total_weight
    df["composite_score"] = composite_scores * 100

    # Clean up internal columns
    internal_cols = [c for c in df.columns if c.startswith("_factor_cfg_")]
    df = df.drop(columns=internal_cols)

    # Sort by composite score descending
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    return df


def select_portfolio(scored_df: pd.DataFrame, portfolio_config: dict) -> pd.DataFrame:
    """Select top-N stocks with sector concentration limits."""
    top_n = portfolio_config.get("top_n", 20)
    max_sector_pct = portfolio_config.get("max_sector_pct", 0.30)
    max_per_sector = max(1, int(top_n * max_sector_pct))

    selected = []
    sector_counts = {}

    for _, row in scored_df.iterrows():
        if len(selected) >= top_n:
            break
        sector = row.get("sector", "Unknown")
        count = sector_counts.get(sector, 0)
        if count >= max_per_sector:
            continue
        selected.append(row)
        sector_counts[sector] = count + 1

    result = pd.DataFrame(selected)

    # Assign weights
    weighting = portfolio_config.get("weighting", "equal")
    if weighting == "equal":
        result["weight"] = 1.0 / len(result) if len(result) > 0 else 0
    elif weighting == "score_weighted":
        total = result["composite_score"].sum()
        result["weight"] = result["composite_score"] / total if total > 0 else 0
    else:
        result["weight"] = 1.0 / len(result) if len(result) > 0 else 0

    return result.reset_index(drop=True)
