"""
Long-Short Equity Strategy Configuration
==========================================
Extends the stock picker to also short the bottom-N stocks.
Tests whether the factor model works symmetrically — does shorting
the worst stocks add alpha beyond just buying the best?

STRATEGY_TYPE: long_short_equity
Benchmark: SPY
Rebalance: Quarterly
"""

STRATEGY_TYPE = "long_short_equity"

# ---------------------------------------------------------------------------
# Universe Definition
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "sp500",
    "min_market_cap": 2_000_000_000,
    "exclude_sectors": ["Financial Services"],
    "exclude_tickers": [],
}

# ---------------------------------------------------------------------------
# Screens — looser than stock picker (need enough stocks for both legs)
# ---------------------------------------------------------------------------
SCREENS = [
    {"metric": "market_cap",        "op": ">=", "value": 2_000_000_000},
    {"metric": "avg_volume_30d",    "op": ">=", "value": 1_000_000},
]

# ---------------------------------------------------------------------------
# Factor Model — same as optimized stock picker
# ---------------------------------------------------------------------------
FACTORS = {
    "value": {
        "weight": 0.10,
        "sub_factors": {
            "earnings_yield":       0.25,
            "fcf_yield":            0.35,
            "ps_ratio_inv":         0.25,
            "dividend_yield":       0.15,
        },
    },
    "quality": {
        "weight": 0.40,
        "sub_factors": {
            "gross_margin":         0.40,
            "roe":                  0.30,
            "operating_margin":     0.05,
            "roa":                  0.10,
            "debt_to_equity_inv":   0.15,
        },
    },
    "growth": {
        "weight": 0.25,
        "sub_factors": {
            "eps_growth_1y":        1.00,
        },
    },
    "momentum": {
        "weight": 0.25,
        "sub_factors": {
            "return_12m_1m":        1.00,
        },
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 10,                        # Long top 10
    "short_n": 10,                      # Short bottom 10
    "long_weight": 0.50,               # 50% gross long
    "short_weight": 0.50,              # 50% gross short (market neutral target)
    "weighting": "score_weighted",
    "max_sector_pct": 0.30,
    "rebalance_frequency": "quarterly",
}

# ---------------------------------------------------------------------------
# Short Configuration
# ---------------------------------------------------------------------------
SHORT_CONFIG = {
    "borrow_cost_bps": 50,             # 50bps annual borrow cost
    "short_from": "bottom",            # Short the bottom-N scored stocks
}
