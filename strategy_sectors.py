"""
Sector Rotation Strategy Configuration
=======================================
Rotate between sector ETFs based on momentum + relative strength signals.
Uses cross-sectional percentile ranking, same as stock picker.

STRATEGY_TYPE: sector_rotation
Benchmark: SPY
Rebalance: Monthly
"""

STRATEGY_TYPE = "sector_rotation"

# ---------------------------------------------------------------------------
# Universe: Sector ETFs
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "sector_etfs",
    "etfs": [
        "XLK",   # Technology
        "XLF",   # Financials
        "XLE",   # Energy
        "XLV",   # Health Care
        "XLI",   # Industrials
        "XLP",   # Consumer Staples
        "XLY",   # Consumer Discretionary
        "XLU",   # Utilities
        "XLC",   # Communication Services
        "XLRE",  # Real Estate
        "XLB",   # Materials
    ],
}

# ---------------------------------------------------------------------------
# Screens — ETFs must pass to enter scoring
# ---------------------------------------------------------------------------
SCREENS = [
    # No fundamental screens — ETFs don't have PE ratios etc.
    # Volume screen ensures liquidity
    {"metric": "avg_volume_30d", "op": ">=", "value": 500_000},
]

# ---------------------------------------------------------------------------
# Signal Model (equivalent to FACTORS for stock picker)
# ---------------------------------------------------------------------------
FACTORS = {
    "momentum": {
        "weight": 0.60,
        "sub_factors": {
            "return_12m_1m":    0.35,   # 12-month return minus last month
            "return_6m":        0.35,   # 6-month return
            "return_3m":        0.30,   # 3-month return (intermediate trend)
        },
    },
    "trend": {
        "weight": 0.40,
        "sub_factors": {
            "return_12m_1m":    1.00,   # Proxy for relative strength
        },
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 4,                         # Hold top 4 sectors
    "weighting": "score_weighted",       # Weight by signal strength
    "max_sector_pct": 1.0,              # No sector cap (sectors ARE the positions)
    "rebalance_frequency": "monthly",
}
