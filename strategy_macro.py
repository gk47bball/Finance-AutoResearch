"""
Global Macro / Tactical Asset Allocation Strategy Configuration
================================================================
Allocate between asset classes (stocks, bonds, gold, cash) based on
macro regime classification and momentum overlay.

STRATEGY_TYPE: tactical_allocation
Benchmark: SPY
Rebalance: Monthly
"""

STRATEGY_TYPE = "tactical_allocation"

# ---------------------------------------------------------------------------
# Universe: Asset Class ETFs
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "asset_classes",
    "assets": {
        "equity": "SPY",
        "bonds_long": "TLT",
        "bonds_mid": "IEF",
        "gold": "GLD",
        "cash": "SHY",
    },
}

# ---------------------------------------------------------------------------
# Screens — all assets pass (ETFs always liquid)
# ---------------------------------------------------------------------------
SCREENS = []

# ---------------------------------------------------------------------------
# Signal Model — momentum-based allocation among asset classes
# ---------------------------------------------------------------------------
FACTORS = {
    "momentum": {
        "weight": 0.60,
        "sub_factors": {
            "return_6m":        0.50,   # 6-month return
            "return_12m_1m":    0.50,   # 12-month return minus last month
        },
    },
    "trend": {
        "weight": 0.40,
        "sub_factors": {
            "return_3m":        1.00,   # 3-month trend direction
        },
    },
}

# ---------------------------------------------------------------------------
# Regime Rules — not used in percentile scoring, but available for future
# ---------------------------------------------------------------------------
REGIME_RULES = {
    "risk_on": {
        "conditions": [
            {"indicator": "yield_curve_spread", "op": ">", "value": 0.0},
            {"indicator": "vix", "op": "<", "value": 25},
        ],
        "match": "majority",
        "allocation": {"equity": 0.60, "bonds_long": 0.20, "gold": 0.10, "cash": 0.10},
    },
    "risk_off": {
        "conditions": [
            {"indicator": "vix", "op": ">", "value": 30},
        ],
        "match": "any",
        "allocation": {"equity": 0.10, "bonds_long": 0.30, "gold": 0.30, "cash": 0.30},
    },
    "neutral": {
        "allocation": {"equity": 0.40, "bonds_mid": 0.30, "gold": 0.15, "cash": 0.15},
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 5,                         # Allocate across all 5 asset classes
    "weighting": "score_weighted",
    "max_sector_pct": 0.60,             # Max 60% in any single asset
    "rebalance_frequency": "monthly",
}
