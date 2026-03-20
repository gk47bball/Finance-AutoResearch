"""
FinAutoResearch Strategy Configuration
=======================================
This file is the ONLY file modified by the AutoResearch optimization loop.
It defines the investment research methodology as structured Python data.
The optimizer agent reads this, proposes changes, and evaluates results
via walk-forward backtesting with Sharpe ratio as the primary metric.

Last modified: 2026-03-20
Experiment: 1 — top_n=15 (KEPT, Sharpe 1.2012)
Sharpe: 1.2012
"""

# ---------------------------------------------------------------------------
# Universe Definition
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "sp500",                      # sp500 | custom
    "min_market_cap": 2_000_000_000,        # $2B minimum
    "exclude_sectors": [],                  # e.g. ["Utilities", "Real Estate"]
    "exclude_tickers": [],                  # specific exclusions
}

# ---------------------------------------------------------------------------
# Pass/Fail Screens — stocks must pass ALL criteria to enter scoring
# ---------------------------------------------------------------------------
SCREENS = [
    {"metric": "market_cap",        "op": ">=", "value": 2_000_000_000},
    {"metric": "avg_volume_30d",    "op": ">=", "value": 500_000},
    {"metric": "revenue_growth_1y", "op": ">",  "value": 0.0},
    {"metric": "debt_to_equity",    "op": "<",  "value": 3.0},
    {"metric": "current_ratio",     "op": ">",  "value": 1.0},
]

# ---------------------------------------------------------------------------
# Multi-Factor Scoring Model
# ---------------------------------------------------------------------------
# Each factor category has a weight (should sum to ~1.0 across categories).
# Within each category, sub_factors have their own weights (should sum to ~1.0).
# Higher values are always better (use _inv suffix for inverted metrics).
FACTORS = {
    "value": {
        "weight": 0.25,
        "sub_factors": {
            "earnings_yield":       0.40,   # 1/PE — higher = cheaper
            "fcf_yield":            0.30,   # FCF/market cap
            "ev_to_ebitda_inv":     0.30,   # 1/EV-to-EBITDA — higher = cheaper
        },
    },
    "quality": {
        "weight": 0.30,
        "sub_factors": {
            "roe":                  0.35,   # Return on equity
            "gross_margin":         0.25,   # Gross margin
            "debt_to_equity_inv":   0.20,   # 1/(1+D/E) — lower debt = higher score
            "interest_coverage":    0.20,   # Interest coverage (proxied)
        },
    },
    "growth": {
        "weight": 0.25,
        "sub_factors": {
            "revenue_growth_3y_cagr": 0.40, # Revenue growth CAGR (proxied from 1yr)
            "eps_growth_1y":          0.35,  # EPS growth trailing
            "rd_to_revenue":          0.25,  # R&D intensity (innovation proxy)
        },
    },
    "momentum": {
        "weight": 0.20,
        "sub_factors": {
            "return_12m_1m":        1.00,   # 12-month return minus last month
        },
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 15,                            # Number of stocks to hold
    "weighting": "score_weighted",           # equal | score_weighted
    "max_sector_pct": 0.30,                 # Max 30% in any single sector
    "rebalance_frequency": "quarterly",     # quarterly | monthly
}

# ---------------------------------------------------------------------------
# Deep Analysis Configuration (for LLM-powered research reports)
# ---------------------------------------------------------------------------
DEEP_ANALYSIS = {
    "top_n_for_deep_dive": 5,               # How many top picks get LLM analysis
    "analyze_10k": True,                    # Read 10-K Risk Factors + MD&A
    "analyze_recent_8k": True,              # Check recent 8-K filings
    "focus_areas": [
        "competitive_moat",
        "management_quality",
        "risk_factors",
        "growth_catalysts",
    ],
}
