"""
FinAutoResearch Strategy Configuration
=======================================
This file is the ONLY file modified by the AutoResearch optimization loop.
It defines the investment research methodology as structured Python data.
The optimizer agent reads this, proposes changes, and evaluates results
via walk-forward backtesting with Sharpe ratio as the primary metric.

Last modified: 2026-03-20
Experiment: 39 — Boost gross_margin weight in quality to 0.35 (Novy-Marx #1), reduce operating_margin to 0.10 (correlated)
Hypothesis: Gross profitability / assets is the strongest academic quality signal (Novy-Marx 2013); operating_margin overlaps so reduce it
Sharpe: 1.4293 (best)
"""

# ---------------------------------------------------------------------------
# Universe Definition
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "sp500",                      # sp500 | custom
    "min_market_cap": 2_000_000_000,        # $2B minimum
    "exclude_sectors": ["Financial Services"],  # Financials have structural high leverage
    "exclude_tickers": [],                  # specific exclusions
}

# ---------------------------------------------------------------------------
# Pass/Fail Screens — stocks must pass ALL criteria to enter scoring
# ---------------------------------------------------------------------------
SCREENS = [
    {"metric": "market_cap",        "op": ">=", "value": 2_000_000_000},
    {"metric": "avg_volume_30d",    "op": ">=", "value": 1_000_000},
    {"metric": "revenue_growth_1y", "op": ">",  "value": 0.0},
    {"metric": "debt_to_equity",    "op": "<",  "value": 2.0},
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
        "weight": 0.20,
        "sub_factors": {
            "earnings_yield":       0.40,   # 1/PE — higher = cheaper
            "fcf_yield":            0.30,   # FCF/market cap
            "ps_ratio_inv":         0.30,   # 1/P-to-Sales — more reliably populated
        },
    },
    "quality": {
        "weight": 0.30,
        "sub_factors": {
            "gross_margin":         0.35,   # Gross margin — Novy-Marx best quality signal
            "roe":                  0.25,   # Return on equity
            "operating_margin":     0.10,   # Operating margin (correlated with gross_margin)
            "roa":                  0.15,   # Return on assets (capital efficiency)
            "debt_to_equity_inv":   0.15,   # 1/(1+D/E) — lower debt = higher score
        },
    },
    "growth": {
        "weight": 0.25,
        "sub_factors": {
            "revenue_growth_3y_cagr": 0.25, # Revenue growth CAGR (proxied from 1yr)
            "eps_growth_1y":          0.50,  # EPS growth trailing — more direct compounding signal
            "rd_to_revenue":          0.25,  # R&D intensity (neutral placeholder)
        },
    },
    "momentum": {
        "weight": 0.25,
        "sub_factors": {
            "return_12m_1m":        1.00,   # 12-month return minus last month
        },
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 10,                            # Number of stocks to hold
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
