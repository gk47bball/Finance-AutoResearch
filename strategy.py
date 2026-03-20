"""
FinAutoResearch Strategy Configuration
=======================================
This file is the ONLY file modified by the AutoResearch optimization loop.
It defines the investment research methodology as structured Python data.
The optimizer agent reads this, proposes changes, and evaluates results
via walk-forward backtesting with Sharpe ratio as the primary metric.

Last modified: 2026-03-20
Experiment: 53 — Reduce growth to 0.25, boost quality to 0.40 (with single EPS signal, growth is noisier; quality is more stable)
Hypothesis: Pure EPS growth is noisy in downturns; balanced weight with stronger quality gives better risk-adjusted returns
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
        "weight": 0.10,
        "sub_factors": {
            "earnings_yield":       0.25,   # 1/PE — higher = cheaper
            "fcf_yield":            0.35,   # FCF/market cap — manipulation-resistant
            "ps_ratio_inv":         0.25,   # 1/P-to-Sales — more reliably populated
            "dividend_yield":       0.15,   # Dividend yield — quality income signal
        },
    },
    "quality": {
        "weight": 0.40,
        "sub_factors": {
            "gross_margin":         0.40,   # Gross margin — Novy-Marx best quality signal
            "roe":                  0.30,   # Return on equity — compounder signal
            "operating_margin":     0.05,   # Operating margin (small signal, not removed)
            "roa":                  0.10,   # Return on assets (capital efficiency)
            "debt_to_equity_inv":   0.15,   # 1/(1+D/E) — lower debt = higher score
        },
    },
    "growth": {
        "weight": 0.25,
        "sub_factors": {
            "eps_growth_1y":        1.00,   # EPS growth — pure single-signal growth factor
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
