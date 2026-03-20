"""
Crypto Momentum Strategy Configuration
========================================
Apply momentum + trend-following + risk-adjusted scoring to crypto assets.
All signals are price-derived (no fundamentals for crypto).

STRATEGY_TYPE: crypto_momentum
Benchmark: BTC-USD
Rebalance: Weekly
"""

STRATEGY_TYPE = "crypto_momentum"

# ---------------------------------------------------------------------------
# Universe: Top Crypto Assets (yfinance tickers)
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "crypto",
    "tickers": [
        "BTC-USD",    # Bitcoin
        "ETH-USD",    # Ethereum
        "SOL-USD",    # Solana
        "ADA-USD",    # Cardano
        "AVAX-USD",   # Avalanche
        "DOT-USD",    # Polkadot
        "LINK-USD",   # Chainlink
        "UNI-USD",    # Uniswap
        "ATOM-USD",   # Cosmos
        "NEAR-USD",   # Near Protocol
        "AAVE-USD",   # Aave
        "XLM-USD",    # Stellar
    ],
}

# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
SCREENS = [
    # Crypto doesn't use fundamental screens
    # Volume screen for liquidity
    {"metric": "avg_volume_30d", "op": ">=", "value": 1_000_000},
]

# ---------------------------------------------------------------------------
# Signal Model — all price-derived
# ---------------------------------------------------------------------------
FACTORS = {
    "momentum": {
        "weight": 0.50,
        "sub_factors": {
            "return_3m":        0.40,   # 3-month return
            "return_6m":        0.35,   # 6-month return
            "return_12m_1m":    0.25,   # 12-1 month return
        },
    },
    "trend": {
        "weight": 0.30,
        "sub_factors": {
            "return_3m":        1.00,   # 3-month trend (proxy for above SMA)
        },
    },
    "risk": {
        "weight": 0.20,
        "sub_factors": {
            "beta_inv":         1.00,   # 1/beta — lower correlation with BTC scores higher
        },
    },
}

# ---------------------------------------------------------------------------
# Portfolio Construction
# ---------------------------------------------------------------------------
PORTFOLIO = {
    "top_n": 5,                         # Hold top 5 crypto assets
    "weighting": "equal",               # Equal weight (crypto is too volatile for score weighting)
    "max_sector_pct": 0.40,             # Max 40% in any single asset
    "rebalance_frequency": "monthly",   # Monthly (weekly is expensive)
}
