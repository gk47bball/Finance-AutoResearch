"""
Technical Analysis Strategy Configuration — THE MUTABLE FILE
=============================================================
This is the strategy file that the AutoResearch loop mutates.
It defines which indicators to use, how to combine them, and
trading rules for generating signals.

Hypothesis: EXP-SCENARIO1 — Scenario-aware trading.
Based on full scenario analysis across 50+ stocks, 8 sectors, 4 indices:
- All indicators are mean-reversion (negative IC) — buy oversold, sell overbought
- Strongest on broad ETFs (SPY/QQQ/DIA), NOT on high-beta stocks
- Bull_volatile regime (dips in uptrends) has 2-3x IC of other regimes
- Confluence: 0-2/9 bullish → +30% ann (contrarian buy), 7+/9 → +25% (momentum)
- Optimal hold: 15-30 days (NOT day-trading)
- Signal extremes: bottom 2 deciles capture most alpha

STRATEGY_TYPE: technical_analysis
"""

STRATEGY_TYPE = "technical_analysis"

# ---------------------------------------------------------------------------
# Universe: What to trade
# ---------------------------------------------------------------------------
# Scenario analysis: indices have 2-3x stronger IC than individual stocks.
# Best tickers: SPY (IC -0.135), QQQ (-0.139), DIA (-0.148), IWM (-0.064)
# Sector ETFs: XLU (-0.204), XLV (-0.149), XLF (-0.124), XLK (-0.143)
# AVOID: high-beta individual stocks (IC flips positive/noise)
UNIVERSE = {
    "tickers": ["SPY"],              # Primary backtest ticker
    "multi_ticker": ["SPY", "QQQ", "DIA"],  # Scenario-validated ETFs
    "sector_etfs": ["XLU", "XLV", "XLF", "XLK", "XLP", "XLI"],  # Best sectors
    "avoid": ["TSLA", "AMD", "COIN", "MARA", "RIVN"],  # High-beta: signals flip
    "period": "10y",
}

# ---------------------------------------------------------------------------
# Indicator Configuration
# ---------------------------------------------------------------------------
# Weights: restored from 14-experiment optimization (Sharpe 0.965 on SPY)
# Scenario findings applied to: asset selection, multi-ticker expansion, sector ETFs
INDICATORS = {
    "multimac": {
        "enabled": True,
        "weight": 0.25,  # Backbone: smooth multi-timeframe trend. IC -0.132 SPY, -0.162 DIA
        "params": {
            "ma_len_a": 7, "ma_len_b": 11,
            "ma_len_1": 17, "ma_len_2": 27,
            "ma_len_3": 44, "ma_len_4": 72,
        },
        "signal_col": "multimac",
    },
    "multimac_fib": {
        "enabled": True,
        "weight": 0.15,  # Fib variant: IC -0.162 DIA, -0.204 XLU
        "params": {
            "ma_len_a": 8, "ma_len_b": 13,
            "ma_len_1": 21, "ma_len_2": 34,
            "ma_len_3": 55, "ma_len_4": 89,
        },
        "signal_col": "multimac_fib",
    },
    "hybrid_osc": {
        "enabled": True,
        "weight": 0.15,  # IC -0.196 at 60d. Best in bull_volatile regime
        "params": {"length1": 34, "length2": 55, "ma_len": 8, "scale": 2.7},
        "signal_col": "hybrid_osc",
    },
    "ve_rsi": {
        "enabled": True,
        "weight": 0.15,  # IC -0.179 at 60d. Works all sectors except energy
        "params": {"length": 14},
        "signal_col": "ve_rsi",
    },
    "z_factor": {
        "enabled": True,
        "weight": 0.10,  # IC -0.185 at 20d optimal. Best sector breadth
        "params": {"fast_len": 10, "slow_len": 21},
        "signal_col": "z_factor_fast",
    },
    "z_hybrid": {
        "enabled": True,
        "weight": 0.10,  # Strongest extreme spread (-29.9%). IC -0.167 at 20d
        "params": {"fast_len": 21, "slow_len": 34},
        "signal_col": "z_hybrid",
    },
    "obos": {
        "enabled": True,
        "weight": 0.10,  # IC -0.159 at 40d. Weaker on stocks, strong on SPY/QQQ
        "params": {"ma_len": 17, "lookback": 20},
        "signal_col": "obos",
    },
    "mfoo": {
        "enabled": True,
        "weight": 0.10,  # ve_rsi+obos blend. IC -0.176 at 60d
        "params": {"rsi_length": 14, "obos_ma_len": 17},
        "signal_col": "mfoo",
    },
    "trend_score": {
        "enabled": True,
        "weight": 0.10,  # IC -0.180 at 60d. Explosive bottom decile (+36.5%)
        "params": {"len1": 13, "len2": 21, "len3": 34, "len4": 55},
        "signal_col": "trend_score",
    },
    "rsi_diff": {
        "enabled": False,
        "weight": 0.00,
        "params": {"length1": 34, "length2": 55},
        "signal_col": "rsi_diff",
    },
    "multimac_rsi": {
        "enabled": False,
        "weight": 0.00,
        "params": {
            "ma_len_a": 7, "ma_len_b": 11,
            "ma_len_1": 17, "ma_len_2": 27,
            "ma_len_3": 44, "ma_len_4": 72, "rsi_len": 5,
        },
        "signal_col": "multimac_rsi",
    },
    "multimac_dampened": {
        "enabled": False,
        "weight": 0.00,
        "params": {
            "ma_len_aaa": 4, "ma_len_a": 7, "ma_len_b": 11,
            "ma_len_1": 17, "ma_len_2": 27, "ma_len_3": 44, "ma_len_4": 72,
            "cap3": 0.021, "damp3": 0.5, "cap4": 0.027, "damp4": 0.2,
            "cap5": 0.036, "damp5": 0.2,
        },
        "signal_col": "multimac_dampened",
    },
}

# ---------------------------------------------------------------------------
# Signal Combination Rules
# ---------------------------------------------------------------------------
SIGNAL_RULES = {
    "combination": "weighted_average",  # weighted_average, majority_vote, or strongest
    "normalize": True,                  # Normalize each signal to z-score before combining
    "lookback_for_zscore": 63,          # ~3 months for z-score normalization
    "flip_signal": False,               # Use trend-following direction
}

# ---------------------------------------------------------------------------
# Trading Rules
# ---------------------------------------------------------------------------
TRADING = {
    "position_sizing": "binary",     # binary (in/out), scaled, or always_in
    "long_threshold": -0.3,          # Go long when signal > this (lower = more in market)
    "short_threshold": -1.5,         # Go short when signal < this (if enabled)
    "allow_short": False,            # Whether to take short positions
    "holding_period_min": 1,         # Minimum days to hold a position
    "stop_loss_pct": None,           # Optional stop loss (None = disabled)
    "rebalance_frequency": "daily",  # daily, weekly, monthly
}

# ---------------------------------------------------------------------------
# Scenario-Aware Rules (from scenario analysis findings)
# ---------------------------------------------------------------------------
SCENARIO_RULES = {
    # Signal modulation: disabled — standard threshold-based approach is optimal
    "use_regime_boost": False,
    "use_confluence_boost": False,
    "regime_filter": False,
    "blocked_regimes": [],
    "holding_period_min": 1,
}

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVALUATION = {
    "benchmark": "SPY",
    "test_tickers": ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT"],
    "forward_horizons": [1, 2, 5, 10, 20],
    "commission_bps": 5,
}
