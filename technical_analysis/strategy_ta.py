"""
Technical Analysis Strategy Configuration — THE MUTABLE FILE
=============================================================
This is the strategy file that the AutoResearch loop mutates.
It defines which indicators to use, how to combine them, and
trading rules for generating signals.

Hypothesis: Baseline — use MultiMAC as primary trend signal
with Hybrid Oscillator for timing entries.

STRATEGY_TYPE: technical_analysis
"""

STRATEGY_TYPE = "technical_analysis"

# ---------------------------------------------------------------------------
# Universe: What to trade
# ---------------------------------------------------------------------------
UNIVERSE = {
    "tickers": ["SPY"],  # Test on SPY first, expand later
    "period": "10y",
}

# ---------------------------------------------------------------------------
# Indicator Configuration
# ---------------------------------------------------------------------------
INDICATORS = {
    "multimac": {
        "enabled": True,
        "weight": 0.30,
        "params": {
            "ma_len_a": 7, "ma_len_b": 11,
            "ma_len_1": 17, "ma_len_2": 27,
            "ma_len_3": 44, "ma_len_4": 72,
        },
        "signal_col": "multimac",
    },
    "multimac_fib": {
        "enabled": True,
        "weight": 0.15,
        "params": {
            "ma_len_a": 8, "ma_len_b": 13,
            "ma_len_1": 21, "ma_len_2": 34,
            "ma_len_3": 55, "ma_len_4": 89,
        },
        "signal_col": "multimac_fib",
    },
    "hybrid_osc": {
        "enabled": True,
        "weight": 0.20,
        "params": {"length1": 34, "length2": 55, "ma_len": 8, "scale": 2.7},
        "signal_col": "hybrid_osc",
    },
    "obos": {
        "enabled": True,
        "weight": 0.15,
        "params": {"ma_len": 17, "lookback": 20},
        "signal_col": "obos",
    },
    "trend_score": {
        "enabled": True,
        "weight": 0.10,
        "params": {"len1": 13, "len2": 21, "len3": 34, "len4": 55},
        "signal_col": "trend_score",
    },
    "z_factor": {
        "enabled": True,
        "weight": 0.10,
        "params": {"fast_len": 10, "slow_len": 21},
        "signal_col": "z_factor_fast",
    },
    # Disabled by default — can be enabled during optimization
    "ve_rsi": {
        "enabled": False,
        "weight": 0.00,
        "params": {"length": 14},
        "signal_col": "ve_rsi",
    },
    "mfoo": {
        "enabled": False,
        "weight": 0.00,
        "params": {"rsi_length": 14, "obos_ma_len": 17},
        "signal_col": "mfoo",
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
    "z_hybrid": {
        "enabled": False,
        "weight": 0.00,
        "params": {"fast_len": 21, "slow_len": 34},
        "signal_col": "z_hybrid",
    },
}

# ---------------------------------------------------------------------------
# Signal Combination Rules
# ---------------------------------------------------------------------------
SIGNAL_RULES = {
    "combination": "weighted_average",  # weighted_average, majority_vote, or strongest
    "normalize": True,                  # Normalize each signal to z-score before combining
    "lookback_for_zscore": 63,          # ~3 months for z-score normalization
}

# ---------------------------------------------------------------------------
# Trading Rules
# ---------------------------------------------------------------------------
TRADING = {
    "position_sizing": "binary",     # binary (in/out), scaled, or always_in
    "long_threshold": 0.5,           # Combined signal > this → go long
    "short_threshold": -0.5,         # Combined signal < this → go short (if enabled)
    "allow_short": False,            # Whether to take short positions
    "holding_period_min": 5,         # Minimum days to hold a position
    "stop_loss_pct": None,           # Optional stop loss (None = disabled)
    "rebalance_frequency": "daily",  # daily, weekly, monthly
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
