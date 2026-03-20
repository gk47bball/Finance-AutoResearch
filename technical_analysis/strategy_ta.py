"""
Technical Analysis Strategy Configuration — THE MUTABLE FILE
=============================================================
This is the strategy file that the AutoResearch loop mutates.
It defines which indicators to use, how to combine them, and
trading rules for generating signals.

Hypothesis: EXP-TA09 — Increase the weight of the 'trend_score' indicator and decrease the weight of the 'multimac' indicator.
Based on the current strategy configuration, the 'trend_score' indicator has the highest individual information coefficient (IC) at -0.180 on a 60-day lookback. Increasing its weight in the signal combination and decreasing the weight of the 'multimac' indicator, which has a lower IC of -0.101, should further improve the strategy's overall Sharpe ratio.
"""

STRATEGY_TYPE = "technical_analysis"

# ---------------------------------------------------------------------------
# Universe: What to trade
# ---------------------------------------------------------------------------
UNIVERSE = {
    "tickers": ["SPY"],
    "multi_ticker": ["SPY", "QQQ", "DIA"],
    "sector_etfs": ["XLU", "XLV", "XLF", "XLK", "XLP", "XLI"],
    "avoid": ["TSLA", "AMD", "COIN", "MARA", "RIVN"],
    "period": "10y",
}

# ---------------------------------------------------------------------------
# Indicator Configuration
# ---------------------------------------------------------------------------
INDICATORS = {
    "multimac": {
        "enabled": True,
        "weight": 0.15,  # Decreased weight for the 'multimac' indicator
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
        "weight": 0.10,
        "params": {"length1": 34, "length2": 55, "ma_len": 8, "scale": 2.7},
        "signal_col": "hybrid_osc",
    },
    "ve_rsi": {
        "enabled": True,
        "weight": 0.15,
        "params": {"length": 14},
        "signal_col": "ve_rsi",
    },
    "z_factor": {
        "enabled": True,
        "weight": 0.10,
        "params": {"fast_len": 10, "slow_len": 21},
        "signal_col": "z_factor_fast",
    },
    "z_hybrid": {
        "enabled": True,
        "weight": 0.10,
        "params": {"fast_len": 21, "slow_len": 34},
        "signal_col": "z_hybrid",
    },
    "obos": {
        "enabled": True,
        "weight": 0.10,
        "params": {"ma_len": 17, "lookback": 20},
        "signal_col": "obos",
    },
    "mfoo": {
        "enabled": True,
        "weight": 0.10,
        "params": {"rsi_length": 14, "obos_ma_len": 17},
        "signal_col": "mfoo",
    },
    "trend_score": {
        "enabled": True,
        "weight": 0.25,  # Increased weight for the 'trend_score' indicator
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
    "combination": "weighted_average",
    "normalize": True,
    "lookback_for_zscore": 63,
    "flip_signal": False,
}

# ---------------------------------------------------------------------------
# Trading Rules
# ---------------------------------------------------------------------------
TRADING = {
    "position_sizing": "binary",
    "long_threshold": -0.3,
    "short_threshold": -1.5,
    "allow_short": False,
    "holding_period_min": 1,
    "stop_loss_pct": None,
    "rebalance_frequency": "daily",
}

# ---------------------------------------------------------------------------
# Scenario-Aware Rules
# ---------------------------------------------------------------------------
SCENARIO_RULES = {
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