"""
Immutable orchestration layer — the 'prepare.py' of the Karpathy pattern.

This file connects the mutable strategy.py to the data, analysis, and
evaluation modules. It MUST NOT be modified by the optimization loop.
"""

import importlib
import sys
from dataclasses import dataclass, field
from data.universe import build_universe
from analysis.screener import run_screen
from analysis.scoring import score_stocks, select_portfolio
from evaluation.backtest import run_backtest, BacktestResult
from evaluation.metrics import PRIMARY_METRIC
import pandas as pd
import yaml
import os


@dataclass
class CycleResult:
    universe_size: int = 0
    screened_count: int = 0
    scored_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: BacktestResult = field(default_factory=BacktestResult)
    primary_metric: float = 0.0


def load_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def load_strategy():
    """Load (or reload) the strategy module."""
    module_name = "strategy"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def run_full_cycle(strategy_module=None, show_progress: bool = True) -> CycleResult:
    """Execute one complete research cycle: universe → screen → score → portfolio → backtest."""
    if strategy_module is None:
        strategy_module = load_strategy()

    config = load_config()
    bt_config = config.get("backtest", {})

    # 1. Build universe
    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    universe = build_universe(universe_cfg)

    # 2. Screen
    screens = getattr(strategy_module, "SCREENS", [])
    screen_result = run_screen(universe, screens, show_progress=show_progress)

    # 3. Score
    factors = getattr(strategy_module, "FACTORS", {})
    scored = score_stocks(
        screen_result.passed, factors, ratios_cache=screen_result.data
    )

    # 4. Select portfolio
    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})
    portfolio = select_portfolio(scored, portfolio_cfg) if not scored.empty else pd.DataFrame()

    # 5. Backtest
    backtest_result = run_backtest(
        strategy_module,
        lookback_years=bt_config.get("lookback_years", 5),
        rebalance_months=bt_config.get("rebalance_months", 3),
        benchmark=bt_config.get("benchmark", "SPY"),
        initial_capital=bt_config.get("initial_capital", 100000),
        commission_bps=bt_config.get("commission_bps", 5),
        show_progress=show_progress,
    )

    return CycleResult(
        universe_size=len(universe),
        screened_count=len(screen_result.passed),
        scored_df=scored,
        portfolio=portfolio,
        backtest=backtest_result,
        primary_metric=backtest_result.primary_metric,
    )


def evaluate(cycle_result: CycleResult) -> float:
    """Return the single scalar metric the loop optimizes."""
    return cycle_result.primary_metric
