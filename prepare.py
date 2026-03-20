"""
Immutable orchestration layer — the 'prepare.py' of the Karpathy pattern.

This file connects the mutable strategy files to the data, analysis, and
evaluation modules. It MUST NOT be modified by the optimization loop.

Supports multiple strategy domains via STRATEGY_TYPE dispatch.
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


DOMAIN_MODULE_MAP = {
    "stock_picker": "strategy",
    "sector_rotation": "strategy_sectors",
    "tactical_allocation": "strategy_macro",
    "long_short": "strategy_longshort",
    "crypto_momentum": "strategy_crypto",
}


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


def load_strategy(domain: str = "stock_picker"):
    """Load (or reload) the strategy module for the given domain."""
    module_name = DOMAIN_MODULE_MAP.get(domain, domain)
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def run_full_cycle(strategy_module=None, show_progress: bool = True, domain: str = None) -> CycleResult:
    """Execute one complete research cycle, dispatching by strategy type."""
    if strategy_module is None:
        strategy_module = load_strategy(domain=domain or "stock_picker")

    strategy_type = getattr(strategy_module, "STRATEGY_TYPE", "stock_picker")

    dispatch = {
        "stock_picker": _run_stock_picker_cycle,
        "sector_rotation": _run_sector_rotation_cycle,
        "tactical_allocation": _run_tactical_cycle,
        "long_short_equity": _run_longshort_cycle,
        "crypto_momentum": _run_crypto_cycle,
    }

    runner = dispatch.get(strategy_type, _run_stock_picker_cycle)
    return runner(strategy_module, show_progress)


def _run_stock_picker_cycle(strategy_module, show_progress: bool) -> CycleResult:
    """Original stock picker pipeline."""
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


def _run_sector_rotation_cycle(strategy_module, show_progress: bool) -> CycleResult:
    """Sector rotation: score sector ETFs by momentum + macro, hold top N."""
    config = load_config()
    domain_cfg = config.get("domains", {}).get("sector_rotation", {})

    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    etfs = universe_cfg.get("etfs", [])
    signals = getattr(strategy_module, "SIGNALS", {})
    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})

    # Score ETFs using the same percentile-rank engine (signals act like factors)
    from data.fundamentals import get_key_ratios
    scored = score_stocks(etfs, signals, ratios_cache={})

    portfolio = select_portfolio(scored, portfolio_cfg) if not scored.empty else pd.DataFrame()

    backtest_result = run_backtest(
        strategy_module,
        lookback_years=config.get("backtest", {}).get("lookback_years", 5),
        rebalance_freq="monthly",
        benchmark=domain_cfg.get("benchmark", "SPY"),
        commission_bps=domain_cfg.get("commission_bps", 2),
        show_progress=show_progress,
    )

    return CycleResult(
        universe_size=len(etfs),
        screened_count=len(etfs),
        scored_df=scored,
        portfolio=portfolio,
        backtest=backtest_result,
        primary_metric=backtest_result.primary_metric,
    )


def _run_tactical_cycle(strategy_module, show_progress: bool) -> CycleResult:
    """Tactical allocation: regime-based asset class allocation."""
    config = load_config()
    domain_cfg = config.get("domains", {}).get("tactical_allocation", {})

    backtest_result = run_backtest(
        strategy_module,
        lookback_years=config.get("backtest", {}).get("lookback_years", 5),
        rebalance_freq="monthly",
        benchmark=domain_cfg.get("benchmark", "SPY"),
        commission_bps=domain_cfg.get("commission_bps", 2),
        show_progress=show_progress,
    )

    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    assets = universe_cfg.get("assets", {})

    return CycleResult(
        universe_size=len(assets),
        screened_count=len(assets),
        backtest=backtest_result,
        primary_metric=backtest_result.primary_metric,
    )


def _run_longshort_cycle(strategy_module, show_progress: bool) -> CycleResult:
    """Long-short equity: long top-N, short bottom-N."""
    config = load_config()
    bt_config = config.get("backtest", {})
    domain_cfg = config.get("domains", {}).get("long_short", {})

    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    universe = build_universe(universe_cfg)

    screens = getattr(strategy_module, "SCREENS", [])
    screen_result = run_screen(universe, screens, show_progress=show_progress)

    factors = getattr(strategy_module, "FACTORS", {})
    scored = score_stocks(screen_result.passed, factors, ratios_cache=screen_result.data)

    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})
    portfolio = select_portfolio(scored, portfolio_cfg) if not scored.empty else pd.DataFrame()

    short_n = portfolio_cfg.get("short_n", 10)

    backtest_result = run_backtest(
        strategy_module,
        lookback_years=bt_config.get("lookback_years", 5),
        rebalance_months=bt_config.get("rebalance_months", 3),
        benchmark=domain_cfg.get("benchmark", "SPY"),
        commission_bps=domain_cfg.get("commission_bps", 5),
        allow_short=True,
        short_n=short_n,
        borrow_cost_bps=domain_cfg.get("borrow_cost_bps", 50),
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


def _run_crypto_cycle(strategy_module, show_progress: bool) -> CycleResult:
    """Crypto momentum: score crypto assets by momentum + trend + risk."""
    config = load_config()
    domain_cfg = config.get("domains", {}).get("crypto_momentum", {})

    universe_cfg = getattr(strategy_module, "UNIVERSE", {})
    tickers = universe_cfg.get("tickers", [])
    factors = getattr(strategy_module, "FACTORS", {})
    portfolio_cfg = getattr(strategy_module, "PORTFOLIO", {})

    scored = score_stocks(tickers, factors, ratios_cache={})
    portfolio = select_portfolio(scored, portfolio_cfg) if not scored.empty else pd.DataFrame()

    backtest_result = run_backtest(
        strategy_module,
        lookback_years=config.get("backtest", {}).get("lookback_years", 5),
        rebalance_freq="weekly",
        benchmark=domain_cfg.get("benchmark", "BTC-USD"),
        commission_bps=domain_cfg.get("commission_bps", 10),
        show_progress=show_progress,
    )

    return CycleResult(
        universe_size=len(tickers),
        screened_count=len(tickers),
        scored_df=scored,
        portfolio=portfolio,
        backtest=backtest_result,
        primary_metric=backtest_result.primary_metric,
    )


def evaluate(cycle_result: CycleResult) -> float:
    """Return the single scalar metric the loop optimizes."""
    return cycle_result.primary_metric
