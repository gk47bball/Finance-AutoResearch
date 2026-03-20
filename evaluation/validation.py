"""
Rigorous validation framework for strategy evaluation.

Provides out-of-sample testing, cross-validation, regime analysis,
parameter sensitivity, and a composite robustness score.
"""

import numpy as np
import pandas as pd
import copy
import importlib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from evaluation.backtest import run_backtest, BacktestResult
from evaluation.metrics import (
    sharpe_ratio, compute_all_metrics, bootstrap_sharpe_ci,
    sharpe_difference_test, PRIMARY_METRIC, TRADING_DAYS,
    max_drawdown, annual_return,
)


@dataclass
class SplitBacktestResult:
    train: BacktestResult = field(default_factory=BacktestResult)
    validation: BacktestResult = field(default_factory=BacktestResult)
    test: BacktestResult = field(default_factory=BacktestResult)
    train_sharpe: float = 0.0
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0


@dataclass
class TimeSeriesCVResult:
    fold_sharpes: list = field(default_factory=list)
    fold_results: list = field(default_factory=list)
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    min_sharpe: float = 0.0
    max_sharpe: float = 0.0


@dataclass
class RegimeResult:
    regimes: dict = field(default_factory=dict)  # regime_name -> {sharpe, alpha, max_dd, n_days, pct_time}
    worst_regime_sharpe: float = 0.0
    best_regime_sharpe: float = 0.0


@dataclass
class SensitivityResult:
    base_sharpe: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    min_sharpe: float = 0.0
    max_sharpe: float = 0.0
    pct_positive: float = 0.0
    fragility_score: float = 0.0  # std/base — lower is more robust


@dataclass
class ValidationReport:
    split: SplitBacktestResult = field(default_factory=SplitBacktestResult)
    cv: TimeSeriesCVResult = field(default_factory=TimeSeriesCVResult)
    bootstrap_ci: dict = field(default_factory=dict)
    significance: dict = field(default_factory=dict)
    regime: RegimeResult = field(default_factory=RegimeResult)
    sensitivity: SensitivityResult = field(default_factory=SensitivityResult)
    robustness_score: float = 0.0


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------

def run_split_backtest(
    strategy_module,
    train_years: int = 6,
    val_years: int = 2,
    test_years: int = 2,
    benchmark: str = "SPY",
    commission_bps: float = 5,
    show_progress: bool = False,
) -> SplitBacktestResult:
    """Run backtest on three non-overlapping time windows."""
    now = datetime.now()
    total_years = train_years + val_years + test_years

    test_end = now
    test_start = now - timedelta(days=test_years * 365)
    val_end = test_start
    val_start = val_end - timedelta(days=val_years * 365)
    train_end = val_start
    train_start = train_end - timedelta(days=train_years * 365)

    fmt = lambda d: d.strftime("%Y-%m-%d")

    train_bt = run_backtest(
        strategy_module,
        start_date=fmt(train_start), end_date=fmt(train_end),
        benchmark=benchmark, commission_bps=commission_bps,
        show_progress=show_progress,
    )
    val_bt = run_backtest(
        strategy_module,
        start_date=fmt(val_start), end_date=fmt(val_end),
        benchmark=benchmark, commission_bps=commission_bps,
        show_progress=show_progress,
    )
    test_bt = run_backtest(
        strategy_module,
        start_date=fmt(test_start), end_date=fmt(test_end),
        benchmark=benchmark, commission_bps=commission_bps,
        show_progress=show_progress,
    )

    return SplitBacktestResult(
        train=train_bt,
        validation=val_bt,
        test=test_bt,
        train_sharpe=train_bt.metrics.get("sharpe_ratio", 0.0),
        val_sharpe=val_bt.metrics.get("sharpe_ratio", 0.0),
        test_sharpe=test_bt.metrics.get("sharpe_ratio", 0.0),
    )


# ---------------------------------------------------------------------------
# K-Fold Expanding-Window Cross-Validation
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    strategy_module,
    n_folds: int = 5,
    embargo_months: int = 1,
    min_train_years: int = 3,
    total_years: int = 10,
    benchmark: str = "SPY",
    commission_bps: float = 5,
    show_progress: bool = False,
) -> TimeSeriesCVResult:
    """Expanding-window time-series cross-validation.

    Each fold uses a growing training window and a fixed-length test window.
    An embargo gap prevents information leakage.
    """
    now = datetime.now()
    total_start = now - timedelta(days=total_years * 365)
    test_window_years = (total_years - min_train_years) / n_folds
    embargo_days = embargo_months * 30

    fold_sharpes = []
    fold_results = []
    fmt = lambda d: d.strftime("%Y-%m-%d")

    for fold in range(n_folds):
        # Training window: total_start to (total_start + min_train_years + fold * test_window_years)
        train_end = total_start + timedelta(days=(min_train_years + fold * test_window_years) * 365)
        test_start = train_end + timedelta(days=embargo_days)
        test_end = test_start + timedelta(days=test_window_years * 365)

        if test_end > now:
            test_end = now

        if (test_end - test_start).days < 60:
            continue

        bt = run_backtest(
            strategy_module,
            start_date=fmt(test_start), end_date=fmt(test_end),
            benchmark=benchmark, commission_bps=commission_bps,
            show_progress=show_progress,
        )

        fold_sharpe = bt.metrics.get("sharpe_ratio", 0.0)
        fold_sharpes.append(fold_sharpe)
        fold_results.append(bt)

    if not fold_sharpes:
        return TimeSeriesCVResult()

    return TimeSeriesCVResult(
        fold_sharpes=fold_sharpes,
        fold_results=fold_results,
        mean_sharpe=float(np.mean(fold_sharpes)),
        std_sharpe=float(np.std(fold_sharpes)),
        min_sharpe=float(min(fold_sharpes)),
        max_sharpe=float(max(fold_sharpes)),
    )


# ---------------------------------------------------------------------------
# Regime Analysis
# ---------------------------------------------------------------------------

def regime_analysis(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> RegimeResult:
    """Classify market regimes and compute per-regime strategy performance.

    Regimes based on rolling 6-month benchmark returns and rolling volatility:
    - Bull: rolling 126-day benchmark return > +10% annualized
    - Bear: rolling 126-day benchmark return < -10% annualized
    - Sideways: everything else
    - High-vol: rolling 63-day annualized vol > 25%
    - Low-vol: rolling 63-day annualized vol < 15%
    """
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 126:
        return RegimeResult()
    aligned.columns = ["portfolio", "benchmark"]

    # Rolling benchmark metrics
    rolling_6m_return = aligned["benchmark"].rolling(126).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    )
    rolling_vol = aligned["benchmark"].rolling(63).std() * np.sqrt(TRADING_DAYS)

    # Classify regimes
    regime_masks = {
        "bull": rolling_6m_return > 0.10,
        "bear": rolling_6m_return < -0.10,
        "sideways": (rolling_6m_return >= -0.10) & (rolling_6m_return <= 0.10),
        "high_vol": rolling_vol > 0.25,
        "low_vol": rolling_vol < 0.15,
    }

    regimes = {}
    for name, mask in regime_masks.items():
        mask = mask.dropna()
        if mask.sum() < 20:
            continue

        regime_port = aligned.loc[mask.index[mask], "portfolio"]
        regime_bench = aligned.loc[mask.index[mask], "benchmark"]

        if len(regime_port) < 20:
            continue

        sr = sharpe_ratio(regime_port)
        metrics = compute_all_metrics(regime_port, regime_bench)

        regimes[name] = {
            "sharpe": sr,
            "alpha": metrics.get("alpha", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "annual_return": metrics.get("annual_return", 0.0),
            "n_days": int(mask.sum()),
            "pct_time": float(mask.sum() / len(mask)),
        }

    sharpe_values = [r["sharpe"] for r in regimes.values()]
    return RegimeResult(
        regimes=regimes,
        worst_regime_sharpe=min(sharpe_values) if sharpe_values else 0.0,
        best_regime_sharpe=max(sharpe_values) if sharpe_values else 0.0,
    )


# ---------------------------------------------------------------------------
# Monte Carlo Parameter Sensitivity
# ---------------------------------------------------------------------------

def parameter_sensitivity(
    strategy_module,
    n_trials: int = 100,
    perturbation_pct: float = 0.20,
    benchmark: str = "SPY",
    commission_bps: float = 5,
    lookback_years: int = 5,
    show_progress: bool = False,
) -> SensitivityResult:
    """Randomly perturb factor weights and measure Sharpe distribution.

    For each trial: perturb all factor category weights by +/-perturbation_pct,
    renormalize to sum to 1.0, run the backtest, record Sharpe.

    The fragility score = std(sharpes) / base_sharpe — lower is more robust.
    """
    # Get base Sharpe
    base_bt = run_backtest(
        strategy_module,
        lookback_years=lookback_years,
        benchmark=benchmark,
        commission_bps=commission_bps,
        show_progress=False,
    )
    base_sharpe = base_bt.metrics.get("sharpe_ratio", 0.0)

    # Get current factor weights
    factors = getattr(strategy_module, "FACTORS", {})
    factor_names = list(factors.keys())
    base_weights = {name: factors[name]["weight"] for name in factor_names}

    rng = np.random.default_rng(42)
    trial_sharpes = []

    for trial in range(n_trials):
        # Perturb weights
        perturbed_weights = {}
        for name, w in base_weights.items():
            perturbation = rng.uniform(-perturbation_pct, perturbation_pct)
            perturbed_weights[name] = max(0.01, w * (1 + perturbation))

        # Renormalize
        total = sum(perturbed_weights.values())
        for name in perturbed_weights:
            perturbed_weights[name] /= total

        # Apply perturbed weights to a copy of the strategy module
        # We modify the FACTORS dict in-place temporarily, then restore
        original_weights = {}
        for name in factor_names:
            original_weights[name] = factors[name]["weight"]
            factors[name]["weight"] = perturbed_weights[name]

        try:
            bt = run_backtest(
                strategy_module,
                lookback_years=lookback_years,
                benchmark=benchmark,
                commission_bps=commission_bps,
                show_progress=False,
            )
            trial_sharpes.append(bt.metrics.get("sharpe_ratio", 0.0))
        finally:
            # Restore original weights
            for name in factor_names:
                factors[name]["weight"] = original_weights[name]

    if not trial_sharpes:
        return SensitivityResult(base_sharpe=base_sharpe)

    mean_s = float(np.mean(trial_sharpes))
    std_s = float(np.std(trial_sharpes))
    fragility = std_s / base_sharpe if base_sharpe > 0 else float("inf")

    return SensitivityResult(
        base_sharpe=base_sharpe,
        mean_sharpe=mean_s,
        std_sharpe=std_s,
        min_sharpe=float(min(trial_sharpes)),
        max_sharpe=float(max(trial_sharpes)),
        pct_positive=float(np.mean([s > 0 for s in trial_sharpes])),
        fragility_score=fragility,
    )


# ---------------------------------------------------------------------------
# Composite Robustness Score
# ---------------------------------------------------------------------------

def compute_robustness_score(
    val_sharpe: float,
    cv_sharpes: list,
    bootstrap_ci: dict,
    significance: dict,
    regime_result: RegimeResult,
    avg_turnover: float,
    fragility_score: float,
    weights: dict = None,
) -> float:
    """Compute a single composite robustness score.

    Default weights:
    - 0.40: validation-set Sharpe (normalized to 0-1 range, cap at 3.0)
    - 0.15: CV consistency (mean/std of cross-val Sharpes)
    - 0.10: statistical significance bonus
    - 0.15: worst regime Sharpe (normalized)
    - 0.10: stability under perturbation (1 - fragility)
    - 0.10: turnover penalty
    """
    if weights is None:
        weights = {
            "val_sharpe": 0.40,
            "cv_consistency": 0.15,
            "significance": 0.10,
            "worst_regime": 0.15,
            "stability": 0.10,
            "turnover": 0.10,
        }

    # Normalized val Sharpe (cap at 3.0, scale to 0-1)
    val_score = min(val_sharpe / 3.0, 1.0) if val_sharpe > 0 else 0.0

    # CV consistency: mean / (std + epsilon)
    if cv_sharpes and len(cv_sharpes) > 1:
        cv_mean = np.mean(cv_sharpes)
        cv_std = np.std(cv_sharpes)
        cv_score = min(cv_mean / (cv_std + 0.1), 1.0) if cv_mean > 0 else 0.0
    else:
        cv_score = 0.5  # neutral if no CV data

    # Significance bonus
    p_value = significance.get("p_value", 1.0)
    if p_value < 0.01:
        sig_score = 1.0
    elif p_value < 0.05:
        sig_score = 0.8
    elif p_value < 0.10:
        sig_score = 0.5
    else:
        sig_score = 0.0

    # Worst regime Sharpe (cap at 2.0)
    worst_regime = regime_result.worst_regime_sharpe
    regime_score = min(max(worst_regime / 2.0, 0.0), 1.0)

    # Stability (1 - fragility, capped at 0-1)
    stability_score = max(0.0, min(1.0 - fragility_score, 1.0))

    # Turnover penalty (200%+ annual turnover = 0 score)
    turnover_score = max(0.0, 1.0 - avg_turnover / 2.0)

    robustness = (
        weights["val_sharpe"] * val_score
        + weights["cv_consistency"] * cv_score
        + weights["significance"] * sig_score
        + weights["worst_regime"] * regime_score
        + weights["stability"] * stability_score
        + weights["turnover"] * turnover_score
    )

    return float(robustness)


# ---------------------------------------------------------------------------
# Full Validation Pipeline
# ---------------------------------------------------------------------------

def run_full_validation(
    strategy_module,
    benchmark: str = "SPY",
    commission_bps: float = 5,
    train_years: int = 6,
    val_years: int = 2,
    test_years: int = 2,
    cv_folds: int = 5,
    sensitivity_trials: int = 50,
    sensitivity_perturbation: float = 0.20,
    show_progress: bool = True,
) -> ValidationReport:
    """Run the complete validation suite and return a ValidationReport."""
    from rich.console import Console
    console = Console()

    if show_progress:
        console.print("\n[bold blue]Running Full Validation Suite[/bold blue]\n")

    # 1. Train/Val/Test split
    if show_progress:
        console.print("  [dim]1/6 Train/Validation/Test split...[/dim]")
    split = run_split_backtest(
        strategy_module,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        benchmark=benchmark,
        commission_bps=commission_bps,
    )

    # 2. Time-series CV
    if show_progress:
        console.print("  [dim]2/6 Time-series cross-validation...[/dim]")
    cv = run_timeseries_cv(
        strategy_module,
        n_folds=cv_folds,
        benchmark=benchmark,
        commission_bps=commission_bps,
    )

    # 3. Bootstrap CI (use full-period backtest)
    if show_progress:
        console.print("  [dim]3/6 Bootstrap confidence intervals...[/dim]")
    full_bt = run_backtest(
        strategy_module,
        lookback_years=train_years + val_years + test_years,
        benchmark=benchmark,
        commission_bps=commission_bps,
        show_progress=False,
    )
    ci = bootstrap_sharpe_ci(full_bt.portfolio_returns)

    # 4. Significance test
    if show_progress:
        console.print("  [dim]4/6 Statistical significance test...[/dim]")
    sig = sharpe_difference_test(full_bt.portfolio_returns, full_bt.benchmark_returns)

    # 5. Regime analysis
    if show_progress:
        console.print("  [dim]5/6 Regime analysis...[/dim]")
    regime = regime_analysis(full_bt.portfolio_returns, full_bt.benchmark_returns)

    # 6. Parameter sensitivity
    if show_progress:
        console.print("  [dim]6/6 Parameter sensitivity ({} trials)...[/dim]".format(sensitivity_trials))
    sens = parameter_sensitivity(
        strategy_module,
        n_trials=sensitivity_trials,
        perturbation_pct=sensitivity_perturbation,
        benchmark=benchmark,
        commission_bps=commission_bps,
    )

    # Composite robustness score
    robustness = compute_robustness_score(
        val_sharpe=split.val_sharpe,
        cv_sharpes=cv.fold_sharpes,
        bootstrap_ci=ci,
        significance=sig,
        regime_result=regime,
        avg_turnover=full_bt.avg_annual_turnover,
        fragility_score=sens.fragility_score,
    )

    return ValidationReport(
        split=split,
        cv=cv,
        bootstrap_ci=ci,
        significance=sig,
        regime=regime,
        sensitivity=sens,
        robustness_score=robustness,
    )
