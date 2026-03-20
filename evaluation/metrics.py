"""Performance metrics for strategy evaluation."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if downside.empty or downside.std() == 0:
        return float(np.sqrt(TRADING_DAYS) * excess.mean() / 0.001)  # near-infinite
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside.std())


def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


def calmar_ratio(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    annual_ret = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annual_ret / mdd)


def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    if returns.empty or benchmark_returns.empty:
        return 0.0, 1.0
    # Align dates
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return 0.0, 1.0
    aligned.columns = ["portfolio", "benchmark"]

    cov = np.cov(aligned["portfolio"], aligned["benchmark"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 1.0
    alpha = (aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()) * TRADING_DAYS
    return float(alpha), float(beta)


def information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    if returns.empty or benchmark_returns.empty:
        return 0.0
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < 10:
        return 0.0
    aligned.columns = ["portfolio", "benchmark"]
    active = aligned["portfolio"] - aligned["benchmark"]
    if active.std() == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * active.mean() / active.std())


def total_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() - 1)


def annual_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    years = len(returns) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def annual_volatility(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> dict:
    """Compute all performance metrics. Returns dict."""
    a, b = alpha_beta(returns, benchmark_returns)
    return {
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns),
        "alpha": a,
        "beta": b,
        "information_ratio": information_ratio(returns, benchmark_returns),
        "total_return": total_return(returns),
        "annual_return": annual_return(returns),
        "annual_volatility": annual_volatility(returns),
        "win_rate": win_rate(returns),
    }


PRIMARY_METRIC = "sharpe_ratio"


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------

def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    block_size: int = 21,
) -> dict:
    """Circular block bootstrap confidence interval for the Sharpe ratio.

    Uses blocks of ~21 trading days (1 month) to preserve autocorrelation.
    Returns: {"sharpe": float, "ci_lower": float, "ci_upper": float, "se": float}
    """
    if returns.empty or len(returns) < block_size * 2:
        sr = sharpe_ratio(returns)
        return {"sharpe": sr, "ci_lower": sr, "ci_upper": sr, "se": 0.0}

    n = len(returns)
    values = returns.values
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(42)

    bootstrap_sharpes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        # Circular block bootstrap
        block_starts = rng.integers(0, n, size=n_blocks)
        sample_indices = []
        for start in block_starts:
            sample_indices.extend(range(start, start + block_size))
        sample_indices = [idx % n for idx in sample_indices][:n]
        sample = values[sample_indices]

        std = sample.std()
        if std == 0:
            bootstrap_sharpes[b] = 0.0
        else:
            bootstrap_sharpes[b] = float(np.sqrt(TRADING_DAYS) * sample.mean() / std)

    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(bootstrap_sharpes, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_sharpes, (1 - alpha) * 100))

    return {
        "sharpe": sharpe_ratio(returns),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "se": float(np.std(bootstrap_sharpes)),
    }


def sharpe_difference_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_bootstrap: int = 10000,
    block_size: int = 21,
) -> dict:
    """Test whether the strategy Sharpe is significantly greater than the benchmark Sharpe.

    Uses a paired bootstrap approach (Ledoit-Wolf inspired) to account for
    correlation between strategy and benchmark returns.

    Returns: {"delta_sharpe": float, "p_value": float, "significant_05": bool, "significant_10": bool}
    """
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(aligned) < block_size * 2:
        return {"delta_sharpe": 0.0, "p_value": 1.0, "significant_05": False, "significant_10": False}

    aligned.columns = ["strategy", "benchmark"]
    strat_vals = aligned["strategy"].values
    bench_vals = aligned["benchmark"].values
    n = len(strat_vals)

    # Observed Sharpe difference
    obs_sharpe_s = sharpe_ratio(aligned["strategy"])
    obs_sharpe_b = sharpe_ratio(aligned["benchmark"])
    obs_delta = obs_sharpe_s - obs_sharpe_b

    # Center the returns for the null hypothesis (no difference)
    # Under H0: both return series have the same Sharpe
    mean_diff = strat_vals.mean() - bench_vals.mean()
    centered_strat = strat_vals - mean_diff / 2
    centered_bench = bench_vals + mean_diff / 2

    rng = np.random.default_rng(42)
    n_blocks = int(np.ceil(n / block_size))
    bootstrap_deltas = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        block_starts = rng.integers(0, n, size=n_blocks)
        indices = []
        for start in block_starts:
            indices.extend(range(start, start + block_size))
        indices = [idx % n for idx in indices][:n]

        s_sample = centered_strat[indices]
        b_sample = centered_bench[indices]

        s_std = s_sample.std()
        b_std = b_sample.std()

        s_sharpe = np.sqrt(TRADING_DAYS) * s_sample.mean() / s_std if s_std > 0 else 0.0
        b_sharpe = np.sqrt(TRADING_DAYS) * b_sample.mean() / b_std if b_std > 0 else 0.0
        bootstrap_deltas[b] = s_sharpe - b_sharpe

    # One-sided p-value: P(delta >= observed_delta | H0)
    p_value = float(np.mean(bootstrap_deltas >= obs_delta))

    return {
        "delta_sharpe": obs_delta,
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_10": p_value < 0.10,
    }
