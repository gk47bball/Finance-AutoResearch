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
    annual_return = (1 + returns.mean()) ** TRADING_DAYS - 1
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return float(annual_return / mdd)


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
