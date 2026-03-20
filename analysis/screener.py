"""Stock screening engine — applies pass/fail filters from strategy."""

import operator
from dataclasses import dataclass, field
from data.fundamentals import get_key_ratios
from data.prices import get_prices
import pandas as pd
from datetime import datetime, timedelta
from rich.progress import Progress, SpinnerColumn, TextColumn


OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass
class ScreenResult:
    passed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    data: dict[str, dict] = field(default_factory=dict)  # ticker -> ratios for passed


def _get_metric_value(ratios: dict, metric: str, ticker: str) -> float | None:
    """Resolve a metric name to a value from ratios or computed data."""
    # Direct ratio lookup
    if metric in ratios:
        return ratios[metric]

    # Computed metrics
    if metric == "avg_volume_30d":
        return ratios.get("avg_volume")

    if metric == "revenue_growth_1y":
        return ratios.get("revenue_growth")

    if metric.startswith("return_"):
        # Price return over N months
        parts = metric.replace("return_", "").replace("m", "")
        try:
            months = int(parts)
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
            prices = get_prices(ticker, start, end)
            if not prices.empty and len(prices) > 1:
                return (prices["Close"].iloc[-1] / prices["Close"].iloc[0]) - 1
        except (ValueError, IndexError):
            pass
        return None

    return None


def run_screen(tickers: list[str], screens: list[dict], show_progress: bool = True) -> ScreenResult:
    """Apply screening criteria to a list of tickers."""
    result = ScreenResult()

    iterator = tickers
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        )
        task = None

    if show_progress:
        with progress:
            task = progress.add_task(f"Screening {len(tickers)} stocks...", total=len(tickers))
            for ticker in tickers:
                _screen_one(ticker, screens, result)
                progress.update(task, advance=1)
    else:
        for ticker in tickers:
            _screen_one(ticker, screens, result)

    return result


def _screen_one(ticker: str, screens: list[dict], result: ScreenResult):
    """Screen a single ticker against all criteria."""
    ratios = get_key_ratios(ticker)
    if not ratios:
        result.failed[ticker] = "no data available"
        return

    for screen in screens:
        metric = screen["metric"]
        op_str = screen["op"]
        threshold = screen["value"]

        value = _get_metric_value(ratios, metric, ticker)
        if value is None:
            result.failed[ticker] = f"{metric}: no data"
            return

        op_func = OPERATORS.get(op_str)
        if op_func is None:
            continue

        if not op_func(value, threshold):
            result.failed[ticker] = f"{metric}={value:.4f} fails {op_str} {threshold}"
            return

    result.passed.append(ticker)
    result.data[ticker] = ratios
