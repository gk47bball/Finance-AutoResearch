"""
Four Pillars Backtest
======================
Backtests the Four Pillars strategy against historical data
and compares with the original 9-indicator weighted average approach.
"""

import numpy as np
import pandas as pd

from technical_analysis.bot.pillars import FourPillarsEngine
from technical_analysis.backtest.signal_tester import fetch_data


def backtest_four_pillars(
    ticker: str = "SPY",
    period: str = "10y",
    initial_capital: float = 100_000.0,
    verbose: bool = True,
) -> dict:
    """
    Run full historical backtest of the Four Pillars strategy.
    Includes stop loss, trailing stop, and time stop logic.
    """
    engine = FourPillarsEngine()
    hist = engine.compute_historical(ticker, period)

    prices = hist["close"]
    daily_ret = prices.pct_change()

    # Use the graduated position sizing directly from the pillar engine.
    # Stop-loss and trailing-stop are tracked for trades that scale ABOVE baseline.
    position = hist["position"].copy()
    trade_log = []

    # Track tactical entries (above baseline) for stop-loss purposes
    tactical_entry_price = None
    tactical_entry_idx = None
    hwm = None
    baseline = {
        "bull": engine.BULL_BASELINE,
        "chop": engine.CHOP_BASELINE,
        "bear": engine.BEAR_BASELINE,
    }

    for i in range(1, len(position)):
        prev_pos = position.iloc[i - 1]
        target_pos = position.iloc[i]
        price = prices.iloc[i]
        date = hist.index[i]
        regime = hist["regime"].iloc[i]
        base = baseline.get(regime, 0)

        # Track entries above baseline (tactical positions)
        if target_pos > base and prev_pos <= base:
            tactical_entry_price = price
            tactical_entry_idx = i
            hwm = price
            trade_log.append({"date": date, "action": "TACTICAL_BUY",
                              "price": price, "position": target_pos})

        # Exit checks for tactical positions
        if tactical_entry_price is not None and target_pos > base:
            pnl_pct = (price - tactical_entry_price) / tactical_entry_price
            days_held = (date - hist.index[tactical_entry_idx]).days

            # Stop loss: -5% on the tactical portion → drop to baseline
            if pnl_pct <= -0.05:
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "STOP_LOSS", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
                tactical_entry_price = None
                continue

            # Trailing stop
            hwm = max(hwm or price, price)
            if pnl_pct > 0.03 and price < hwm * 0.98:
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "TRAIL_STOP", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
                tactical_entry_price = None
                continue

        # Reset tactical tracking when back to baseline
        if target_pos <= base:
            if tactical_entry_price is not None:
                pnl_pct = (price - tactical_entry_price) / tactical_entry_price
                days_held = (date - hist.index[tactical_entry_idx]).days
                trade_log.append({"date": date, "action": "SIGNAL_EXIT", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
            tactical_entry_price = None

    # Compute returns
    strategy_returns = position.shift(1) * daily_ret
    strategy_returns = strategy_returns.dropna()

    # Commission: 5bps per trade
    trades = position.diff().abs()
    strategy_returns -= trades.shift(1).fillna(0) * 0.0005

    bench_returns = daily_ret.loc[strategy_returns.index]

    # Metrics
    ann = 252
    s_mean = strategy_returns.mean() * ann
    s_std = strategy_returns.std() * np.sqrt(ann)
    sharpe = s_mean / s_std if s_std > 0 else 0

    b_mean = bench_returns.mean() * ann
    b_std = bench_returns.std() * np.sqrt(ann)
    bench_sharpe = b_mean / b_std if b_std > 0 else 0

    cum = (1 + strategy_returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    max_dd = ((cum / cum.cummax()) - 1).min()

    bench_cum = (1 + bench_returns).cumprod()
    bench_total = bench_cum.iloc[-1] - 1

    exposure = (position > 0).mean()
    n_trades = len([t for t in trade_log if t["action"] == "BUY"])

    # Win rate on completed trades
    exits = [t for t in trade_log if t.get("pnl_pct") is not None]
    wins = [t for t in exits if t["pnl_pct"] > 0]
    win_rate = len(wins) / len(exits) if exits else 0

    avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
    losses = [t for t in exits if t["pnl_pct"] <= 0]
    avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    avg_hold = np.mean([t.get("days_held", 0) for t in exits]) if exits else 0

    # Exit type breakdown
    exit_types = {}
    for t in exits:
        exit_types[t["action"]] = exit_types.get(t["action"], 0) + 1

    results = {
        "sharpe_ratio": round(sharpe, 4),
        "benchmark_sharpe": round(bench_sharpe, 4),
        "annual_return": round(s_mean, 4),
        "benchmark_return": round(b_mean, 4),
        "total_return": round(total_ret, 4),
        "benchmark_total": round(bench_total, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_volatility": round(s_std, 4),
        "exposure_pct": round(exposure, 4),
        "n_trades": n_trades,
        "n_exits": len(exits),
        "win_rate": round(win_rate, 4),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "avg_hold_days": round(avg_hold, 1),
        "exit_types": exit_types,
        "trade_log": trade_log,
        "equity_curve": cum,
        "bench_curve": bench_cum,
    }

    if verbose:
        print(f"\n{'='*55}")
        print(f"  FOUR PILLARS BACKTEST — {ticker} ({period})")
        print(f"{'='*55}")
        print(f"  Sharpe:        {sharpe:.4f}  (benchmark: {bench_sharpe:.4f})")
        print(f"  Annual Return: {s_mean:.1%}  (benchmark: {b_mean:.1%})")
        print(f"  Total Return:  {total_ret:.1%}  (benchmark: {bench_total:.1%})")
        print(f"  Max Drawdown:  {max_dd:.1%}")
        print(f"  Volatility:    {s_std:.1%}")
        print(f"  Exposure:      {exposure:.0%}")
        print(f"  Trades:        {n_trades} entries, {len(exits)} exits")
        print(f"  Win Rate:      {win_rate:.0%}")
        print(f"  Avg Win:       {avg_win*100:+.2f}%  Avg Loss: {avg_loss*100:+.2f}%")
        print(f"  Avg Hold:      {avg_hold:.0f} days")
        print(f"  Exit Types:    {exit_types}")
        print(f"{'='*55}")

    return results
