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
    params: dict = None,
    start: str = None,
    end: str = None,
) -> dict:
    """
    Run full historical backtest of the Four Pillars strategy.
    Includes stop loss, trailing stop, and time stop logic.

    Args:
        params: Optional dict of parameter overrides applied to the engine as
                instance attributes (thread-safe — does NOT mutate class state).
                If None, the engine auto-loads state/best_params.json.
    """
    engine = FourPillarsEngine()
    if params:
        for key, val in params.items():
            if hasattr(engine, key):
                setattr(engine, key, val)
    hist = engine.compute_historical(ticker, period, start=start, end=end)

    prices = hist["close"]
    daily_ret = prices.pct_change()

    # Use the graduated position sizing directly from the pillar engine.
    # Stop-loss and trailing-stop are tracked for trades that scale ABOVE baseline.
    position = hist["position"].copy()
    trade_log = []

    # Per-ticker realistic transaction costs
    TICKER_SPREADS = {
        "SPY": 0.0001, "QQQ": 0.0001, "DIA": 0.0002, "IWM": 0.0005,
        "XLK": 0.0003, "XLF": 0.0003, "XLE": 0.0004, "XLV": 0.0003,
        "XLC": 0.0004, "XLI": 0.0004, "XLY": 0.0003, "XLP": 0.0003,
        "XLRE": 0.0005, "XLB": 0.0005, "XLU": 0.0004,
    }
    spread = TICKER_SPREADS.get(ticker, 0.0005)
    cost_per_trade = spread + 0.0003  # spread + 3bps market impact

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

            # Stop loss → drop to baseline (uses engine params, not hardcoded)
            if pnl_pct <= -engine.STOP_LOSS_PCT:
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "STOP_LOSS", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
                tactical_entry_price = None
                continue

            # Adaptive trailing stop (wider for big winners)
            hwm = max(hwm or price, price)
            effective_trail = engine.TRAIL_STOP_PCT
            if pnl_pct > 0.08:
                effective_trail = engine.TRAIL_STOP_PCT * 1.5
            elif pnl_pct > 0.05:
                effective_trail = engine.TRAIL_STOP_PCT * 1.25
            if pnl_pct > engine.TRAIL_ACTIVATE_PCT and price < hwm * (1 - effective_trail):
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "TRAIL_STOP", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
                tactical_entry_price = None
                continue

            # Profit decay: held 30+ days with no movement → reduce to baseline
            if days_held >= 30 and abs(pnl_pct) < 0.01:
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "PROFIT_DECAY", "price": price,
                                  "pnl_pct": pnl_pct, "days_held": days_held})
                tactical_entry_price = None
                continue

            # Hard time stop as absolute backstop → reduce to baseline
            if days_held >= engine.TIME_STOP_DAYS:
                position.iloc[i] = base
                trade_log.append({"date": date, "action": "TIME_STOP", "price": price,
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

    # Realistic transaction costs with minimum rebalance threshold
    trades = position.diff().abs()
    # Minimum 5% position change to count as a trade (reduces churn)
    trades[trades < 0.05] = 0
    strategy_returns -= trades.shift(1).fillna(0) * cost_per_trade

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
    n_trades = len([t for t in trade_log if t["action"] in ("BUY", "TACTICAL_BUY")])

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


# ---------------------------------------------------------------------------
# Regime-Only Baseline
# ---------------------------------------------------------------------------

def backtest_regime_only(
    ticker: str = "SPY",
    period: str = "10y",
    params: dict = None,
    start: str = None,
    end: str = None,
    verbose: bool = True,
) -> dict:
    """
    Regime-only baseline: position = BULL_BASELINE in bull, CHOP_BASELINE in chop,
    BEAR_BASELINE (0) in bear. NO P2/P3/P4 timing signals at all.

    Purpose: quantify how much value the P2/P3/P4 timing pillars actually add over
    the regime filter alone. If full Four Pillars Sharpe is only marginally higher
    than this, the timing system is adding complexity without proportional payoff.

    Compare against backtest_four_pillars() on the same ticker/period.
    """
    engine = FourPillarsEngine()
    if params:
        for key, val in params.items():
            if hasattr(engine, key):
                setattr(engine, key, val)

    hist = engine.compute_historical(ticker, period, start=start, end=end)
    prices = hist["close"]
    daily_ret = prices.pct_change()

    # Regime-only position: ignore all P2/P3/P4 signals
    baseline_map = {
        "bull": engine.BULL_BASELINE,
        "chop": engine.CHOP_BASELINE,
        "bear": engine.BEAR_BASELINE,
    }
    position = hist["regime"].map(baseline_map).fillna(engine.CHOP_BASELINE)

    strategy_returns = position.shift(1) * daily_ret
    strategy_returns = strategy_returns.dropna()

    # Realistic per-ticker transaction costs per regime transition
    TICKER_SPREADS = {
        "SPY": 0.0001, "QQQ": 0.0001, "DIA": 0.0002, "IWM": 0.0005,
        "XLK": 0.0003, "XLF": 0.0003, "XLE": 0.0004, "XLV": 0.0003,
    }
    spread = TICKER_SPREADS.get(ticker, 0.0005)
    cost_per_trade = spread + 0.0003
    trades = position.diff().abs()
    strategy_returns -= trades.shift(1).fillna(0) * cost_per_trade

    bench_returns = daily_ret.loc[strategy_returns.index]

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
    exposure = (position > 0).mean()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  REGIME-ONLY BASELINE — {ticker} ({period})")
        print(f"  (P1 regime filter only — no P2/P3/P4 timing)")
        print(f"{'='*55}")
        print(f"  Sharpe:        {sharpe:.4f}  (benchmark: {bench_sharpe:.4f})")
        print(f"  Annual Return: {s_mean:.1%}  (benchmark: {b_mean:.1%})")
        print(f"  Total Return:  {total_ret:.1%}")
        print(f"  Max Drawdown:  {max_dd:.1%}")
        print(f"  Exposure:      {exposure:.0%}")
        print(f"{'='*55}")

    return {
        "sharpe_ratio": round(sharpe, 4),
        "benchmark_sharpe": round(bench_sharpe, 4),
        "annual_return": round(s_mean, 4),
        "total_return": round(total_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "annual_volatility": round(s_std, 4),
        "exposure_pct": round(float(exposure), 4),
        "equity_curve": cum,
    }


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    ticker: str = "MULTI",
    period: str = "10y",
    train_frac: float = 0.65,
    params: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Walk-forward validation: split historical data into a training window and a
    held-out test window. Run the SAME parameter set on both. A large Sharpe
    degradation signals overfitting to the training period.

    Args:
        ticker:     Ticker or "MULTI" (SPY/QQQ/DIA/IWM composite).
        period:     Total history to use (e.g. "10y").
        train_frac: Fraction of data used as training window (default 0.65 = ~6.5 of 10 years).
        params:     Parameter set to test. Defaults to current best_params.json.

    Returns dict with:
        train_sharpe, test_sharpe, degradation (train - test),
        degradation_pct ((train-test)/train * 100),
        per_ticker breakdown if MULTI,
        verdict: "ROBUST" | "MODERATE" | "OVERFIT"
    """
    from technical_analysis.bot.self_learner import load_best_params, run_backtest_with_params
    import yfinance as yf
    from datetime import datetime, timedelta

    if params is None:
        params = load_best_params()

    # Determine the date split by fetching SPY to get the actual date range
    spy_df = fetch_data("SPY", period)
    all_dates = spy_df.index
    if len(all_dates) < 200:
        raise ValueError("Insufficient data for walk-forward validation")

    split_idx = int(len(all_dates) * train_frac)
    train_end = all_dates[split_idx].strftime("%Y-%m-%d")
    # Use 3 years before train_end as train_start to avoid cold-start issues
    # Actually: use the full early window. Start from the beginning.
    train_start = all_dates[0].strftime("%Y-%m-%d")
    test_start = all_dates[split_idx + 1].strftime("%Y-%m-%d")
    test_end = all_dates[-1].strftime("%Y-%m-%d")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  WALK-FORWARD VALIDATION — {ticker}")
        print(f"  Train: {train_start} → {train_end} ({split_idx} bars)")
        print(f"  Test:  {test_start} → {test_end} ({len(all_dates) - split_idx - 1} bars)")
        print(f"{'='*60}")

    def _run_window(start_dt, end_dt, label):
        if ticker == "MULTI":
            tickers = ["SPY", "QQQ", "DIA", "IWM"]
            weights = {"SPY": 0.35, "QQQ": 0.35, "DIA": 0.15, "IWM": 0.15}
            results_list = {}
            for t in tickers:
                r = backtest_four_pillars(
                    ticker=t, period=period, verbose=False,
                    params=params, start=start_dt, end=end_dt
                )
                results_list[t] = r
            composite = sum(weights[t] * results_list[t]["sharpe_ratio"] for t in tickers)
            for t in tickers:
                r = results_list[t]
                if r["sharpe_ratio"] < r["benchmark_sharpe"]:
                    composite *= 0.90
            return round(composite, 4), {
                t: {"sharpe": results_list[t]["sharpe_ratio"],
                    "bm_sharpe": results_list[t]["benchmark_sharpe"],
                    "annual_return": results_list[t]["annual_return"]}
                for t in tickers
            }
        else:
            r = backtest_four_pillars(
                ticker=ticker, period=period, verbose=False,
                params=params, start=start_dt, end=end_dt
            )
            return round(r["sharpe_ratio"], 4), {}

    train_sharpe, train_per_ticker = _run_window(train_start, train_end, "TRAIN")
    test_sharpe, test_per_ticker = _run_window(test_start, test_end, "TEST")

    degradation = train_sharpe - test_sharpe  # positive = OOS worse, negative = OOS better
    degradation_pct = (degradation / train_sharpe * 100) if train_sharpe != 0 else 0

    # Verdict: based on how much OOS UNDERPERFORMS in-sample.
    # Negative degradation_pct = OOS beat IS = no overfitting.
    if degradation_pct <= 0:
        verdict = "ROBUST"          # OOS = IS or better
    elif degradation_pct < 20:
        verdict = "ROBUST"          # small degradation, acceptable
    elif degradation_pct < 40:
        verdict = "MODERATE"
    else:
        verdict = "OVERFIT"

    if verbose:
        print(f"\n  Results:")
        print(f"  Train Sharpe: {train_sharpe:.4f}")
        print(f"  Test  Sharpe: {test_sharpe:.4f}")
        if degradation_pct <= 0:
            print(f"  OOS Change:   {-degradation:+.4f} ({-degradation_pct:+.1f}%) — OOS BEAT in-sample ✅")
        else:
            print(f"  Degradation:  {degradation:+.4f} ({degradation_pct:+.1f}%) in-sample → OOS")
        print(f"  Verdict:      {verdict}")
        if train_per_ticker:
            print(f"\n  Train per-ticker:")
            for t, d in train_per_ticker.items():
                beat = "✓" if d["sharpe"] >= d["bm_sharpe"] else "✗"
                print(f"    {t}: {d['sharpe']:.4f} vs BM {d['bm_sharpe']:.4f} {beat}")
            print(f"\n  Test per-ticker:")
            for t, d in test_per_ticker.items():
                beat = "✓" if d["sharpe"] >= d["bm_sharpe"] else "✗"
                print(f"    {t}: {d['sharpe']:.4f} vs BM {d['bm_sharpe']:.4f} {beat}")
        print(f"\n  Interpretation:")
        if degradation_pct <= 0:
            print(f"  ✅ OOS period beat in-sample. No overfitting detected.")
            print(f"     (Note: test window includes 2022 bear recovery — that may flatter OOS.)")
        elif verdict == "ROBUST":
            print(f"  ✅ Params generalize well. <20% Sharpe degradation OOS.")
        elif verdict == "MODERATE":
            print(f"  ⚠️  Moderate overfitting. Test Sharpe still positive but degraded.")
            print(f"     Consider whether the test window includes enough bear market exposure.")
        else:
            print(f"  ❌ OVERFIT signal. >40% degradation. Params too tuned to training period.")
            print(f"     Do NOT run more AutoResearch without investigating WHY.")
        print(f"{'='*60}")

    return {
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "degradation": round(degradation, 4),
        "degradation_pct": round(degradation_pct, 1),
        "verdict": verdict,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_per_ticker": train_per_ticker,
        "test_per_ticker": test_per_ticker,
    }
