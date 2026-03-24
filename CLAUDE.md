# JK Four Pillars Trading Bot — Agent Guide

This document is written for AI agents. It describes the full architecture, how to run things, empirical lessons, and how to extend the system. Read this before making any changes.

---

## What This Is

A self-improving trading bot that:
1. Generates signals using the **Four Pillars strategy** (regime + timing + momentum + volume)
2. **Paper trades** those signals with simulated capital
3. **Backtests** itself over 10 years of historical data
4. **Auto-researches** parameter improvements via an LLM-in-the-loop optimization loop (Karpathy-style)
5. **Posts everything to Discord** — signals, scans, backtest results, learning progress

The optimization target is **Composite Sharpe** across SPY (35%), QQQ (35%), DIA (15%), IWM (15%) with a 10% penalty per ticker that underperforms its benchmark. Current best: **~0.95** (as of 2026-03-22, after 693 experiments).

---

## Directory Structure

```
AutoResearch/
├── technical_analysis/
│   ├── bot/
│   │   ├── pillars.py           # Core signal engine (Four Pillars)
│   │   ├── backtest_pillars.py  # Historical backtest engine (daily bars)
│   │   ├── intraday_sim.py      # Intraday day-trading simulator
│   │   ├── self_learner.py      # AutoResearch loop (LLM-powered)
│   │   ├── paper_trader.py      # Paper trading with real-time signals
│   │   ├── alerts.py            # Discord webhooks + trade log posting
│   │   ├── discord_bot.py       # Interactive Discord bot (!scan, !learn, etc.)
│   │   ├── scheduled_scan.py    # Called by launchd for automated scans
│   │   ├── post_help.py         # Posts help embeds to Discord #general
│   │   ├── post_backtests.py    # Posts sector ETF backtests + intraday sims to Discord
│   │   ├── cli.py               # CLI: scan, trade, backtest, learn, params, status
│   │   └── state/
│   │       ├── best_params.json     # Current optimal parameters
│   │       ├── learning_log.jsonl   # Full experiment history (every AutoResearch run)
│   │       ├── prompt_evolution.jsonl # Meta-prompt improvements over time
│   │       ├── portfolio.json       # Paper trading portfolio state
│   │       └── alerts.jsonl         # Historical signal log
│   ├── indicators/
│   │   └── jk_indicators.py     # All raw indicator computations
│   ├── backtest/
│   │   └── signal_tester.py     # fetch_data() — pulls yfinance OHLCV
│   └── experiments/             # Historical experiment notes
├── .env                         # API keys and Discord config
└── CLAUDE.md                    # This file
```

---

## The Four Pillars Strategy

Every signal requires all four pillars to align:

| Pillar | What | Signal |
|--------|------|--------|
| **P1 Regime** | trend_score (-5 to +5) from SMA crossovers + ADX + Aroon | BULL (≥2), CHOP, BEAR (≤-2) |
| **P2 Timing** | Z-score of hybrid oscillator — mean-reversion entry timing | oversold (z≤-0.8), neutral, overbought (z≥2.5) |
| **P3 Momentum** | Slope of hybrid oscillator vs signal line | confirming if slope turning up |
| **P4 Volume** | Volume-enhanced RSI + volume ratio vs 65-day avg | confirming if ve_rsi strong + vol>avg |

**Position sizing by regime:**
- BULL: 50% baseline + up to 100% on strong entries
- CHOP: 50% baseline (reduced on overbought)
- BEAR: 0% baseline (flat)

**Stops:**
- Stop loss: -5% from entry → drop to baseline
- Trailing stop: activates after +3% gain, trails 2% below high-water mark
- Time stop: 60 days max hold

---

## Current Best Parameters

From `technical_analysis/bot/state/best_params.json` (do not change without backtesting):

```json
{
  "BULL_THRESHOLD": 3,       // trend_score >= 3 → BULL regime (raised from 2, less whipsaw)
  "BEAR_THRESHOLD": -2,      // trend_score <= -2 → BEAR regime
  "DEEP_OVERSOLD": -1.5,     // z-score threshold for max position
  "OVERSOLD": -0.9,          // z-score threshold for standard entry (loosened from -1.1)
  "OVERBOUGHT": 3.7,         // z-score threshold for reducing position (raised from 2.5)
  "STOP_LOSS_PCT": 0.05,     // 5% stop loss
  "TRAIL_STOP_PCT": 0.015,   // 1.5% trailing stop (tight — locks in gains quickly)
  "TRAIL_ACTIVATE_PCT": 0.03,// trailing stop activates after +3%
  "TIME_STOP_DAYS": 60,      // max 60 days per trade
  "BULL_BASELINE": 0.50,     // always 50% in bull regime
  "CHOP_BASELINE": 0.50,     // always 50% in chop regime
  "BEAR_BASELINE": 0.0,      // always flat in bear regime
  "ZSCORE_LOOKBACK": 63      // bars for z-score normalization (~3 months, vs 42 before)
}
```

**Empirically validated — do not tweak without a clear hypothesis:**
- `BULL_BASELINE=0.50`: 42+ experiments confirmed. Higher values (0.6–0.9) reduce Sharpe to 0.58–0.73. Mean-reversion timing provides more value than raw exposure.
- `CHOP_BASELINE=0.50`: Was the single biggest improvement (+0.25 Sharpe). Don't lower below 0.45.
- `OVERBOUGHT=3.7`: Raised 2.5 → 3.0 → 3.5 → 3.7 through AutoResearch. Letting winners run much longer is critical on QQQ. Do NOT revert below 3.0.
- `BULL_THRESHOLD=3`: Requires stronger trend signal before entering BULL regime. Reduces whipsaws in choppy markets. Confirmed across 3 independent sessions.
- `OVERSOLD=-0.9`: Slightly looser entry than -1.1. Pairs well with the conservative OVERBOUGHT exit.
- `TRAIL_STOP_PCT=0.015`: Tight trailing stop locks in gains quickly once activated. Confirmed across multiple sessions.
- `ZSCORE_LOOKBACK=63`: 3-month normalization window (vs 2-month=42). More stable z-score signals.
- `TIME_STOP_DAYS=60`: 30-day stop identical to 60 in testing. Not a lever.

**Architectural note on params:**
- `FourPillarsEngine.__init__` now auto-loads `state/best_params.json` into instance attributes.
- All engine instances (live scan, backtest, Discord commands) use optimized params automatically.
- `backtest_four_pillars(params=...)` accepts a `params` dict applied as instance overrides — thread-safe.
- Previous code mutated class-level attributes (race condition in parallel backtests) — now fixed.

---

## AutoResearch Loop

**What it does:** Runs an LLM-in-the-loop optimization over strategy parameters. Each experiment:
1. Runs baseline backtest (or MULTI mode: SPY+QQQ+DIA+IWM in parallel)
2. Asks Claude Haiku to propose 3 parameter tweaks simultaneously (batch API call)
3. Runs all 3 backtests in parallel using `ThreadPoolExecutor` — thread-safe (instance attrs, not class attrs)
4. Applies out-of-sample gate: new params must also achieve ≥0.50 Sharpe on a recent 2-year window
5. Keeps the best improvement that passes OOS gate, logs all 3 to JSONL
6. Optionally evolves the system prompt based on what worked (every 20 experiments)

**Search space status (after 550+ experiments):**
- SETTLED (do not change): BULL_BASELINE, CHOP_BASELINE, BEAR_BASELINE, TIME_STOP_DAYS
- CONVERGED IMPROVEMENTS: OVERBOUGHT (2.5→3.7), OVERSOLD (-1.1→-0.9), TRAIL_STOP_PCT (0.02→0.015), ZSCORE_LOOKBACK (42→63), BULL_THRESHOLD (2→3)
- EXHAUSTED: STOP_LOSS_PCT variations (0.04, 0.06 — fail OOS gate), TRAIL_ACTIVATE_PCT=0.04, ZSCORE=84 (unstable)
- UNEXPLORED: DEEP_OVERSOLD below -1.8, BEAR_THRESHOLD beyond -2, OVERBOUGHT > 3.7, TRAIL_ACTIVATE_PCT > 0.04

**OOS gate design (as of March 2026):**
- Gate 1: trailing 4y Sharpe ≥ 0.40
- Gate 2 (relative): candidate 2022 Sharpe ≥ max(baseline_2022 - 0.20, -1.0)
  - NOTE: Fixed -0.30 threshold was unreachable (current params score -0.79 in 2022). Relative gate is correct.

**Run it:**
```bash
cd /Users/gkornblatt/Desktop/AutoResearch
# Standard run (MULTI mode, 30 experiments, 2 hours)
python -m technical_analysis.bot.cli learn --model haiku -n 30 --time 120 --ticker MULTI

# Quick test (5 experiments)
python -m technical_analysis.bot.cli learn --model sonnet -n 5 --ticker MULTI

# Background (use this for long runs)
nohup python -m technical_analysis.bot.cli learn --model haiku -n 50 --time 240 --ticker MULTI > technical_analysis/bot/state/learn.log 2>&1 &
```

**Key insight:** The composite Sharpe objective is MULTI mode. Always run with `--ticker MULTI` unless debugging a single ticker.

**Experiment history:** `state/learning_log.jsonl` — every experiment logged. Read this to understand what's been tried.

---

## Running the Bot

### Daily Operations
```bash
# Scan tickers for signals (posts to Discord)
python -m technical_analysis.bot.cli scan --tickers SPY,QQQ,DIA,IWM,XLK,XLF

# Paper trade cycle
python -m technical_analysis.bot.cli trade --tickers SPY,QQQ,DIA,IWM

# Backtest a single ticker
python -m technical_analysis.bot.cli backtest --tickers SPY --period 10y

# Show current params
python -m technical_analysis.bot.cli params

# Show portfolio status
python -m technical_analysis.bot.cli status
```

### Discord Bot (interactive commands)
```bash
# Start the bot (keep running in background)
cd /Users/gkornblatt/Desktop/AutoResearch
nohup /opt/anaconda3/bin/python -m technical_analysis.bot.discord_bot > technical_analysis/bot/state/discord_bot.log 2>&1 &
```

Discord commands: `!scan [tickers]`, `!trade [tickers]`, `!status`, `!history`, `!params`, `!learn [n]`

### Automated Schedules (launchd)
- **9:15 AM ET weekdays:** Pre-market scan — `com.jkbot.premarket-scan`
- **4:15 PM ET weekdays:** Post-close scan + paper trade — `com.jkbot.postclose-scan`
- **Saturday 10 PM ET:** Weekly AutoResearch run — `com.jkbot.weekly-learn`

Check status: `launchctl list | grep jkbot`

### Backtest Channel Refresh
```bash
python technical_analysis/bot/post_backtests.py
```
Posts sector ETF backtests + intraday day-trading simulations to `#backtest-results`.

---

## Environment Variables (`.env`)

```
ANTHROPIC_API_KEY=          # Required for AutoResearch + Discord bot
JK_DISCORD_WEBHOOK=         # General webhook for scans/alerts
JK_DISCORD_BOT_TOKEN=       # Bot token for interactive commands
JK_DISCORD_BACKTEST_CHANNEL=1485009190748688465
JK_DISCORD_TRADELOG_CHANNEL=1485009191998591107
JK_DISCORD_HELP_CHANNEL=1485098676694155354   # Read-only #bot-guide channel
```

---

## Key Code Paths

### Running a backtest
```python
from technical_analysis.bot.backtest_pillars import backtest_four_pillars
results = backtest_four_pillars(ticker="SPY", period="10y", verbose=True)
# results: sharpe_ratio, benchmark_sharpe, annual_return, max_drawdown, trade_log, ...
```

### Getting current signals
```python
from technical_analysis.bot.pillars import FourPillarsEngine
engine = FourPillarsEngine()
snap = engine.compute("SPY")  # returns PillarSnapshot
# snap.regime, snap.timing_signal, snap.position_pct, snap.signal_label
# snap.multimac_rsi_score  — cross-sectional laggard score (lower = more washed out)
```

### Cross-sectional laggard ranking (multi-ticker scans)
```python
snaps = [engine.compute(t) for t in ["SPY", "QQQ", "DIA", "IWM", "XLK", "XLF"]]
ranked = FourPillarsEngine.rank_snapshots(snaps)
# ranked[0] = highest priority (most washed out by multimac_rsi)
# ranked[0].laggard_rank == 1
```

### Computing Retracement/Reversal Factor
```python
from technical_analysis.indicators.jk_indicators import jk_rrf
rrf_df = jk_rrf(ohlcv_df)
# rrf_df["rrf"]        — raw RRF (typical: 8–25 for S&P components)
# rrf_df["rrf_smooth"] — 21-day smoothed for cleaner display
# High RRF = noisy/oscillating stock = mean reversion more reliable
```

### Posting to Discord
```python
from technical_analysis.bot.alerts import send_alerts, send_discord_scan
# For trade signals:
send_alerts(signal)
# For scan summaries:
send_discord_scan(snapshots, prices)
```

### Running AutoResearch
```python
from technical_analysis.bot.self_learner import run_learning_loop
params, best_sharpe = run_learning_loop(
    max_experiments=30, time_limit_minutes=120,
    model_backend="haiku", ticker="MULTI", period="10y"
)
```

### Walk-forward + regime-only validation (run this before any major AutoResearch push)
```bash
python -m technical_analysis.bot.cli validate --ticker MULTI --period 10y
python -m technical_analysis.bot.cli validate --ticker SPY --period 10y  # shows timing value-add
```

### Date-ranged backtests
```python
from technical_analysis.bot.backtest_pillars import backtest_four_pillars, backtest_regime_only
backtest_four_pillars(ticker="SPY", period="1y", start="2022-01-01", end="2022-12-31")
regime = backtest_regime_only(ticker="SPY", period="10y")
full   = backtest_four_pillars(ticker="SPY", period="10y")
timing_add = full["sharpe_ratio"] - regime["sharpe_ratio"]  # what P2/P3/P4 actually contribute
```

---

## Extending the System

**To add a new indicator:** Add it to `technical_analysis/indicators/jk_indicators.py`, then wire it into `FourPillarsEngine.compute()` in `pillars.py`.

**To add a new parameter to AutoResearch:** Add it to `PARAM_BOUNDS` and `DEFAULT_PARAMS` in `self_learner.py`, and document its meaning in `LEARNER_SYSTEM_PROMPT`.

**To add a new Discord command:** Add a handler in `discord_bot.py` following the existing `@bot.command()` pattern.

**To backtest a sector ETF:** It works out of the box — `backtest_four_pillars(ticker="XLK", period="5y")`.

---

## Theoretical Foundation: The 2007 Research Paper

The strategy's indicators were designed by Jonathan Kornblatt for a 2007 MTAEF award submission: **"Leaders vs Laggards: Revelations from a Technical Indicator"**. Key findings directly relevant to this system:

1. **The `multimac_rsi` indicator IS the paper's validated hybrid** — exact match: EMAs at 7/11/27/44/72 (Fibonacci midpoints) + RSI(5) shifted -50, divided asymmetrically (5.5 when bullish+oversold, 7 otherwise). This indicator ranks stocks by their degree of "lagging" or "leading."

2. **Cross-sectional ranking beats absolute thresholds** — the paper ranked 500 large-cap stocks by the hybrid daily, grouped them into 11 buckets of 30. The bottom group (laggards, Group 11) outperformed the top group (leaders, Group 1) in **12 out of 12** hundred-day periods over 5 years.

3. **The signal is uncorrelated to market returns** — a long-laggard/short-leader position showed **0.03 correlation** to S&P 500 returns. That's essentially pure alpha, not market beta.

4. **Mean reversion is fastest in 2 days** — the paper measured 2-day subsequent performance. The strongest reversion happens quickly. The current 60-day time stop is fine for trend capture, but the bulk of the mean-reversion edge is in the first few days after entry.

5. **`jk_rrf` (Retracement/Reversal Factor)** — new indicator added based on the paper. Measures how many times greater a stock's total daily movement is vs its net directional move. Median S&P 500 component travels ~12.5× its net distance per year. High RRF = more "noisy" = more prone to mean reversion. Use as a pre-filter or display metric.

6. **`FourPillarsEngine.rank_snapshots()`** — new static method. Call after computing a batch of snapshots; assigns `.laggard_rank` (1=most washed out) and sorts by `multimac_rsi_score`. The CLI scan and Discord scan both apply this automatically.

7. **Liquidity filter note** — the paper used "average minimum dollar volume" = average of the 10 lowest daily dollar volume days in a prior 65-day window (≥$10M threshold). The current system's 65-day volume average in VE-RSI is deliberately consistent with this.

---

## Walk-Forward Validation Results (March 2026)

Run: `validate --ticker MULTI --period 10y` (train 65% = 2016–Sept 2022, test 35% = Oct 2022–Mar 2026)

**Key findings:**
- Composite Sharpe degradation: **+12.5%** (train 0.7542 → test 0.6600) — classified **ROBUST**
- Train period beats benchmarks on SPY ✓ QQQ ✓ IWM ✓, **DIA fails** (0.61 vs 0.67 BM)
- Test period beats only DIA ✓; SPY/QQQ/IWM all underperform benchmarks in the high-momentum 2023–2025 bull
- Timing pillars value-add on SPY: **+0.06 Sharpe** over regime-only (0.9022 regime-only → 0.9645 full)
- Regime-only on its own beats benchmark on SPY, DIA; barely on QQQ/IWM

**Interpretation:** The strategy is not overfit (12.5% < 20% threshold). But the test period exposes a **structural weakness in strong trending markets**: QQQ went from 0.96 train → 0.81 test while its benchmark went from 0.83 → 1.12. The 50%-baseline regime system leaves money on the table during secular growth runs. The OOS gate has been upgraded to cover the 4-year trailing window (which includes 2022 bear) + 2022 calendar year specifically.

---

## Empirical Lessons (from 63+ experiments)

1. **CHOP_BASELINE was the biggest lever** — raising from 0.25 → 0.50 was the single biggest Sharpe improvement. Always maintain at 0.50.
2. **BULL_BASELINE plateau at 0.50** — counterintuitively, higher baseline exposure hurts because the mean-reversion timing adds more value than raw exposure. 42 of 63 experiments tested higher values; all were worse.
3. **QQQ and IWM are the hardest tickers** — SPY and DIA benchmark-beat easily; QQQ (secular growth) and IWM (small-cap) require specific treatment. MULTI mode with per-ticker penalty forces the LLM to solve for all four simultaneously.
4. **OVERBOUGHT=2.5 was both the best value AND the ceiling** — ceiling extended to 4.0. Try 3.0 or 3.5. Conservative exits let winners run; aggressive exits (1.5–2.0) cut them too early.
5. **TIME_STOP=60 days** — 30 days tested identically. Not a meaningful lever.
6. **Stop loss 5%** — tighter (2-3%) causes excessive whipsawing; looser (8-10%) increases max drawdown meaningfully.
7. **ZSCORE_LOOKBACK=42 is the current best** — was unexplored before March 2026. Shorter lookback = more reactive timing z-score. Still worth testing 63 or 84.
8. **Parallel experiments had a race condition** — fixed March 2026. Old results from parallel runs (before the fix) may be unreliable. The 0.8705 baseline was verified with the fixed single-threaded computation.
9. **OOS gate upgraded** — now tests both trailing 4-year window (Sharpe ≥ 0.40) AND 2022 calendar year specifically (Sharpe ≥ -0.30). Two-gate system prevents both recency overfitting and bear-market blindness.
10. **DEFAULT_PARAMS synced** — `self_learner.py` DEFAULT_PARAMS now reflects current validated best params including ZSCORE_LOOKBACK=42. Was previously stale (OVERBOUGHT=1.5, CHOP_BASELINE=0.25).
11. **Timing adds real but modest value** — P2/P3/P4 timing pillars add ~+0.06 Sharpe over the P1 regime filter alone (validated March 2026). The regime filter is doing more work. Timing contribution is genuine but the four pillars are not fully orthogonal.

---

## Python Environment

Use `/opt/anaconda3/bin/python` (not system Python). Packages: `yfinance`, `pandas`, `numpy`, `anthropic`, `discord.py`, `requests`, `python-dotenv`.

Install missing packages: `/opt/anaconda3/bin/pip install <package>`
