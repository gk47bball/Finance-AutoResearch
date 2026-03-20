# FinAutoResearch

**Karpathy's AutoResearch pattern applied to equity investing.**

An autonomous system that iterates on investment strategies using LLM agents, walk-forward backtesting, and the same optimize-evaluate-keep/revert loop that Karpathy used to run 700 ML experiments in 2 days.

## The Karpathy Pattern

| AutoResearch (ML) | FinAutoResearch (Investing) |
|---|---|
| `prepare.py` (immutable) | Data clients + backtester + evaluation |
| `train.py` (mutable) | `strategy.py` — factor weights, screens, rules |
| `val_bpb` metric | Sharpe ratio from walk-forward backtest |
| 5-min experiment | Screen → Score → Backtest cycle |
| `program.md` | Research agent instructions |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API keys
cp .env.example .env
# Edit .env with your keys (ANTHROPIC_API_KEY required for LLM features)

# 3. Run a research cycle (no LLM needed)
python cli.py research --no-deep-dive

# 4. Run the optimization loop (requires ANTHROPIC_API_KEY)
python cli.py optimize --experiments 10

# 5. View experiment history
python cli.py experiment-log
```

## Commands

| Command | Description |
|---|---|
| `python cli.py research` | Run one research cycle, generate report |
| `python cli.py research --no-deep-dive` | Skip LLM analysis (free, no API key needed) |
| `python cli.py optimize -n 20` | Run AutoResearch loop (20 experiments) |
| `python cli.py backtest` | Backtest current strategy |
| `python cli.py show-strategy` | Display current strategy parameters |
| `python cli.py analyze AAPL` | Deep-dive one stock (requires API key) |
| `python cli.py macro` | Show macro environment (requires FRED key) |
| `python cli.py experiment-log` | Show optimization history |

## How It Works

### Research Mode
1. Builds universe (S&P 500)
2. Applies pass/fail screens (revenue growth, debt limits, liquidity)
3. Scores stocks with a multi-factor model (value, quality, growth, momentum)
4. Selects top-20 portfolio with sector limits
5. Backtests over 5 years with quarterly rebalancing
6. Optionally deep-dives top picks via Claude (reads SEC 10-K filings)
7. Generates a markdown report

### AutoResearch Loop
1. Establishes baseline Sharpe from current `strategy.py`
2. Claude proposes a focused change (adjust weights, add factors, modify screens)
3. Validates and applies the change (git commit)
4. Runs backtest → new Sharpe
5. **KEEP** if improved, **REVERT** if not
6. Repeats, learning from experiment history

## Architecture

```
strategy.py     ← The ONE file the agent edits (factor weights, screens, rules)
prepare.py      ← Immutable orchestration (data → screen → score → backtest)
loop.py         ← The AutoResearch optimization loop
program.md      ← Natural language instructions for the agent

data/           ← Market data clients (yfinance, FRED, SEC EDGAR)
analysis/       ← Screener, scorer, report generator
evaluation/     ← Walk-forward backtester, performance metrics
agent/          ← Claude-powered research + optimization agents
```

## API Keys

| Key | Required For | Free? |
|---|---|---|
| `ANTHROPIC_API_KEY` | Deep analysis + optimization loop | Paid |
| `FRED_API_KEY` | Macro economic data | Free (register at fred.stlouisfed.org) |
| `SEC_EDGAR_USER_AGENT` | SEC filing access | Free (just your name + email) |

The quantitative pipeline (screening, scoring, backtesting) works without any API keys using yfinance data.

## Known Limitations

- **Survivorship bias**: Uses current S&P 500 constituents
- **Point-in-time approximation**: yfinance provides current fundamentals, not true historical
- **Simple cost model**: Flat commission, no slippage modeling
- **Overfitting risk**: Loop can overfit to backtest period (mitigated by improvement threshold)

## Disclaimer

This is for educational and research purposes only. Not investment advice. Past performance does not guarantee future results.
