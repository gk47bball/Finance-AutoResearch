"""
Self-Learning Loop (Karpathy AutoResearch for the Trading Bot)
===============================================================
Analyzes the bot's own trade log, identifies patterns in wins/losses,
proposes parameter improvements, backtests them, and keeps/reverts.

The mutable "genome" is the set of thresholds in FourPillarsEngine:
  - BULL_THRESHOLD, BEAR_THRESHOLD (regime sensitivity)
  - DEEP_OVERSOLD, OVERSOLD, OVERBOUGHT (timing gates)
  - STOP_LOSS_PCT, TRAIL_STOP_PCT, TRAIL_ACTIVATE_PCT (risk mgmt)
  - TIME_STOP_DAYS (max hold)
  - Position sizing baselines per regime

Each experiment:
  1. Analyze trade log → extract patterns (what's working, what's not)
  2. LLM proposes a parameter tweak with hypothesis
  3. Backtest the tweak on historical data
  4. If Sharpe improves → keep. Otherwise → revert.
  5. Log everything to JSONL for transparency.
"""

import json
import os
import time
import copy
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from technical_analysis.bot.pillars import FourPillarsEngine
from technical_analysis.bot.backtest_pillars import backtest_four_pillars


LEARN_LOG = Path(__file__).parent / "state" / "learning_log.jsonl"
PARAMS_FILE = Path(__file__).parent / "state" / "best_params.json"

# ---------------------------------------------------------------------------
# Parameter genome — the mutable state
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "BULL_THRESHOLD": 2,
    "BEAR_THRESHOLD": -2,
    "DEEP_OVERSOLD": -1.5,
    "OVERSOLD": -0.8,
    "OVERBOUGHT": 1.5,
    "STOP_LOSS_PCT": 0.05,
    "TRAIL_STOP_PCT": 0.02,
    "TRAIL_ACTIVATE_PCT": 0.03,
    "TIME_STOP_DAYS": 60,
    "BULL_BASELINE": 0.50,
    "CHOP_BASELINE": 0.25,
    "BEAR_BASELINE": 0.0,
}

PARAM_BOUNDS = {
    "BULL_THRESHOLD": (1, 4),
    "BEAR_THRESHOLD": (-4, -1),
    "DEEP_OVERSOLD": (-3.0, -1.0),
    "OVERSOLD": (-1.5, -0.3),
    "OVERBOUGHT": (0.8, 2.5),
    "STOP_LOSS_PCT": (0.02, 0.10),
    "TRAIL_STOP_PCT": (0.01, 0.05),
    "TRAIL_ACTIVATE_PCT": (0.02, 0.08),
    "TIME_STOP_DAYS": (20, 90),
    "BULL_BASELINE": (0.25, 0.95),
    "CHOP_BASELINE": (0.0, 0.50),
    "BEAR_BASELINE": (0.0, 0.25),
}


def load_best_params() -> dict:
    """Load the current best parameters (or defaults)."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE) as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


def save_best_params(params: dict):
    """Persist the best parameters."""
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=2)


def apply_params_to_engine(engine: FourPillarsEngine, params: dict):
    """Apply a parameter set to an engine instance."""
    for key, val in params.items():
        if hasattr(engine, key):
            setattr(engine, key, val)


# ---------------------------------------------------------------------------
# Trade log analysis
# ---------------------------------------------------------------------------

def analyze_trade_log(trade_log: list[dict]) -> dict:
    """
    Extract actionable patterns from the trade log.
    Returns a structured analysis dict that the LLM will use.
    """
    if not trade_log:
        return {"status": "empty", "summary": "No trades to analyze."}

    exits = [t for t in trade_log if t.get("pnl_pct") is not None]
    if not exits:
        return {"status": "no_exits", "summary": "No completed trades yet."}

    wins = [t for t in exits if t["pnl_pct"] > 0]
    losses = [t for t in exits if t["pnl_pct"] <= 0]

    # Exit type analysis
    exit_types = {}
    for t in exits:
        action = t["action"]
        if action not in exit_types:
            exit_types[action] = {"count": 0, "avg_pnl": 0, "total_pnl": 0}
        exit_types[action]["count"] += 1
        exit_types[action]["total_pnl"] += t["pnl_pct"]
    for action, data in exit_types.items():
        data["avg_pnl"] = round(data["total_pnl"] / data["count"] * 100, 2)
        data["total_pnl"] = round(data["total_pnl"] * 100, 2)

    # Hold time analysis
    hold_times = [t.get("days_held", 0) for t in exits]
    win_holds = [t.get("days_held", 0) for t in wins]
    loss_holds = [t.get("days_held", 0) for t in losses]

    # Biggest wins/losses
    sorted_trades = sorted(exits, key=lambda t: t["pnl_pct"])
    worst_3 = sorted_trades[:3]
    best_3 = sorted_trades[-3:]

    analysis = {
        "status": "ok",
        "total_trades": len(exits),
        "win_rate": round(len(wins) / len(exits) * 100, 1),
        "avg_win_pct": round(sum(t["pnl_pct"] for t in wins) / len(wins) * 100, 2) if wins else 0,
        "avg_loss_pct": round(sum(t["pnl_pct"] for t in losses) / len(losses) * 100, 2) if losses else 0,
        "avg_hold_days": round(sum(hold_times) / len(hold_times), 1),
        "avg_win_hold": round(sum(win_holds) / len(win_holds), 1) if win_holds else 0,
        "avg_loss_hold": round(sum(loss_holds) / len(loss_holds), 1) if loss_holds else 0,
        "exit_types": exit_types,
        "worst_trades": [{"pnl_pct": round(t["pnl_pct"]*100, 2),
                          "days_held": t.get("days_held", 0),
                          "action": t["action"]} for t in worst_3],
        "best_trades": [{"pnl_pct": round(t["pnl_pct"]*100, 2),
                         "days_held": t.get("days_held", 0),
                         "action": t["action"]} for t in best_3],
    }

    # Generate plain-text summary for the LLM
    lines = [
        f"Trade Log Analysis ({len(exits)} completed trades):",
        f"  Win Rate: {analysis['win_rate']}%",
        f"  Avg Win: +{analysis['avg_win_pct']}% (held {analysis['avg_win_hold']}d)",
        f"  Avg Loss: {analysis['avg_loss_pct']}% (held {analysis['avg_loss_hold']}d)",
        f"  Exit Types:",
    ]
    for action, data in exit_types.items():
        lines.append(f"    {action}: {data['count']} trades, avg P&L: {data['avg_pnl']}%")
    lines.append(f"  Worst Trades: {analysis['worst_trades']}")
    lines.append(f"  Best Trades: {analysis['best_trades']}")
    analysis["summary"] = "\n".join(lines)

    return analysis


# ---------------------------------------------------------------------------
# LLM-powered hypothesis generation
# ---------------------------------------------------------------------------

LEARNER_SYSTEM_PROMPT = """You are optimizing a trading bot's parameters based on its actual trade performance.
The bot uses a "Four Pillars" strategy with these tunable parameters:

HIGH-IMPACT PARAMETERS (these drive most of the Sharpe ratio):
  BULL_BASELINE: Default position in bull regime [0.25-0.95].
    ⚠️  EMPIRICALLY TESTED: Values above 0.55 consistently REDUCE composite Sharpe.
    The current 0.50 is correct — the mean-reversion signal (adding on dips, reducing on peaks)
    provides more value than raw exposure. DO NOT increase above 0.55.
  CHOP_BASELINE: Default position in chop regime [0.0-0.50].
    ⚠️  EMPIRICALLY TESTED: 0.50 is optimal. Increasing CHOP_BASELINE from 0.25 to 0.50 was the
    biggest single improvement — it ensures adequate exposure in mildly-trending "chop" periods.
    Lowering below 0.45 will hurt. DO NOT change significantly.
  BEAR_BASELINE: Default position in bear regime [0.0-0.25]. Keep at 0.0.
  OVERSOLD: z-score threshold for "oversold" entries [-1.5 to -0.3]. Lower = rarer entries.
    Currently -1.0. Values more aggressive than -0.8 hurt DIA.
  DEEP_OVERSOLD: z-score for "deep oversold" [-3.0 to -1.0]. Currently -1.5 (optimal).
  OVERBOUGHT: z-score to reduce positions [0.8 to 2.5]. Currently 2.5 (optimal — conservative reduces).
  BULL_THRESHOLD: trend_score for "bull" regime [1 to 4]. Currently 2 (optimal).
    Higher values (3-4) help QQQ but hurt IWM. Lower (1) helps IWM but hurts SPY Sharpe marginally.
  BEAR_THRESHOLD: trend_score for "bear" regime [-4 to -1]. Currently -2 (optimal).
    More negative values (-3, -4) help SPY but hurt QQQ significantly.

LOWER-IMPACT PARAMETERS (stop/trail logic, rarely triggered):
  STOP_LOSS_PCT [0.02-0.10], TRAIL_STOP_PCT [0.01-0.05], TRAIL_ACTIVATE_PCT [0.02-0.08]
  TIME_STOP_DAYS [20-90]

OBJECTIVE: Maximize COMPOSITE Sharpe across SPY, QQQ, DIA, IWM simultaneously.
  Weights: SPY 35%, QQQ 35%, DIA 15%, IWM 15%.
  A 10% penalty is applied for each ticker that underperforms its benchmark.
  Current best: Composite 0.8705 with ALL tickers beating benchmarks.

RULES:
1. Change 1-2 parameters per experiment. Fine-tuning only — the big levers are already set.
2. Respond with ONLY valid JSON: {"changes": {"PARAM": new_value, ...}, "hypothesis": "why"}
3. The baseline params are already near-optimal. Focus on timing thresholds and stop parameters.
4. DO NOT repeat experiments that already failed (check the history below).
5. AVOID: BULL_BASELINE changes, CHOP_BASELINE changes, BEAR_BASELINE changes (all tested, optimal).
6. TRY: OVERSOLD fine-tuning (-0.9 to -1.2), STOP_LOSS_PCT, TRAIL_ACTIVATE_PCT, TIME_STOP_DAYS.
"""


def propose_with_anthropic(
    current_params: dict,
    trade_analysis: dict,
    backtest_results: dict,
    experiment_history: list[dict],
    model: str = "haiku",
) -> dict:
    """Use Claude Haiku to propose a parameter tweak."""
    import anthropic

    model_id = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-sonnet-4-5-20241022",
    }.get(model, model)

    # Build context
    history_summary = ""
    failed_changes = set()
    if experiment_history:
        recent = experiment_history[-15:]
        history_summary = "\nPast experiments (DO NOT repeat failed ones):\n"
        for exp in recent:
            kept = "KEPT" if exp.get("kept") else "REVERTED"
            changes = exp.get("changes", {})
            params_changed = ", ".join(changes.keys())
            history_summary += (
                f"  [{kept}] Changed {params_changed}: {changes} "
                f"→ Sharpe {exp.get('new_sharpe', '?')}\n"
            )
            if not exp.get("kept"):
                failed_changes.update(changes.keys())

        if failed_changes:
            history_summary += f"\nAlready tried (and failed) changing: {', '.join(failed_changes)}\n"
            history_summary += "Try DIFFERENT parameters this time!\n"

    # Build per-ticker breakdown if available (MULTI mode)
    per_ticker_str = ""
    per_ticker = backtest_results.get("_per_ticker", {})
    if per_ticker:
        per_ticker_str = "\nPer-ticker Sharpe breakdown:\n"
        for t, data in per_ticker.items():
            beat = "✓ BEATS" if data["sharpe"] >= data["bm_sharpe"] else "✗ UNDERPERFORMS"
            per_ticker_str += (
                f"  {t}: Sharpe={data['sharpe']:.4f} vs BM={data['bm_sharpe']:.4f} "
                f"({beat} benchmark) | Annual={data['annual_return']:.1%}\n"
            )
        underperformers = [t for t, d in per_ticker.items() if d["sharpe"] < d["bm_sharpe"]]
        if underperformers:
            per_ticker_str += f"\n  ⚠️  UNDERPERFORMERS NEEDING FIX: {', '.join(underperformers)}\n"
            per_ticker_str += "  → These tickers trend strongly. Increasing BULL_BASELINE will help them most.\n"

    user_msg = f"""Current parameters:
{json.dumps(current_params, indent=2)}

Current backtest results (COMPOSITE across SPY/QQQ/DIA/IWM):
  Composite Sharpe: {backtest_results.get('sharpe_ratio', '?')}
  Win Rate: {backtest_results.get('win_rate', '?')}
  Exposure: {backtest_results.get('exposure_pct', '?')}
  Max Drawdown: {backtest_results.get('max_drawdown', '?')}
  Annual Return: {backtest_results.get('annual_return', '?')}
  Avg Hold: {backtest_results.get('avg_hold_days', '?')} days
  Exit Types: {backtest_results.get('exit_types', '{}')}
{per_ticker_str}
Trade log analysis:
{trade_analysis.get('summary', 'No trade data available.')}
{history_summary}
Propose a parameter change to improve the COMPOSITE Sharpe. Fix underperformers.
Respond with ONLY JSON: {{"changes": {{"PARAM": value}}, "hypothesis": "why"}}"""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if api_key.startswith("sk-ant-oat"):
        client = anthropic.Anthropic(auth_token=api_key)
    else:
        client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=500,
                temperature=0.9,
                system=LEARNER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            text = response.content[0].text.strip()
            if not text:
                raise ValueError("Empty response from API")
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

    # Extract JSON from response
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Find JSON object in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    return json.loads(text)


def propose_with_ollama(
    current_params: dict,
    trade_analysis: dict,
    backtest_results: dict,
    experiment_history: list[dict],
    model: str = "qwen2.5-coder:7b",
) -> dict:
    """Use local Ollama model to propose a parameter tweak."""
    import requests

    history_summary = ""
    if experiment_history:
        recent = experiment_history[-5:]
        history_summary = "\nRecent experiments:\n"
        for exp in recent:
            kept = "KEPT" if exp.get("kept") else "REVERTED"
            history_summary += f"  [{kept}] {exp.get('hypothesis', '?')[:60]} → Sharpe {exp.get('new_sharpe', '?')}\n"

    user_msg = f"""Current params: {json.dumps(current_params)}
Backtest: Sharpe={backtest_results.get('sharpe_ratio')}, WinRate={backtest_results.get('win_rate')}, DD={backtest_results.get('max_drawdown')}
Trade analysis: {trade_analysis.get('summary', 'None')}
{history_summary}
Propose 1-2 param changes as JSON: {{"changes": {{"PARAM": value}}, "hypothesis": "why"}}"""

    resp = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": f"{LEARNER_SYSTEM_PROMPT}\n\nUser: {user_msg}\n\nAssistant:",
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }, timeout=120)
    resp.raise_for_status()
    text = resp.json()["response"].strip()

    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    # Find JSON in response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    return json.loads(text)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_proposal(proposal: dict, current_params: dict) -> dict:
    """Validate and clamp proposed parameter changes."""
    changes = proposal.get("changes", {})
    validated = {}

    for param, value in changes.items():
        if param not in PARAM_BOUNDS:
            continue
        lo, hi = PARAM_BOUNDS[param]

        # Type coercion
        if isinstance(PARAM_BOUNDS[param][0], int):
            value = int(round(value))
        else:
            value = float(value)

        # Clamp to bounds
        value = max(lo, min(hi, value))

        # Skip if unchanged
        if abs(value - current_params.get(param, 0)) < 1e-6:
            continue

        validated[param] = value

    return validated


# ---------------------------------------------------------------------------
# Main learning loop
# ---------------------------------------------------------------------------

def run_backtest_with_params(params: dict, ticker: str = "SPY", period: str = "10y") -> dict:
    """
    Run a backtest with custom parameters.
    If ticker is "MULTI", evaluates across SPY, QQQ, DIA, IWM and returns
    a composite result (weighted average Sharpe, worst-ticker drawdown).
    """
    original_values = {}
    for key, val in params.items():
        if hasattr(FourPillarsEngine, key):
            original_values[key] = getattr(FourPillarsEngine, key)
            setattr(FourPillarsEngine, key, val)

    try:
        if ticker == "MULTI":
            tickers = ["SPY", "QQQ", "DIA", "IWM"]
            # Weights: SPY/QQQ are primary trading targets; DIA/IWM secondary
            weights = {"SPY": 0.35, "QQQ": 0.35, "DIA": 0.15, "IWM": 0.15}
            results_list = {}
            for t in tickers:
                results_list[t] = backtest_four_pillars(ticker=t, period=period, verbose=False)

            # Composite Sharpe (weighted average)
            composite_sharpe = sum(
                weights[t] * results_list[t]["sharpe_ratio"] for t in tickers
            )
            # Penalize if any single ticker Sharpe < its benchmark
            for t in tickers:
                r = results_list[t]
                if r["sharpe_ratio"] < r["benchmark_sharpe"]:
                    # Apply a 10% penalty per underperforming ticker
                    composite_sharpe *= 0.90

            # Return a composite result dict (using SPY as the "primary" for metadata)
            composite = results_list["SPY"].copy()
            composite["sharpe_ratio"] = round(composite_sharpe, 4)
            composite["_per_ticker"] = {
                t: {"sharpe": r["sharpe_ratio"], "bm_sharpe": r["benchmark_sharpe"],
                    "annual_return": r["annual_return"]}
                for t, r in results_list.items()
            }
            return composite
        else:
            return backtest_four_pillars(ticker=ticker, period=period, verbose=False)
    finally:
        for key, val in original_values.items():
            setattr(FourPillarsEngine, key, val)


def log_experiment(entry: dict):
    """Append experiment to JSONL log."""
    LEARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARN_LOG, "a") as f:
        # Filter out non-serializable items
        clean = {k: v for k, v in entry.items()
                 if not isinstance(v, (type(None.__class__),))}
        f.write(json.dumps(clean, default=str) + "\n")


def load_experiment_history() -> list[dict]:
    """Load previous experiments."""
    if not LEARN_LOG.exists():
        return []
    experiments = []
    with open(LEARN_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    experiments.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return experiments


def run_learning_loop(
    max_experiments: int = 30,
    time_limit_minutes: int = 120,
    model_backend: str = "haiku",
    ollama_model: str = "qwen2.5-coder:7b",
    ticker: str = "SPY",
    period: str = "10y",
    verbose: bool = True,
):
    """
    Main self-learning loop. Runs experiments to improve the bot's parameters.

    Args:
        max_experiments: Max number of experiments to run
        time_limit_minutes: Stop after this many minutes
        model_backend: "haiku", "sonnet", or "ollama"
        ollama_model: Which Ollama model to use
        ticker: Ticker to backtest on
        period: Historical period for backtest
        verbose: Print progress
    """
    start_time = time.time()
    current_params = load_best_params()
    experiment_history = load_experiment_history()

    # Baseline backtest
    if verbose:
        print(f"\n{'='*60}")
        print(f"  JK BOT SELF-LEARNING LOOP")
        print(f"  Model: {model_backend} | Ticker: {ticker} | Max: {max_experiments} experiments")
        print(f"{'='*60}")
        print(f"\n  Running baseline backtest...")

    baseline = run_backtest_with_params(current_params, ticker, period)
    best_sharpe = baseline["sharpe_ratio"]

    if verbose:
        print(f"  Baseline Sharpe: {best_sharpe:.4f}")
        per_ticker = baseline.get("_per_ticker", {})
        if per_ticker:
            print(f"  Per-ticker breakdown:")
            for t, data in per_ticker.items():
                beat = "BEATS" if data["sharpe"] >= data["bm_sharpe"] else "UNDERPERFORMS"
                print(f"    {t}: {data['sharpe']:.4f} vs BM {data['bm_sharpe']:.4f} ({beat})")
        print(f"  Current params: {json.dumps(current_params, indent=4)}")

    # Get trade analysis from the backtest
    trade_analysis = analyze_trade_log(baseline.get("trade_log", []))

    kept_count = 0
    for exp_num in range(1, max_experiments + 1):
        elapsed = (time.time() - start_time) / 60
        if elapsed >= time_limit_minutes:
            if verbose:
                print(f"\n  Time limit reached ({time_limit_minutes} min). Stopping.")
            break

        if verbose:
            print(f"\n  --- Experiment {exp_num}/{max_experiments} (elapsed: {elapsed:.1f}m) ---")

        # 1. Propose change
        try:
            if model_backend == "ollama":
                proposal = propose_with_ollama(
                    current_params, trade_analysis, baseline, experiment_history, ollama_model)
            else:
                proposal = propose_with_anthropic(
                    current_params, trade_analysis, baseline, experiment_history, model_backend)
        except Exception as e:
            if verbose:
                print(f"  Proposal failed: {e}")
            continue

        hypothesis = proposal.get("hypothesis", "no hypothesis")
        changes = validate_proposal(proposal, current_params)

        if not changes:
            if verbose:
                print(f"  No valid changes proposed. Skipping.")
            continue

        if verbose:
            print(f"  Hypothesis: {hypothesis[:80]}")
            print(f"  Changes: {changes}")

        # 2. Apply changes and backtest
        test_params = current_params.copy()
        test_params.update(changes)

        try:
            results = run_backtest_with_params(test_params, ticker, period)
            new_sharpe = results["sharpe_ratio"]
        except Exception as e:
            if verbose:
                print(f"  Backtest failed: {e}")
            continue

        # 3. Keep or revert
        improved = new_sharpe > best_sharpe
        if improved:
            current_params = test_params
            best_sharpe = new_sharpe
            save_best_params(current_params)
            kept_count += 1
            trade_analysis = analyze_trade_log(results.get("trade_log", []))
            baseline = results

        entry = {
            "experiment": exp_num,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "changes": changes,
            "old_sharpe": round(baseline["sharpe_ratio"] if not improved else best_sharpe - (new_sharpe - baseline["sharpe_ratio"]), 4),
            "new_sharpe": round(new_sharpe, 4),
            "kept": improved,
            "new_params": test_params if improved else None,
            "metrics": {
                "win_rate": results.get("win_rate"),
                "exposure": results.get("exposure_pct"),
                "max_drawdown": results.get("max_drawdown"),
                "annual_return": results.get("annual_return"),
            },
        }
        log_experiment(entry)
        experiment_history.append(entry)

        # Send to Discord (only kept experiments or every 10th)
        if improved or exp_num % 10 == 0:
            try:
                from technical_analysis.bot.alerts import send_discord_learning
                send_discord_learning(entry)
            except Exception:
                pass

        if verbose:
            status = "KEPT ✓" if improved else "REVERTED ✗"
            print(f"  Sharpe: {new_sharpe:.4f} (was {baseline['sharpe_ratio']:.4f}) → {status}")
            per_ticker = results.get("_per_ticker", {})
            if per_ticker:
                breakdown = " | ".join(
                    f"{t}={d['sharpe']:.3f}{'✓' if d['sharpe'] >= d['bm_sharpe'] else '✗'}"
                    for t, d in per_ticker.items()
                )
                print(f"  Per-ticker: {breakdown}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LEARNING COMPLETE")
        print(f"  Experiments: {exp_num} | Kept: {kept_count}")
        print(f"  Best Sharpe: {best_sharpe:.4f}")
        print(f"  Best params: {json.dumps(current_params, indent=4)}")
        print(f"  Log: {LEARN_LOG}")
        print(f"{'='*60}")

    return current_params, best_sharpe
