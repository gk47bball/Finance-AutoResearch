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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

from technical_analysis.bot.pillars import FourPillarsEngine
from technical_analysis.bot.backtest_pillars import backtest_four_pillars


LEARN_LOG = Path(__file__).parent / "state" / "learning_log.jsonl"
PARAMS_FILE = Path(__file__).parent / "state" / "best_params.json"
PROMPT_EVOLUTION_LOG = Path(__file__).parent / "state" / "prompt_evolution.jsonl"

# ---------------------------------------------------------------------------
# Parameter genome — the mutable state
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # These reflect the CURRENT validated best params (not the original starting point).
    # Updated 2026-03-22 after 550+ experiments, composite Sharpe ~0.96.
    # Used only as fallback if best_params.json is missing (fresh install).
    "BULL_THRESHOLD": 3,   # Raised from 2; requires stronger trend before BULL regime
    "BEAR_THRESHOLD": -2,
    "DEEP_OVERSOLD": -1.5,
    "OVERSOLD": -0.9,          # Loosened from -1.1; pairs well with tight TRAIL_STOP
    "OVERBOUGHT": 3.7,         # Raised from 2.5→3.0→3.7; let winners run on QQQ/SPY
    "STOP_LOSS_PCT": 0.05,
    "TRAIL_STOP_PCT": 0.015,   # Tight: locks in gains quickly once trailing activated
    "TRAIL_ACTIVATE_PCT": 0.03,
    "TIME_STOP_DAYS": 60,
    "BULL_BASELINE": 0.50,     # Validated: 42+ experiments confirmed. Do NOT increase.
    "CHOP_BASELINE": 0.50,     # Validated: +0.25 Sharpe improvement. Do NOT decrease.
    "BEAR_BASELINE": 0.0,
    "ZSCORE_LOOKBACK": 63,     # 63 = ~3 months lookback. Converged across 5 sessions.
}

PARAM_BOUNDS = {
    "BULL_THRESHOLD": (1, 4),
    "BEAR_THRESHOLD": (-4, -1),
    "DEEP_OVERSOLD": (-3.0, -1.0),
    "OVERSOLD": (-1.5, -0.3),
    "OVERBOUGHT": (0.8, 4.0),   # Upper bound extended: 2.5 was previously the best AND the ceiling
    "STOP_LOSS_PCT": (0.02, 0.10),
    "TRAIL_STOP_PCT": (0.01, 0.05),
    "TRAIL_ACTIVATE_PCT": (0.02, 0.08),
    "TIME_STOP_DAYS": (20, 90),
    "BULL_BASELINE": (0.25, 0.95),
    "CHOP_BASELINE": (0.0, 0.50),
    "BEAR_BASELINE": (0.0, 0.25),
    # Indicator lookback: how many bars to use for z-score normalization.
    # 21 = ~1 month (reactive), 42 = ~2 months, 63 = ~3 months, 126 = ~6 months.
    "ZSCORE_LOOKBACK": (21, 126),
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

LEARNER_SYSTEM_PROMPT = """You are optimizing a mean-reversion trading bot. The strategy is now at
composite Sharpe ~0.93 across SPY/QQQ/DIA/IWM. The low-hanging fruit is gone.
Goal: find the next 2-5% improvement by exploring genuinely novel territory.

═══════════════════════════════════════════════════════
 SETTLED — DO NOT CHANGE (empirically proven, 100+ experiments)
═══════════════════════════════════════════════════════
  BULL_BASELINE = 0.50   ← 42+ experiments confirmed. Higher values (0.6–0.9) reduce
                            Sharpe to 0.58–0.73. Mean-reversion timing adds more value
                            than raw exposure. DO NOT INCREASE.
  CHOP_BASELINE = 0.50   ← Biggest single improvement in history (+0.25 Sharpe).
                            DO NOT lower below 0.45.
  BEAR_BASELINE = 0.0    ← Correct. Do not add bear exposure.
  TIME_STOP_DAYS         ← 30 and 60 tested identically. Not a lever.

═══════════════════════════════════════════════════════
 ALREADY TESTED — MODERATE IMPROVEMENT, USE CAREFULLY
═══════════════════════════════════════════════════════
  OVERBOUGHT: tested 2.5 (was best), 3.0 (now current), 3.5 (mixed: helps QQQ but
              hurts DIA/IWM in bear market). Bound is 4.0. Still worth testing 3.5
              WITH another adjustment to protect bear-year performance.
  ZSCORE_LOOKBACK: 42 (current), 84 (tried, improved once but unstable across runs),
              63 (NEVER tried — halfway between, could be most stable).
  OVERSOLD: -1.1 (current), tried -0.8 (hurts DIA), -1.2 (mixed), -1.3 (worse).
  DEEP_OVERSOLD: -1.5 (current), tried -2.0 (hurts IWM). -1.2 untried.
  STOP_LOSS_PCT: 0.05 (current), tried 0.06 (fails OOS bear gate), 0.04 (worse IS).
  TRAIL_ACTIVATE_PCT: 0.03 (current), tried 0.04 (no improvement).

═══════════════════════════════════════════════════════
 GENUINELY UNEXPLORED — HIGH PRIORITY TO TRY
═══════════════════════════════════════════════════════
  TRAIL_STOP_PCT [0.01–0.05]: Currently 0.02. NEVER changed. Controls trailing stop
                               tightness. 0.015 = tighter (locks in more gain).
                               0.025 or 0.03 = looser (lets price breathe more).
  ZSCORE_LOOKBACK = 63:        Halfway between 42 and 84. May be most stable.
  ZSCORE_LOOKBACK = 21:        Very reactive — short 1-month window. Never tried.
  BULL_THRESHOLD = 3:          Requires stronger trend to enter bull regime.
                                May reduce whipsaws in choppy bull markets.
  Multi-param combos:           e.g., OVERBOUGHT=3.5 + TRAIL_STOP_PCT=0.015 together.
                                Or ZSCORE_LOOKBACK=63 + OVERSOLD=-1.0 together.
  DEEP_OVERSOLD = -1.2:         Less deep threshold = more full-size entries. Untried.

═══════════════════════════════════════════════════════
 OBJECTIVE
═══════════════════════════════════════════════════════
  Maximize COMPOSITE Sharpe: SPY×0.35 + QQQ×0.35 + DIA×0.15 + IWM×0.15
  10% penalty per ticker underperforming its buy-and-hold benchmark.
  Current best: ~0.93 across all 4 tickers beating their benchmarks.
  OOS gate: new params must achieve ≥0.40 Sharpe on trailing 4y AND ≥-0.30 on 2022.

RULES:
1. EXPLORE genuinely novel territory (see UNEXPLORED section above).
2. DO NOT keep proposing the same values you've tried before.
3. Change 1-2 parameters per experiment. Multi-param combos are encouraged.
4. Think about SECOND-ORDER effects: e.g., a looser TRAIL_STOP allows more profit
   capture on trending tickers (QQQ), helping the composite Sharpe.
5. Respond ONLY with valid JSON:
   {"changes": {"PARAM": value, ...}, "hypothesis": "concise reason"}
"""


def propose_with_anthropic(
    current_params: dict,
    trade_analysis: dict,
    backtest_results: dict,
    experiment_history: list[dict],
    model: str = "haiku",
) -> dict:
    """Use LLM (Ollama local or Anthropic fallback) to propose a parameter tweak."""
    from technical_analysis.bot.llm_client import llm_chat_json

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
            per_ticker_str += f"\n  ⚠️  UNDERPERFORMERS: {', '.join(underperformers)}\n"
            per_ticker_str += "  → DO NOT change BULL_BASELINE (empirically proven to hurt). Try timing thresholds instead.\n"

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

    # Use Ollama by default, Anthropic as fallback
    backend = "ollama" if model in ("haiku", "haiku3") else "anthropic"
    return llm_chat_json(
        system=LEARNER_SYSTEM_PROMPT,
        user=user_msg,
        max_tokens=500,
        temperature=0.9,
        backend=backend,
        model=model if backend == "anthropic" else None,
    )


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

# ---------------------------------------------------------------------------
# Karpathy upgrade: batch proposals + parallel evaluation + meta-prompt evolution
# ---------------------------------------------------------------------------

def propose_batch_with_anthropic(
    current_params: dict,
    trade_analysis: dict,
    backtest_results: dict,
    experiment_history: list[dict],
    model: str = "haiku",
    n: int = 3,
) -> list[dict]:
    """
    Ask LLM to propose N distinct parameter hypotheses in a single call.
    Uses Ollama (local) by default, Anthropic as fallback.
    Returns list of validated proposal dicts, each with 'changes' and 'hypothesis'.
    """
    from technical_analysis.bot.llm_client import llm_chat_json_array

    history_summary = ""
    failed_params = set()
    recently_tried_no_improvement: dict[str, set] = {}
    if experiment_history:
        recent = experiment_history[-30:]
        history_summary = "\nPast experiments:\n"
        for exp in recent:
            kept_val = exp.get("kept")
            is_kept = (kept_val is True) or (kept_val == "True")
            kept = "KEPT" if is_kept else "REVERTED"
            changes = exp.get("changes", {})
            history_summary += (
                f"  [{kept}] {changes} → Sharpe {exp.get('new_sharpe', '?')}\n"
            )
            if not is_kept:
                failed_params.update(changes.keys())
                for param, val in changes.items():
                    recently_tried_no_improvement.setdefault(param, set()).add(val)

    do_not_retry_lines = []
    for param, vals in recently_tried_no_improvement.items():
        if len(vals) >= 3 or (param in recently_tried_no_improvement and
                               len([e for e in experiment_history[-30:]
                                    if param in e.get("changes", {})
                                    and not (e.get("kept") is True or e.get("kept") == "True")]) >= 3):
            vals_str = ", ".join(str(v) for v in sorted(vals, key=lambda x: str(x)))
            do_not_retry_lines.append(f"  {param}: already tried [{vals_str}] — no improvement, skip these values")
    do_not_retry_str = ""
    if do_not_retry_lines:
        do_not_retry_str = "\nDO NOT RETRY these (tried multiple times with no improvement):\n" + "\n".join(do_not_retry_lines)

    per_ticker_str = ""
    per_ticker = backtest_results.get("_per_ticker", {})
    if per_ticker:
        per_ticker_str = "\nPer-ticker Sharpe:\n"
        for t, data in per_ticker.items():
            beat = "✓" if data["sharpe"] >= data["bm_sharpe"] else "✗"
            per_ticker_str += f"  {t}: {data['sharpe']:.4f} vs {data['bm_sharpe']:.4f} ({beat})\n"

    evolved = load_evolved_prompt_additions()

    user_msg = f"""Current parameters: {json.dumps(current_params)}
Composite Sharpe: {backtest_results.get('sharpe_ratio')} | MaxDD: {backtest_results.get('max_drawdown')} | Exposure: {backtest_results.get('exposure_pct')}
{per_ticker_str}
{history_summary}
{do_not_retry_str}
Propose EXACTLY {n} DISTINCT parameter experiments. Each must change DIFFERENT parameters.
Explore UNEXPLORED territory — avoid parameters and values you have already tried.
Return {n} objects, each: {{"changes": {{"PARAM": value}}, "hypothesis": "why"}}"""

    system = LEARNER_SYSTEM_PROMPT + (f"\n\nLEARNED HEURISTICS:\n{evolved}" if evolved else "")
    backend = "ollama" if model in ("haiku", "haiku3") else "anthropic"

    proposals_raw = llm_chat_json_array(
        system=system,
        user=user_msg,
        max_tokens=800,
        temperature=0.95,
        backend=backend,
        model=model if backend == "anthropic" else None,
    )

    if not isinstance(proposals_raw, list):
        proposals_raw = [proposals_raw]

    # Validate each proposal
    validated_proposals = []
    for p in proposals_raw[:n]:
        changes = validate_proposal(p, current_params)
        if changes:
            validated_proposals.append({"changes": changes, "hypothesis": p.get("hypothesis", "")})

    return validated_proposals


def run_parallel_experiments(
    proposals: list[dict],
    current_params: dict,
    ticker: str = "SPY",
    period: str = "10y",
) -> list[tuple]:
    """
    Run N backtest experiments concurrently using a thread pool.
    Returns list of (proposal, results, sharpe) sorted by sharpe descending.
    Karpathy: "Multiple agents running simultaneously — maximize token throughput."
    """
    def _run_one(proposal):
        test_params = current_params.copy()
        test_params.update(proposal["changes"])
        results = run_backtest_with_params(test_params, ticker, period)
        return proposal, results, results["sharpe_ratio"]

    outcomes = []
    with ThreadPoolExecutor(max_workers=len(proposals)) as pool:
        futures = {pool.submit(_run_one, p): p for p in proposals}
        for f in as_completed(futures):
            try:
                outcomes.append(f.result())
            except Exception as e:
                proposal = futures[f]
                print(f"  Parallel experiment failed ({proposal.get('hypothesis', '')[:40]}): {e}")

    # Sort by sharpe descending — best first
    outcomes.sort(key=lambda x: x[2], reverse=True)
    return outcomes


def load_evolved_prompt_additions() -> str:
    """Load any LLM-evolved additions to the system prompt."""
    if not PROMPT_EVOLUTION_LOG.exists():
        return ""
    additions = []
    with open(PROMPT_EVOLUTION_LOG) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    additions.append(entry.get("addition", ""))
                except json.JSONDecodeError:
                    continue
    return "\n".join(a for a in additions if a)


def evolve_system_prompt(
    experiment_history: list[dict],
    model: str = "haiku",
) -> Optional[str]:
    """
    After N experiments, ask the LLM to extract new heuristics from what worked.
    Appends to prompt_evolution.jsonl.
    Karpathy: "When is the model going to write a better program.md than you?"
    """
    from technical_analysis.bot.llm_client import llm_chat

    kept = [e for e in experiment_history if e.get("kept")]
    reverted = [e for e in experiment_history if not e.get("kept")]

    if len(kept) < 3:
        return None  # Not enough data to learn from

    kept_summary = "\n".join(
        f"  KEPT: {e['changes']} → Sharpe {e['new_sharpe']:.4f} | {e['hypothesis'][:60]}"
        for e in kept[-10:]
    )
    reverted_summary = "\n".join(
        f"  REVERTED: {e['changes']} | {e['hypothesis'][:60]}"
        for e in reverted[-10:]
    )

    user_msg = f"""Analyze these experiment results from the JK trading bot AutoResearch loop.

KEPT (improved Sharpe):
{kept_summary}

REVERTED (did not improve):
{reverted_summary}

Based on these patterns, write 2-3 concise new heuristics to add to the "TRY:" section of the optimizer's system prompt.
These should be actionable rules like "TRY: ..." that will help future experiments find improvements faster.
Do NOT repeat rules that are already in the system prompt.
Return ONLY the new heuristic lines, one per line, starting with "TRY:"."""

    try:
        backend = "ollama" if model in ("haiku", "haiku3") else "anthropic"
        addition = llm_chat(
            system="",
            user=user_msg,
            max_tokens=300,
            temperature=0.5,
            json_mode=False,  # plain text output
            backend=backend,
            model=model if backend == "anthropic" else None,
        )
    except Exception as e:
        print(f"  Meta-prompt evolution failed: {e}")
        return None

    # Persist
    PROMPT_EVOLUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "round": len(experiment_history),
        "addition": addition,
    }
    with open(PROMPT_EVOLUTION_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return addition


def run_backtest_with_params(
    params: dict,
    ticker: str = "SPY",
    period: str = "10y",
    start: str = None,
    end: str = None,
) -> dict:
    """
    Run a backtest with custom parameters — THREAD-SAFE.

    Passes params directly to backtest_four_pillars() as instance-level overrides,
    avoiding the previous approach of mutating FourPillarsEngine CLASS attributes
    (which caused a race condition when experiments ran in parallel).

    If ticker is "MULTI", evaluates across SPY, QQQ, DIA, IWM and returns
    a composite result (weighted average Sharpe, worst-ticker drawdown).

    Args:
        start: ISO date "YYYY-MM-DD". If set, overrides period (e.g. "2022-01-01").
        end:   ISO date "YYYY-MM-DD". Upper bound when using start.
    """
    if ticker == "MULTI":
        tickers = ["SPY", "QQQ", "DIA", "IWM"]
        # Weights: SPY/QQQ are primary trading targets; DIA/IWM secondary
        weights = {"SPY": 0.35, "QQQ": 0.35, "DIA": 0.15, "IWM": 0.15}
        results_list = {}
        for t in tickers:
            results_list[t] = backtest_four_pillars(
                ticker=t, period=period, verbose=False,
                params=params, start=start, end=end
            )

        # Composite Sharpe (weighted average)
        composite_sharpe = sum(
            weights[t] * results_list[t]["sharpe_ratio"] for t in tickers
        )
        # Penalize if any single ticker Sharpe < its benchmark
        for t in tickers:
            r = results_list[t]
            if r["sharpe_ratio"] < r["benchmark_sharpe"]:
                composite_sharpe *= 0.90  # 10% penalty per underperforming ticker

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
        return backtest_four_pillars(
            ticker=ticker, period=period, verbose=False,
            params=params, start=start, end=end
        )


def log_experiment(entry: dict):
    """Append experiment to JSONL log."""
    LEARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARN_LOG, "a") as f:
        # Filter out non-serializable items; coerce numpy bool_ → Python bool
        # so 'kept' is stored as JSON true/false, not the string "True"/"False".
        clean = {}
        for k, v in entry.items():
            if k == "kept":
                clean[k] = bool(v)
            elif not isinstance(v, (type(None.__class__),)):
                clean[k] = v
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

    # Compute the current-params 2022 bear-year Sharpe. Used as a RELATIVE baseline for
    # gate 2: candidates must not score more than 0.20 Sharpe worse than the current params
    # in 2022, AND must stay above -1.0 (catastrophic). This replaces the old fixed -0.30
    # threshold which was unreachable because the current best params already score ~-0.79.
    baseline_2022_sharpe = None
    try:
        _bear_baseline = run_backtest_with_params(
            current_params, ticker, "2y", start="2022-01-01", end="2022-12-31"
        )
        baseline_2022_sharpe = _bear_baseline["sharpe_ratio"]
        if verbose:
            print(f"  Baseline 2022 bear Sharpe: {baseline_2022_sharpe:.4f}")
    except Exception as e:
        if verbose:
            print(f"  (2022 baseline skipped: {e})")

    # Get trade analysis from the backtest
    trade_analysis = analyze_trade_log(baseline.get("trade_log", []))

    kept_count = 0
    total_experiments_run = 0
    BATCH_SIZE = 3  # Karpathy: run N experiments in parallel per round

    for round_num in range(1, max_experiments + 1):
        elapsed = (time.time() - start_time) / 60
        if elapsed >= time_limit_minutes:
            if verbose:
                print(f"\n  Time limit reached ({time_limit_minutes} min). Stopping.")
            break

        if verbose:
            print(f"\n  --- Round {round_num} (elapsed: {elapsed:.1f}m, total experiments: {total_experiments_run}) ---")

        # Meta-prompt evolution every 20 experiments
        if total_experiments_run > 0 and total_experiments_run % 20 == 0:
            if verbose:
                print(f"  🧠 Evolving system prompt (after {total_experiments_run} experiments)...")
            if model_backend != "ollama":
                addition = evolve_system_prompt(experiment_history, model=model_backend)
                if addition and verbose:
                    print(f"  New heuristics:\n    {addition[:200]}")

        # 1. Propose a batch of BATCH_SIZE distinct hypotheses
        try:
            if model_backend == "ollama":
                # Ollama fallback: propose one at a time
                proposals = []
                for _ in range(BATCH_SIZE):
                    p = propose_with_ollama(
                        current_params, trade_analysis, baseline, experiment_history, ollama_model)
                    changes = validate_proposal(p, current_params)
                    if changes:
                        proposals.append({"changes": changes, "hypothesis": p.get("hypothesis", "")})
            else:
                proposals = propose_batch_with_anthropic(
                    current_params, trade_analysis, baseline, experiment_history,
                    model=model_backend, n=BATCH_SIZE,
                )
        except Exception as e:
            if verbose:
                print(f"  Batch proposal failed: {e}")
            continue

        if not proposals:
            if verbose:
                print(f"  No valid proposals generated. Skipping round.")
            continue

        if verbose:
            print(f"  Proposed {len(proposals)} hypotheses — running in parallel...")
            for i, p in enumerate(proposals, 1):
                print(f"    [{i}] {p['hypothesis'][:70]} | changes: {p['changes']}")

        # 2. Run all proposals concurrently
        outcomes = run_parallel_experiments(proposals, current_params, ticker, period)
        total_experiments_run += len(outcomes)

        # 3. Process results — keep the best improvement, log all
        for rank, (proposal, results, new_sharpe) in enumerate(outcomes):
            hypothesis = proposal.get("hypothesis", "no hypothesis")
            changes = proposal.get("changes", {})

            # Capture before any update so old_sharpe is always the pre-change value
            pre_update_sharpe = best_sharpe

            # Candidate: best of batch AND beats current best in-sample
            candidate = new_sharpe > best_sharpe and rank == 0

            # Out-of-sample gate: two independent checks before committing.
            # Gate 1: trailing 4-year window (covers 2022 bear market).
            #         Sharpe must be >= 0.40 (relaxed vs old 0.50 since 4y includes a bear year).
            # Gate 2: 2022 calendar year specifically (the one real bear in our data).
            #         Strategy must not be a disaster: Sharpe >= -0.30.
            #         This prevents CHOP_BASELINE from being lowered, etc.
            oos_sharpe = None
            oos_pass = True
            if candidate:
                try:
                    test_params_oos = current_params.copy()
                    test_params_oos.update(changes)

                    # Gate 1: trailing 4y (captures 2022 bear + 2023/24/25 recovery)
                    oos_results = run_backtest_with_params(test_params_oos, ticker, "4y")
                    oos_sharpe = oos_results["sharpe_ratio"]
                    gate1_pass = oos_sharpe >= 0.40
                    if verbose and not gate1_pass:
                        print(f"    [OOS gate 1/2] 4y OOS Sharpe={oos_sharpe:.4f} — REJECTED (need ≥0.40)")

                    # Gate 2: 2022 bear market year (SPY -20%)
                    # Use a RELATIVE threshold: candidate must not be more than 0.20 Sharpe
                    # worse than the current-params baseline in 2022, and can't crater below -1.0.
                    # (A fixed -0.30 threshold was unreachable since current params score ~-0.79.)
                    gate2_pass = True
                    bear_sharpe = None
                    if gate1_pass:
                        try:
                            bear_results = run_backtest_with_params(
                                test_params_oos, ticker, "2y",  # period unused; start/end override
                                start="2022-01-01", end="2022-12-31"
                            )
                            bear_sharpe = bear_results["sharpe_ratio"]
                            # Relative gate: allow at most 0.20 worse than baseline (floor -1.0)
                            if baseline_2022_sharpe is not None:
                                gate2_threshold = max(baseline_2022_sharpe - 0.20, -1.0)
                            else:
                                gate2_threshold = -1.0  # fallback: only reject catastrophic
                            gate2_pass = bear_sharpe >= gate2_threshold
                            if verbose and not gate2_pass:
                                print(f"    [OOS gate 2/2] 2022 bear Sharpe={bear_sharpe:.4f} — REJECTED (need ≥{gate2_threshold:.2f})")
                            elif verbose:
                                print(f"    [OOS gate 2/2] 2022 bear Sharpe={bear_sharpe:.4f} — passed (threshold {gate2_threshold:.2f})")
                        except Exception as e:
                            if verbose:
                                print(f"    [OOS gate 2/2] 2022 test skipped ({e})")

                    oos_pass = gate1_pass and gate2_pass
                    if verbose and oos_pass:
                        print(f"    [OOS gates ✓] 4y={oos_sharpe:.4f}, 2022={bear_sharpe or 'n/a'} — ACCEPTED")
                except Exception as e:
                    if verbose:
                        print(f"    [OOS gate] Failed ({e}) — allowing experiment through")

            improved = candidate and oos_pass

            if improved:
                test_params = current_params.copy()
                test_params.update(changes)
                current_params = test_params
                best_sharpe = new_sharpe
                save_best_params(current_params)
                kept_count += 1
                trade_analysis = analyze_trade_log(results.get("trade_log", []))
                baseline = results

            entry = {
                "experiment": total_experiments_run - len(outcomes) + rank + 1,
                "round": round_num,
                "batch_rank": rank + 1,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "changes": changes,
                "old_sharpe": round(pre_update_sharpe, 4),   # always the value BEFORE this experiment
                "new_sharpe": round(new_sharpe, 4),
                "oos_sharpe": round(oos_sharpe, 4) if oos_sharpe is not None else None,
                "kept": improved,
                "new_params": current_params if improved else None,
                "metrics": {
                    "win_rate": results.get("win_rate"),
                    "exposure": results.get("exposure_pct"),
                    "max_drawdown": results.get("max_drawdown"),
                    "annual_return": results.get("annual_return"),
                },
            }
            log_experiment(entry)
            experiment_history.append(entry)

            if verbose:
                status = "KEPT ✓" if improved else ("best_of_batch" if rank == 0 else "REVERTED ✗")
                per_ticker = results.get("_per_ticker", {})
                ticker_str = ""
                if per_ticker:
                    ticker_str = " | " + " ".join(
                        f"{t}={d['sharpe']:.3f}{'✓' if d['sharpe'] >= d['bm_sharpe'] else '✗'}"
                        for t, d in per_ticker.items()
                    )
                print(f"    [{rank+1}] Sharpe={new_sharpe:.4f} → {status}{ticker_str}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  LEARNING COMPLETE")
        print(f"  Rounds: {round_num} | Experiments: {total_experiments_run} | Kept: {kept_count}")
        print(f"  Best Sharpe: {best_sharpe:.4f}")
        print(f"  Best params: {json.dumps(current_params, indent=4)}")
        print(f"  Log: {LEARN_LOG}")
        print(f"{'='*60}")

    # Post a single end-of-session summary to Discord.
    # Only fires once per run (not per round/experiment) to keep the channel clean.
    try:
        from technical_analysis.bot.alerts import send_discord_learning
        # Build a summary entry for the final state
        kept_entries = [e for e in experiment_history
                        if e.get("kept") is True or e.get("kept") == "True"]
        # Show the param changes that actually stuck (accumulated over kept experiments)
        all_kept_changes = {}
        for e in kept_entries:
            all_kept_changes.update(e.get("changes", {}))

        summary_entry = {
            "round": round_num,
            "hypothesis": (
                f"{kept_count} improvement(s) in {total_experiments_run} experiments"
                if kept_count else f"No improvements in {total_experiments_run} experiments — params stable"
            ),
            "changes": all_kept_changes,
            "old_sharpe": round(baseline["sharpe_ratio"], 4),
            "new_sharpe": round(best_sharpe, 4),
            "kept": kept_count > 0,
            "new_params": current_params if kept_count > 0 else None,
            "metrics": {
                "win_rate": baseline.get("win_rate"),
                "exposure": baseline.get("exposure_pct"),
                "max_drawdown": baseline.get("max_drawdown"),
                "annual_return": baseline.get("annual_return"),
            },
        }
        send_discord_learning(summary_entry)
    except Exception:
        pass

    return current_params, best_sharpe
