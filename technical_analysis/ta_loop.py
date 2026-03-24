"""
TA AutoResearch Loop — Lightweight Karpathy-Style Optimization
================================================================
Uses Haiku (or Ollama) as a cheap, fast hypothesis engine to continuously
test parameter changes on the TA strategy. Designed to run for hours
unattended with minimal cost.

Key differences from main loop:
- Uses Haiku by default (~100x cheaper than Opus, fast enough for this)
- Ollama fallback for fully local/free runs
- TA-specific validation (indicator weights, thresholds, params)
- Simpler mutations (small model = simpler instructions = fewer hallucinations)

Usage:
    python -m technical_analysis.ta_loop                    # Haiku, 50 experiments
    python -m technical_analysis.ta_loop --model ollama     # Local Ollama
    python -m technical_analysis.ta_loop -n 200 --time 180  # 200 experiments, 3hr limit
    python -m technical_analysis.ta_loop --ticker DIA       # Optimize for DIA
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

import ast
import time
import json
import shutil
import subprocess
import importlib
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "strategy_ta.py")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
LOG_PATH = os.path.join(EXPERIMENTS_DIR, "ta_experiments.jsonl")

# Minimum improvement to keep a change (avoids noise)
MIN_IMPROVEMENT = 0.005

# ---------------------------------------------------------------------------
# TA Strategy Validator
# ---------------------------------------------------------------------------

def validate_ta_strategy(strategy_text: str) -> tuple[bool, str]:
    """Validate a TA strategy file for correctness."""
    # 1. Syntax check
    try:
        ast.parse(strategy_text)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    # 2. Execution check
    try:
        namespace = {}
        exec(strategy_text, namespace)
    except Exception as e:
        return False, f"Execution error: {e}"

    # 3. Required variables
    required = ["INDICATORS", "SIGNAL_RULES", "TRADING", "EVALUATION", "UNIVERSE"]
    for var in required:
        if var not in namespace:
            return False, f"Missing required variable: {var}"

    # 4. Indicator sanity
    indicators = namespace.get("INDICATORS", {})
    enabled_weights = []
    for name, cfg in indicators.items():
        if cfg.get("enabled", False) and cfg.get("weight", 0) > 0:
            w = cfg["weight"]
            if w < 0 or w > 1:
                return False, f"Indicator {name} weight {w} out of [0, 1]"
            enabled_weights.append(w)
            # Must have params and signal_col
            if "signal_col" not in cfg:
                return False, f"Indicator {name} missing signal_col"

    if len(enabled_weights) < 1:
        return False, "No indicators enabled"

    total = sum(enabled_weights)
    if total < 0.5 or total > 2.0:
        return False, f"Total weight {total:.2f} out of reasonable range [0.5, 2.0]"

    # 5. Trading sanity
    trading = namespace.get("TRADING", {})
    thresh = trading.get("long_threshold", 0)
    if thresh < -5 or thresh > 5:
        return False, f"long_threshold {thresh} out of range [-5, 5]"

    # 6. Signal rules
    rules = namespace.get("SIGNAL_RULES", {})
    lb = rules.get("lookback_for_zscore", 63)
    if lb < 10 or lb > 252:
        return False, f"lookback_for_zscore {lb} out of range [10, 252]"

    return True, "OK"


def extract_hypothesis(strategy_text: str) -> str:
    """Extract hypothesis from docstring."""
    try:
        tree = ast.parse(strategy_text)
        docstring = ast.get_docstring(tree)
        if docstring:
            for line in docstring.split("\n"):
                line = line.strip()
                if line.lower().startswith("hypothesis:") or line.lower().startswith("experiment:"):
                    return line.split(":", 1)[1].strip()
        return "Unknown"
    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# LLM Proposer — supports Haiku, Sonnet, Ollama
# ---------------------------------------------------------------------------

TA_SYSTEM_PROMPT = """You are a technical analysis strategy optimizer. You modify a Python strategy
configuration file to improve its Sharpe ratio on SPY.

The strategy uses 9+ JK technical indicators (MultiMAC, Z-Factor, VE-RSI, etc.) combined via
weighted average of z-score normalized signals. A binary position system goes long when the
combined signal exceeds long_threshold.

KEY FINDINGS FROM SCENARIO ANALYSIS:
- All indicators have negative IC (contrarian/mean-reversion signals)
- Strategy works best on SPY and DIA (broad indices)
- Optimal holding period: 15-40 days
- Signal extremes (bottom 2 deciles) capture most alpha
- Bull_volatile regime has 2-3x stronger IC than sideways

PARAMETERS YOU CAN TUNE:
1. Indicator weights (how much each indicator contributes)
2. long_threshold (when to enter: lower = more in market)
3. lookback_for_zscore (normalization window: 21-252)
4. Indicator params (MA lengths, RSI periods, etc.)
5. Enable/disable indicators
6. holding_period_min (1-30)
7. flip_signal (True for contrarian, False for trend-following)

RULES:
- Make ONE small, focused change per experiment
- Keep indicator names and signal_col exactly as-is (they map to Python functions)
- Weights should be positive and reasonable (0.01-0.50 each)
- Update the docstring with your hypothesis (start with "Hypothesis: EXP-TAxx —")
- Return ONLY the complete Python file, no explanation or markdown
- Do NOT invent new indicator names — only use existing ones from the file
- CRITICAL: Study the RECENT EXPERIMENTS history. NEVER repeat a change that was already tried and reverted (✗).
  If "increase lookback" was tried and failed, do NOT try it again. Try something DIFFERENT.
- Be creative: try different weight distributions, threshold changes, enabling/disabling indicators,
  changing MA lengths, RSI periods, or other indicator-specific params."""


def propose_with_anthropic(strategy_text: str, history: list, metrics: dict,
                           model: str = "claude-haiku-4-5-20250514") -> str:
    """Propose a change using Anthropic API (Haiku by default)."""
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    if api_key.startswith("sk-ant-oat"):
        client = Anthropic(auth_token=api_key)
    else:
        client = Anthropic(api_key=api_key)

    # Build history summary (compact for small models)
    history_text = ""
    if history:
        # Show failed experiments prominently so model avoids repeating them
        failed = [e for e in history if not e.get("kept")]
        kept_exps = [e for e in history if e.get("kept")]

        history_text = "\nRECENT EXPERIMENTS (✗ = reverted/failed, ✓ = kept):\n"
        for exp in history[-15:]:
            marker = "✓ KEPT" if exp.get("kept") else "✗ REVERTED"
            sharpe = exp.get("sharpe", "?")
            if isinstance(sharpe, float):
                sharpe = f"{sharpe:.4f}"
            history_text += f"  {marker} Sharpe={sharpe} — {exp.get('hypothesis', '?')}\n"

        if failed:
            history_text += "\nDO NOT REPEAT these failed approaches:\n"
            # Deduplicate by hypothesis prefix (first 40 chars)
            seen = set()
            for exp in failed[-20:]:
                h = exp.get("hypothesis", "")[:40]
                if h not in seen:
                    seen.add(h)
                    history_text += f"  - {h}\n"

    metrics_text = ""
    if metrics:
        metrics_text = f"\nCURRENT BEST: Sharpe={metrics.get('sharpe_ratio', 0):.4f}"
        metrics_text += f", Return={metrics.get('annual_return', 0):.1%}"
        metrics_text += f", MaxDD={metrics.get('max_drawdown', 0):.1%}"
        metrics_text += f", Exposure={metrics.get('exposure_pct', 0):.0%}"

    prompt = f"""Current strategy_ta.py:

```python
{strategy_text}
```
{history_text}{metrics_text}

Propose ONE focused change to improve the Sharpe ratio. Return the COMPLETE modified
strategy_ta.py file. Update the docstring hypothesis. Return ONLY Python code."""

    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=TA_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    return text


def propose_with_ollama(strategy_text: str, history: list, metrics: dict,
                        model: str = "llama3.2") -> str:
    """Propose a change using local Ollama."""
    import requests

    history_text = ""
    if history:
        history_text = "\nRECENT EXPERIMENTS:\n"
        for exp in history[-5:]:  # Shorter for local models
            kept = "✓" if exp.get("kept") else "✗"
            sharpe = exp.get("sharpe", "?")
            if isinstance(sharpe, float):
                sharpe = f"{sharpe:.4f}"
            history_text += f"  {kept} Sharpe={sharpe} — {exp.get('hypothesis', '?')}\n"

    metrics_text = ""
    if metrics:
        metrics_text = f"\nCURRENT BEST: Sharpe={metrics.get('sharpe_ratio', 0):.4f}"

    prompt = f"""{TA_SYSTEM_PROMPT}

Current strategy_ta.py:

```python
{strategy_text}
```
{history_text}{metrics_text}

Propose ONE focused change to improve the Sharpe ratio. Return the COMPLETE modified
strategy_ta.py file. Update the docstring hypothesis. Return ONLY Python code."""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 4000, "temperature": 0.7},
            },
            timeout=120,
        )
        response.raise_for_status()
        text = response.json()["response"].strip()

        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        return text
    except Exception as e:
        raise RuntimeError(f"Ollama failed: {e}")


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------

def run_ta_backtest(ticker: str = None) -> dict:
    """Run the TA backtest and return metrics dict."""
    # Force reimport of strategy module
    mod_name = "technical_analysis.strategy_ta"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location("strategy_ta", STRATEGY_PATH)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)

    if ticker:
        strategy.UNIVERSE["tickers"] = [ticker]
        strategy.EVALUATION["benchmark"] = ticker

    from technical_analysis.backtest.ta_backtest import run_strategy_backtest
    result = run_strategy_backtest(strategy, verbose=False)
    return result.metrics


# ---------------------------------------------------------------------------
# Experiment Logger
# ---------------------------------------------------------------------------

def _load_log() -> list:
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return [json.loads(line) for line in f if line.strip()]
    return []


def _append_log(entry: dict):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Git Operations
# ---------------------------------------------------------------------------

def _git_commit(message: str):
    """Commit the strategy file."""
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        subprocess.run(["git", "add", STRATEGY_PATH], cwd=base,
                       capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", message], cwd=base,
                       capture_output=True, check=True)
    except subprocess.CalledProcessError:
        pass  # Non-fatal


def _git_revert():
    """Revert strategy file to last commit."""
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        subprocess.run(["git", "checkout", "--", STRATEGY_PATH], cwd=base,
                       capture_output=True, check=True)
    except subprocess.CalledProcessError:
        pass


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def run_ta_loop(
    max_experiments: int = 50,
    time_limit_minutes: int = 180,
    model_backend: str = "haiku",       # "haiku", "sonnet", "ollama"
    ollama_model: str = "llama3.2",
    ticker: str = None,
    verbose: bool = True,
):
    """
    Run the TA AutoResearch loop.

    Args:
        max_experiments: Maximum number of experiments to run
        time_limit_minutes: Time limit in minutes
        model_backend: "haiku" (default, cheap), "sonnet", or "ollama" (free/local)
        ollama_model: Which Ollama model to use if backend is "ollama"
        ticker: Override ticker (default uses strategy's UNIVERSE)
        verbose: Print progress
    """
    start_time = time.time()
    history = _load_log()
    exp_id = len(history) + 1

    # Model selection
    if model_backend == "haiku":
        model_name = os.environ.get("TA_HAIKU_MODEL", "claude-3-haiku-20240307")
        propose_fn = lambda s, h, m: propose_with_anthropic(s, h, m, model_name)
        cost_label = "~$0.001/experiment"
    elif model_backend == "sonnet":
        model_name = os.environ.get("TA_SONNET_MODEL", "claude-3-5-sonnet-20241022")
        propose_fn = lambda s, h, m: propose_with_anthropic(s, h, m, model_name)
        cost_label = "~$0.01/experiment"
    elif model_backend == "ollama":
        propose_fn = lambda s, h, m: propose_with_ollama(s, h, m, ollama_model)
        cost_label = "FREE (local)"
        model_name = f"ollama/{ollama_model}"
    else:
        raise ValueError(f"Unknown model backend: {model_backend}")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TA AutoResearch Loop — Karpathy-Style Optimization")
        print(f"  Model: {model_name} ({cost_label})")
        print(f"  Max experiments: {max_experiments}, Time limit: {time_limit_minutes}m")
        print(f"  Ticker: {ticker or 'SPY (default)'}")
        print(f"  Strategy: {STRATEGY_PATH}")
        print(f"{'='*70}\n")

    # --- Baseline ---
    if verbose:
        print("  [0] Running baseline backtest...")

    try:
        baseline_metrics = run_ta_backtest(ticker)
        best_sharpe = baseline_metrics["sharpe_ratio"]
    except Exception as e:
        print(f"  ERROR: Baseline backtest failed: {e}")
        return

    if verbose:
        print(f"  Baseline Sharpe: {best_sharpe:.4f}")
        print(f"  Return: {baseline_metrics['annual_return']:.1%}, "
              f"MaxDD: {baseline_metrics['max_drawdown']:.1%}, "
              f"Exposure: {baseline_metrics['exposure_pct']:.0%}")
        print()

    # Save baseline strategy
    with open(STRATEGY_PATH) as f:
        best_strategy = f.read()

    kept_count = 0
    failed_count = 0

    # --- Main Loop ---
    for i in range(max_experiments):
        elapsed = (time.time() - start_time) / 60
        if elapsed > time_limit_minutes:
            if verbose:
                print(f"\n  Time limit reached ({time_limit_minutes}m). Stopping.")
            break

        current_exp_id = exp_id + i
        if verbose:
            print(f"  [{current_exp_id}] Proposing change... ", end="", flush=True)

        # Read current strategy
        with open(STRATEGY_PATH) as f:
            current_strategy = f.read()

        # --- Propose ---
        try:
            proposed = propose_fn(current_strategy, history, baseline_metrics)
        except Exception as e:
            error_str = str(e)
            if verbose:
                print(f"LLM ERROR: {error_str[:100]}")
            failed_count += 1
            # Stop immediately on auth errors (token expired)
            if "authentication_error" in error_str or "401" in error_str:
                if verbose:
                    print(f"\n  ⚠ Authentication failed — API token may have expired. Stopping.")
                break
            continue

        # --- Validate ---
        valid, reason = validate_ta_strategy(proposed)
        if not valid:
            if verbose:
                print(f"INVALID: {reason}")
            failed_count += 1
            _append_log({
                "experiment_id": current_exp_id,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": extract_hypothesis(proposed) if proposed else "parse_fail",
                "sharpe": None,
                "kept": False,
                "reason": f"validation: {reason}",
                "model": model_name,
            })
            history.append({"experiment_id": current_exp_id, "kept": False,
                           "hypothesis": f"INVALID: {reason}", "sharpe": None})
            continue

        hypothesis = extract_hypothesis(proposed)
        if verbose:
            print(f"{hypothesis[:60]}... ", end="", flush=True)

        # --- Apply and test ---
        with open(STRATEGY_PATH, "w") as f:
            f.write(proposed)

        try:
            new_metrics = run_ta_backtest(ticker)
            new_sharpe = new_metrics["sharpe_ratio"]
        except Exception as e:
            if verbose:
                print(f"BACKTEST ERROR: {e}")
            # Revert
            with open(STRATEGY_PATH, "w") as f:
                f.write(current_strategy)
            failed_count += 1
            _append_log({
                "experiment_id": current_exp_id,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "sharpe": None,
                "kept": False,
                "reason": f"backtest_error: {e}",
                "model": model_name,
            })
            history.append({"experiment_id": current_exp_id, "kept": False,
                           "hypothesis": hypothesis, "sharpe": None})
            continue

        improvement = new_sharpe - best_sharpe

        # --- Keep or Revert ---
        if improvement > MIN_IMPROVEMENT:
            kept = True
            best_sharpe = new_sharpe
            best_strategy = proposed
            baseline_metrics = new_metrics
            _git_commit(f"TA EXP-{current_exp_id}: {hypothesis[:60]} (Sharpe {new_sharpe:.4f})")
            kept_count += 1
            if verbose:
                print(f"✓ KEPT  Sharpe {new_sharpe:.4f} (+{improvement:.4f})")
        else:
            kept = False
            with open(STRATEGY_PATH, "w") as f:
                f.write(current_strategy)
            if verbose:
                sign = "+" if improvement >= 0 else ""
                print(f"✗ REVERT Sharpe {new_sharpe:.4f} ({sign}{improvement:.4f})")

        # Log
        entry = {
            "experiment_id": current_exp_id,
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "sharpe": new_sharpe,
            "annual_return": new_metrics.get("annual_return"),
            "max_drawdown": new_metrics.get("max_drawdown"),
            "exposure": new_metrics.get("exposure_pct"),
            "n_trades": new_metrics.get("n_trades"),
            "kept": kept,
            "improvement": improvement,
            "model": model_name,
        }
        _append_log(entry)
        history.append(entry)

    # --- Summary ---
    total_time = (time.time() - start_time) / 60
    total_run = len(history) - (exp_id - 1)

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TA AutoResearch Loop Complete")
        print(f"  Experiments: {total_run} ({kept_count} kept, {failed_count} failed)")
        print(f"  Time: {total_time:.1f} minutes")
        print(f"  Best Sharpe: {best_sharpe:.4f}")
        print(f"  Model: {model_name}")
        print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TA AutoResearch Loop")
    parser.add_argument("-n", "--experiments", type=int, default=50,
                        help="Max experiments (default: 50)")
    parser.add_argument("--time", type=int, default=180,
                        help="Time limit in minutes (default: 180)")
    parser.add_argument("--model", choices=["haiku", "sonnet", "ollama"],
                        default="haiku", help="Model backend (default: haiku)")
    parser.add_argument("--ollama-model", default="llama3.2",
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--ticker", default=None,
                        help="Override ticker (default: from strategy)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    run_ta_loop(
        max_experiments=args.experiments,
        time_limit_minutes=args.time,
        model_backend=args.model,
        ollama_model=args.ollama_model,
        ticker=args.ticker,
        verbose=not args.quiet,
    )
