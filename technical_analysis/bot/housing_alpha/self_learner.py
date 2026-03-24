"""
Housing Alpha AutoResearch Loop
=================================
LLM-in-the-loop optimization for housing indicator parameters.

Same architecture as the Four Pillars self_learner:
  1. Run baseline backtest with current params
  2. Ask LLM to propose parameter tweaks
  3. Backtest each tweak
  4. Keep improvements, revert failures
  5. Log everything to JSONL

The "genome" here is different — it's about:
  - Indicator weights (which sub-signals matter most)
  - Z-score lookback windows
  - Regime thresholds
  - Position sizing per regime
  - Rate override sensitivity
"""

import json
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=True)

from technical_analysis.bot.housing_alpha.engine import (
    HousingAlphaEngine,
    DEFAULT_PARAMS,
    load_params,
    save_params,
)
from technical_analysis.bot.housing_alpha.backtest import (
    backtest_housing_alpha,
    backtest_housing_multi,
)
from technical_analysis.bot.llm_client import llm_chat_json_array


LEARN_LOG = Path(__file__).parent / "state" / "learning_log.jsonl"

# Parameter bounds for the LLM
PARAM_BOUNDS = {
    # Composite weights (must sum to ~1.0)
    "weight_activity": (0.10, 0.50),
    "weight_affordability": (0.05, 0.40),
    "weight_supply_demand": (0.05, 0.30),
    "weight_price_momentum": (0.05, 0.30),
    "weight_rate_regime": (0.05, 0.40),

    # Lookback windows (months)
    "activity_mom_window": (2, 12),
    "activity_zscore_window": (12, 60),
    "afford_zscore_window": (12, 60),
    "supply_zscore_window": (12, 60),
    "price_mom_window": (3, 12),
    "price_zscore_window": (12, 60),
    "rate_lookback": (3, 12),
    "rate_zscore_window": (12, 60),
    "composite_zscore_window": (12, 60),

    # Regime thresholds
    "bull_threshold": (0.0, 1.5),
    "bear_threshold": (-1.5, 0.0),

    # Position sizing
    "bull_position": (0.50, 1.0),
    "neutral_position": (0.20, 0.60),
    "bear_position": (0.0, 0.30),

    # Rate override
    "rate_override_threshold": (0.5, 3.0),
    "rate_override_reduction": (0.20, 0.80),
}

LEARNER_SYSTEM_PROMPT = """You are optimizing a housing market trading strategy that trades homebuilder ETFs (XHB, ITB) based on FRED and Zillow housing data.

The strategy combines 5 sub-indicators into a composite signal:
1. Activity Momentum (housing starts, permits, sales — momentum and z-scores)
2. Affordability Index (mortgage rates, home prices — inverted: high = bearish)
3. Supply/Demand Balance (months supply, inventory — inverted: high = bearish)
4. Price Momentum (Case-Shiller, Zillow HVI acceleration)
5. Rate Regime (mortgage rate changes, yield curve — falling rates = bullish)

The composite signal determines a regime (HOUSING_BULL, NEUTRAL, BEAR) which sets position sizing.
There's also a rate override that reduces positions when rates are rising sharply.

IMPORTANT CONSTRAINTS:
- Housing data is monthly. Don't try to over-optimize for noise.
- Signals are lagged 1 month (realistic — you can't trade on data the same month).
- Weight parameters should roughly sum to 1.0 (they get normalized, but keep them reasonable).
- Z-score lookback windows: 12-60 months. Too short = noisy, too long = slow to react.
- The biggest risk is overfitting to the 2020-2022 housing boom/bust cycle.

You will receive the current parameters and backtest results. Propose exactly 3 parameter changes.
Each change should modify 1-2 parameters with a clear hypothesis.

Return a JSON array of objects, each with:
- "hypothesis": brief explanation of why this change should help
- "changes": dict of parameter name → new value
"""


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _log_experiment(entry: dict):
    """Append experiment to JSONL log."""
    LEARN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LEARN_LOG, "a") as f:
        f.write(json.dumps(entry, cls=_NumpyEncoder) + "\n")


def _run_backtest_with_params(params: dict, tickers: list[str]) -> dict:
    """Thread-safe backtest with specific params."""
    return backtest_housing_multi(
        tickers=tickers,
        params=params,
        verbose=False,
    )


def run_learning_loop(
    max_experiments: int = 30,
    time_limit_minutes: int = 120,
    model_backend: str = "ollama",
    model: str = "qwen3:4b",
    tickers: Optional[list[str]] = None,
) -> tuple[dict, float]:
    """
    Run the AutoResearch loop for housing alpha.

    Returns:
        (best_params, best_sharpe)
    """
    if tickers is None:
        tickers = ["XHB", "ITB"]

    start_time = time.time()
    current_params = load_params()
    total_experiments = 0
    round_num = 0

    # Baseline
    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  Housing Alpha AutoResearch")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Max experiments: {max_experiments}")
    print(f"  Time limit: {time_limit_minutes} min")
    print(f"  ═══════════════════════════════════════════════\n")

    print("  Running baseline backtest...")
    baseline_results = backtest_housing_multi(
        tickers=tickers,
        params=current_params,
        verbose=True,
    )
    best_sharpe = baseline_results.get("composite_sharpe", 0)
    print(f"\n  Baseline composite Sharpe: {best_sharpe:+.4f}\n")

    while total_experiments < max_experiments:
        elapsed = (time.time() - start_time) / 60
        if elapsed >= time_limit_minutes:
            print(f"\n  Time limit reached ({elapsed:.1f}m)")
            break

        round_num += 1

        # Build context for LLM
        per_ticker_summary = ""
        for t in tickers:
            r = baseline_results.get(t, {})
            per_ticker_summary += (
                f"  {t}: Sharpe={r.get('sharpe_ratio', 0):+.3f} "
                f"(BM: {r.get('benchmark_sharpe', 0):+.3f}) "
                f"{'✓' if r.get('beats_benchmark') else '✗'}\n"
            )

        user_prompt = f"""Current parameters:
{json.dumps(current_params, indent=2)}

Parameter bounds:
{json.dumps(PARAM_BOUNDS, indent=2)}

Current backtest results:
  Composite Sharpe: {best_sharpe:+.4f}
{per_ticker_summary}

Experiment history: {total_experiments} experiments so far in this session.
Round: {round_num}

Propose exactly 3 parameter changes. Each should modify 1-2 parameters.
Focus on the weakest ticker or the most promising unexplored parameter range.
Return a JSON array of 3 objects with "hypothesis" and "changes" keys."""

        # Ask LLM for proposals
        try:
            proposals = llm_chat_json_array(
                system=LEARNER_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=1200,
                temperature=0.7,
                backend=model_backend,
                model=model,
            )
        except Exception as e:
            print(f"  [learn] LLM call failed: {e}")
            total_experiments += 1
            continue

        if not proposals:
            print(f"  [learn] No proposals returned")
            total_experiments += 1
            continue

        # Limit to 3
        proposals = proposals[:3]

        print(f"\n  --- Round {round_num} (elapsed: {elapsed:.1f}m, total experiments: {total_experiments}) ---")
        print(f"  Proposed {len(proposals)} hypotheses — running in parallel...")

        # Run all proposals in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            for i, prop in enumerate(proposals):
                changes = prop.get("changes", {})
                hypothesis = prop.get("hypothesis", "")[:80]

                # Apply changes to current params
                candidate = copy.deepcopy(current_params)
                for k, v in changes.items():
                    if k in PARAM_BOUNDS:
                        lo, hi = PARAM_BOUNDS[k]
                        v = max(lo, min(hi, v))
                    candidate[k] = v

                print(f"    [{i+1}] {hypothesis} | changes: {changes}")
                futures[pool.submit(_run_backtest_with_params, candidate, tickers)] = (i, prop, candidate)

            # Collect results
            batch_results = []
            for future in as_completed(futures):
                idx, prop, candidate = futures[future]
                try:
                    result = future.result()
                    batch_results.append((idx, prop, candidate, result))
                except Exception as e:
                    print(f"    [{idx+1}] Backtest failed: {e}")
                    total_experiments += 1

        # Find best in batch
        best_in_batch = None
        best_batch_sharpe = best_sharpe

        for idx, prop, candidate, result in sorted(batch_results, key=lambda x: x[0]):
            total_experiments += 1
            candidate_sharpe = result.get("composite_sharpe", 0)
            changes = prop.get("changes", {})

            # Per-ticker breakdown
            ticker_str = ""
            for t in tickers:
                r = result.get(t, {})
                flag = "✓" if r.get("beats_benchmark") else "✗"
                ticker_str += f" {t}={r.get('sharpe_ratio', 0):.3f}{flag}"

            if candidate_sharpe > best_batch_sharpe:
                best_batch_sharpe = candidate_sharpe
                best_in_batch = (candidate, result)
                print(f"    [{idx+1}] Sharpe={candidate_sharpe:+.4f} → best_of_batch |{ticker_str}")
            else:
                print(f"    [{idx+1}] Sharpe={candidate_sharpe:+.4f} → REVERTED ✗ |{ticker_str}")

            # Log
            _log_experiment({
                "timestamp": datetime.now().isoformat(),
                "round": round_num,
                "experiment": total_experiments,
                "hypothesis": prop.get("hypothesis", ""),
                "changes": changes,
                "candidate_sharpe": candidate_sharpe,
                "baseline_sharpe": best_sharpe,
                "kept": candidate_sharpe > best_sharpe,
                "per_ticker": {t: result.get(t, {}).get("sharpe_ratio", 0) for t in tickers},
            })

        # Keep best if improved
        if best_in_batch and best_batch_sharpe > best_sharpe:
            candidate, result = best_in_batch
            print(f"\n  ✅ IMPROVEMENT: {best_sharpe:+.4f} → {best_batch_sharpe:+.4f}")
            best_sharpe = best_batch_sharpe
            current_params = candidate
            baseline_results = result
            save_params(current_params)

    print(f"\n  ═══════════════════════════════════════════════")
    print(f"  AutoResearch complete: {total_experiments} experiments in {(time.time() - start_time) / 60:.1f}m")
    print(f"  Best composite Sharpe: {best_sharpe:+.4f}")
    print(f"  ═══════════════════════════════════════════════\n")

    return current_params, best_sharpe
