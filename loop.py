"""
The AutoResearch Optimization Loop — the heart of the Karpathy pattern.

This loop autonomously iterates on strategy.py to maximize Sharpe ratio:
  1. Establish baseline metrics
  2. Ask Claude to propose a change to strategy.py
  3. Validate and apply the change (git commit)
  4. Run backtest, evaluate Sharpe
  5. If improved → KEEP; else → REVERT
  6. Log experiment, repeat
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import time
import json
import shutil
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "strategy.py")
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments")
LOG_PATH = os.path.join(EXPERIMENTS_DIR, "log.jsonl")
PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "program.md")


def _read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


def _write_file(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)


def _git_init():
    """Initialize git repo if not already initialized."""
    import git
    project_dir = os.path.dirname(__file__)
    try:
        repo = git.Repo(project_dir)
        return repo
    except git.InvalidGitRepositoryError:
        repo = git.Repo.init(project_dir)
        # Initial commit with current state
        repo.index.add([
            "strategy.py", "prepare.py", "run.py", "loop.py", "cli.py",
            "config.yaml", "requirements.txt", ".gitignore",
        ])
        # Add package files
        for pkg in ["data", "analysis", "evaluation", "agent"]:
            pkg_dir = os.path.join(project_dir, pkg)
            if os.path.isdir(pkg_dir):
                for f in os.listdir(pkg_dir):
                    if f.endswith(".py"):
                        repo.index.add([os.path.join(pkg, f)])
        repo.index.commit("Initial FinAutoResearch setup")
        return repo


def _git_commit(repo, message: str):
    """Commit strategy.py changes."""
    try:
        repo.index.add(["strategy.py"])
        repo.index.commit(message)
    except Exception as e:
        console.print(f"[dim]Git commit skipped: {e}[/dim]")


def _load_experiment_log() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    experiments = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def _log_experiment(experiment: dict):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(experiment) + "\n")


def _load_program() -> str:
    if os.path.exists(PROGRAM_PATH):
        return _read_file(PROGRAM_PATH)
    return ""


def run_loop(max_experiments: int = 20, time_limit_minutes: int = 60):
    """Run the AutoResearch optimization loop."""
    from prepare import load_strategy, run_full_cycle, evaluate, load_config
    from agent.optimizer import StrategyOptimizer

    config = load_config()
    loop_config = config.get("loop", {})
    improvement_threshold = loop_config.get("improvement_threshold", 0.02)

    console.print(Panel(
        "[bold blue]FinAutoResearch Optimization Loop[/bold blue]\n\n"
        f"Max experiments: {max_experiments}\n"
        f"Time limit: {time_limit_minutes} minutes\n"
        f"Improvement threshold: {improvement_threshold}\n"
        f"Primary metric: Sharpe Ratio",
        box=box.DOUBLE,
    ))

    start_time = time.time()

    # Initialize
    repo = _git_init()
    optimizer = StrategyOptimizer()
    program = _load_program()

    # 1. Baseline evaluation
    console.print("\n[yellow]Running baseline evaluation...[/yellow]")
    baseline_text = _read_file(STRATEGY_PATH)
    _git_commit(repo, "baseline strategy (loop start)")

    strategy = load_strategy()
    baseline_result = run_full_cycle(strategy, show_progress=True)
    baseline_metric = evaluate(baseline_result)
    best_metric = baseline_metric
    best_text = baseline_text

    _log_experiment({
        "experiment_id": 0,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "baseline",
        "sharpe": best_metric,
        "metrics": baseline_result.backtest.metrics,
        "kept": True,
    })

    console.print(f"\n[bold green]Baseline Sharpe: {baseline_metric:.4f}[/bold green]\n")

    # 2. Optimization loop
    kept_count = 0
    total_count = 0

    for i in range(1, max_experiments + 1):
        elapsed_min = (time.time() - start_time) / 60
        if elapsed_min > time_limit_minutes:
            console.print(f"\n[yellow]Time limit reached ({elapsed_min:.1f} min)[/yellow]")
            break

        total_count += 1
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Experiment {i}/{max_experiments}[/bold] "
                       f"(elapsed: {elapsed_min:.1f} min, best Sharpe: {best_metric:.4f})")

        # 2a. Read current strategy and history
        current_text = _read_file(STRATEGY_PATH)
        history = _load_experiment_log()

        # 2b. Ask Claude to propose a change
        console.print("  [dim]Proposing strategy change...[/dim]")
        try:
            proposed_text = optimizer.propose_change(
                current_text, history,
                baseline_result.backtest.metrics, program,
            )
        except Exception as e:
            console.print(f"  [red]Proposal failed: {e}[/red]")
            _log_experiment({
                "experiment_id": i,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": f"proposal_error: {str(e)[:100]}",
                "sharpe": None,
                "kept": False,
            })
            continue

        # 2c. Validate
        valid, reason = optimizer.validate_strategy(proposed_text)
        if not valid:
            console.print(f"  [red]Invalid proposal: {reason}[/red]")
            _log_experiment({
                "experiment_id": i,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": f"invalid: {reason}",
                "sharpe": None,
                "kept": False,
            })
            continue

        hypothesis = optimizer.extract_hypothesis(proposed_text)
        console.print(f"  [cyan]Hypothesis: {hypothesis}[/cyan]")

        # 2d. Apply change
        _write_file(STRATEGY_PATH, proposed_text)
        _git_commit(repo, f"experiment {i}: {hypothesis}")

        # 2e. Run backtest
        console.print("  [dim]Running backtest...[/dim]")
        try:
            strategy = load_strategy()
            result = run_full_cycle(strategy, show_progress=False)
            new_metric = evaluate(result)
        except Exception as e:
            console.print(f"  [red]Backtest failed: {e}[/red]")
            _write_file(STRATEGY_PATH, best_text)
            _git_commit(repo, f"revert experiment {i}: backtest error")
            _log_experiment({
                "experiment_id": i,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "sharpe": None,
                "metrics": {},
                "kept": False,
                "error": str(e)[:200],
            })
            continue

        # 2f. Evaluate: keep or revert
        improvement = new_metric - best_metric

        if improvement > improvement_threshold:
            # KEEP
            best_metric = new_metric
            best_text = proposed_text
            kept_count += 1
            console.print(
                f"  [bold green]KEPT[/bold green] — Sharpe: {new_metric:.4f} "
                f"(+{improvement:.4f})"
            )
            _log_experiment({
                "experiment_id": i,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "sharpe": new_metric,
                "improvement": improvement,
                "metrics": result.backtest.metrics,
                "kept": True,
            })
        else:
            # REVERT
            _write_file(STRATEGY_PATH, best_text)
            _git_commit(repo, f"revert experiment {i}: no improvement ({new_metric:.4f})")
            console.print(
                f"  [red]REVERTED[/red] — Sharpe: {new_metric:.4f} "
                f"({improvement:+.4f})"
            )
            _log_experiment({
                "experiment_id": i,
                "timestamp": datetime.now().isoformat(),
                "hypothesis": hypothesis,
                "sharpe": new_metric,
                "improvement": improvement,
                "metrics": result.backtest.metrics,
                "kept": False,
            })

    # 3. Summary
    elapsed_total = (time.time() - start_time) / 60

    console.print(f"\n{'='*60}")
    summary = Table(title="Optimization Summary", box=box.ROUNDED)
    summary.add_column("", style="cyan")
    summary.add_column("", style="green", justify="right")
    summary.add_row("Experiments run", str(total_count))
    summary.add_row("Changes kept", str(kept_count))
    summary.add_row("Baseline Sharpe", f"{baseline_metric:.4f}")
    summary.add_row("Final Sharpe", f"{best_metric:.4f}")
    summary.add_row("Improvement", f"{best_metric - baseline_metric:+.4f}")
    summary.add_row("Total time", f"{elapsed_total:.1f} min")
    console.print(summary)

    if best_metric > baseline_metric:
        console.print(
            f"\n[bold green]Strategy improved! "
            f"Sharpe went from {baseline_metric:.4f} to {best_metric:.4f} "
            f"({best_metric - baseline_metric:+.4f})[/bold green]"
        )
    else:
        console.print("\n[yellow]No improvements found. Baseline strategy retained.[/yellow]")

    console.print(f"\nExperiment log: {LOG_PATH}")
    console.print(f"Strategy file:  {STRATEGY_PATH}")
    console.print(f"Git log:        git log --oneline strategy.py\n")


if __name__ == "__main__":
    run_loop()
