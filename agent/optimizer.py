"""Strategy optimization agent — the 'brain' of the AutoResearch loop.

This agent reads strategy.py, proposes changes, and returns a modified version.
It uses Claude to reason about what parameter changes might improve Sharpe ratio.
"""

import os
import json
import ast
from anthropic import Anthropic


SYSTEM_PROMPT = """You are an expert quantitative equity strategist optimizing a multi-factor stock selection model.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy.py contents and experiment history, propose ONE focused
change to improve the Sharpe ratio. You MUST return a complete, valid strategy.py file.

RULES:
1. Make ONE focused change per experiment — don't change everything at once
2. Always update the docstring at the top with your hypothesis
3. Factor category weights should sum to approximately 1.0
4. Sub-factor weights within each category should sum to approximately 1.0
5. PORTFOLIO.top_n must be between 5 and 50
6. All numeric values must be valid Python literals
7. Do not add imports or function definitions — strategy.py is pure data
8. Consider financial theory and empirical research when proposing changes

TYPES OF CHANGES YOU CAN MAKE:
- Adjust factor category weights (e.g., increase quality weight, decrease momentum)
- Add or remove sub-factors within a category
- Change screening criteria (add, remove, or modify thresholds)
- Modify portfolio construction (top_n, weighting method, sector limits)
- Add new factor categories (e.g., "volatility", "size", "liquidity")
- Remove factor categories that aren't contributing
- Change universe exclusions

FINANCIAL REASONING TO APPLY:
- Quality + Value tends to outperform over full cycles
- Momentum works but can be volatile; consider shorter or longer lookbacks
- Smaller portfolios (top 10-15) tend to be more concentrated alpha
- Score-weighted portfolios may outperform equal-weighted
- Sector concentration limits prevent single-sector risk
- Stricter screens can improve quality at the cost of universe size
- Profitability factors (gross profits / assets) have strong academic support
- Low volatility anomaly: lower-beta stocks often have higher risk-adjusted returns"""


class StrategyOptimizer:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # OAuth tokens (sk-ant-oat*) must be passed as auth_token, not api_key
        if api_key.startswith("sk-ant-oat"):
            self.client = Anthropic(auth_token=api_key)
        else:
            self.client = Anthropic(api_key=api_key)
        self.model = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-6")

    def propose_change(
        self,
        current_strategy_text: str,
        experiment_history: list[dict],
        current_metrics: dict,
        program_instructions: str = "",
    ) -> str:
        """Ask Claude to propose a modified strategy.py. Returns the full new file text."""
        # Build experiment history summary
        history_text = ""
        if experiment_history:
            history_text = "\nEXPERIMENT HISTORY (most recent last):\n"
            for exp in experiment_history[-15:]:  # Last 15 experiments
                kept = "KEPT" if exp.get("kept") else "REVERTED"
                sharpe = exp.get("sharpe", "N/A")
                if isinstance(sharpe, float):
                    sharpe = f"{sharpe:.4f}"
                history_text += (
                    f"  #{exp.get('experiment_id', '?')}: [{kept}] "
                    f"Sharpe={sharpe} — {exp.get('hypothesis', 'N/A')}\n"
                )

        # Build metrics summary
        metrics_text = ""
        if current_metrics:
            metrics_text = "\nCURRENT BEST METRICS:\n"
            for k, v in current_metrics.items():
                if isinstance(v, float):
                    metrics_text += f"  {k}: {v:.4f}\n"

        prompt = f"""Here is the current strategy.py:

```python
{current_strategy_text}
```
{history_text}
{metrics_text}
{f"ADDITIONAL INSTRUCTIONS:{chr(10)}{program_instructions}" if program_instructions else ""}

Propose ONE focused change to improve the Sharpe ratio. Return the COMPLETE modified strategy.py
file (not just the diff). Update the docstring with your hypothesis.

Return ONLY the Python code, no markdown fences or explanation."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])  # remove first fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        return text

    def validate_strategy(self, strategy_text: str) -> tuple[bool, str]:
        """Validate that proposed strategy.py is syntactically valid and sane."""
        # 1. Syntax check
        try:
            ast.parse(strategy_text)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # 2. Check required variables exist
        try:
            namespace = {}
            exec(strategy_text, namespace)
        except Exception as e:
            return False, f"Execution error: {e}"

        required = ["UNIVERSE", "SCREENS", "FACTORS", "PORTFOLIO"]
        for var in required:
            if var not in namespace:
                return False, f"Missing required variable: {var}"

        # 3. Sanity checks
        factors = namespace.get("FACTORS", {})
        total_weight = sum(f.get("weight", 0) for f in factors.values())
        if abs(total_weight - 1.0) > 0.15:
            return False, f"Factor weights sum to {total_weight:.2f}, should be ~1.0"

        portfolio = namespace.get("PORTFOLIO", {})
        top_n = portfolio.get("top_n", 20)
        if not (3 <= top_n <= 60):
            return False, f"top_n={top_n} is out of range [3, 60]"

        screens = namespace.get("SCREENS", [])
        if not isinstance(screens, list):
            return False, "SCREENS must be a list"

        return True, "OK"

    def extract_hypothesis(self, strategy_text: str) -> str:
        """Extract the hypothesis from the strategy docstring."""
        try:
            tree = ast.parse(strategy_text)
            docstring = ast.get_docstring(tree)
            if docstring:
                for line in docstring.split("\n"):
                    line = line.strip()
                    if line.lower().startswith("experiment:"):
                        return line.split(":", 1)[1].strip()
                    if line.lower().startswith("hypothesis:"):
                        return line.split(":", 1)[1].strip()
            return "no hypothesis stated"
        except Exception:
            return "could not parse hypothesis"
