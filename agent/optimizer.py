"""Strategy optimization agent — the 'brain' of the AutoResearch loop.

This agent reads strategy.py, proposes changes, and returns a modified version.
It uses Claude to reason about what parameter changes might improve Sharpe ratio.
"""

import os
import json
import ast
from anthropic import Anthropic


DOMAIN_PROMPTS = {
    "stock_picker": """You are an expert quantitative equity strategist optimizing a multi-factor stock selection model.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy file contents and experiment history, propose ONE focused
change to improve the Sharpe ratio. You MUST return a complete, valid strategy file.

RULES:
1. Make ONE focused change per experiment — don't change everything at once
2. Always update the docstring at the top with your hypothesis
3. Factor category weights should sum to approximately 1.0
4. Sub-factor weights within each category should sum to approximately 1.0
5. PORTFOLIO.top_n must be between 5 and 50
6. All numeric values must be valid Python literals
7. Do not add imports or function definitions — the strategy file is pure data
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
- Low volatility anomaly: lower-beta stocks often have higher risk-adjusted returns""",

    "sector_rotation": """You are an expert tactical asset allocator optimizing a sector rotation strategy.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy file and experiment history, propose ONE focused change
to improve the Sharpe ratio. Return the COMPLETE modified strategy file.

RULES:
1. Make ONE focused change per experiment — don't change everything at once
2. Always update the docstring with your hypothesis (line starting with "Hypothesis:")
3. Factor category weights should sum to approximately 1.0
4. Sub-factor weights within each category should sum to approximately 1.0
5. PORTFOLIO.top_n must be between 2 and 11 (there are only 11 sectors)
6. Do not add imports — the strategy file is pure data
7. Available sub-factors for ETFs: return_3m, return_6m, return_12m_1m (all price-derived)

TYPES OF CHANGES YOU CAN MAKE:
- Adjust momentum vs trend weights
- Change sub-factor weights within momentum or trend
- Modify how many sectors to hold (top_n)
- Switch weighting from score_weighted to equal or vice versa
- Add or remove sector ETFs from the universe
- Add or adjust volume/liquidity screens
- Change rebalance frequency (weekly, monthly, quarterly)
- Adjust max_sector_pct concentration limit

SECTOR ROTATION REASONING:
- Momentum (6-12m) captures sector trends driven by business cycle rotation
- Shorter momentum (3m) captures intermediate trends but is noisier
- 12-1m momentum (skip last month) avoids short-term reversal
- Holding fewer sectors (3-4) concentrates in strongest trends
- Equal weighting may reduce whipsaw vs score weighting
- Monthly rebalance balances responsiveness vs transaction costs""",

    "tactical_allocation": """You are an expert macro strategist optimizing a tactical asset allocation model.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy file and experiment history, propose ONE focused change.
Return the COMPLETE modified strategy file.

RULES:
1. Make ONE focused change per experiment
2. Always update the docstring with your hypothesis
3. Factor weights must sum to approximately 1.0
4. Sub-factor weights within each category must sum to approximately 1.0
5. Do not add imports — the strategy file is pure data
6. Available sub-factors: return_3m, return_6m, return_12m_1m (all price-derived)
7. Available assets: SPY (equity), TLT (long bonds), IEF (mid bonds), GLD (gold), SHY (cash)

TYPES OF CHANGES YOU CAN MAKE:
- Adjust momentum vs trend weights
- Change sub-factor weights
- Add or remove asset classes
- Modify top_n (how many assets to hold)
- Switch weighting method
- Adjust max_sector_pct
- Change rebalance frequency
- Modify regime rules (risk_on/risk_off allocations)

MACRO ALLOCATION REASONING:
- Cross-asset momentum captures flight-to-quality and risk-on/risk-off flows
- Trend following (3m) identifies regime shifts before momentum catches up
- Gold tends to rally when real rates fall or risk rises
- TLT outperforms in deflation/recession, underperforms in rising rates
- Holding all 5 assets provides diversification but dilutes momentum signal
- Score-weighting tilts more to winners; equal-weight is more defensive""",

    "long_short": """You are an expert quantitative portfolio manager optimizing a long-short equity strategy.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy file and experiment history, propose ONE focused change.
Return the COMPLETE modified strategy file.

RULES:
1. Make ONE focused change per experiment
2. Always update the docstring with your hypothesis
3. Factor weights must sum to approximately 1.0
4. Sub-factor weights within each category must sum to approximately 1.0
5. Do not add imports — the strategy file is pure data
6. PORTFOLIO must include top_n (long leg) and short_n (short leg)

TYPES OF CHANGES YOU CAN MAKE:
- Adjust factor category weights (value, quality, growth, momentum)
- Change sub-factor weights
- Modify long/short counts (top_n, short_n)
- Adjust long_weight/short_weight (gross exposure)
- Change screening criteria
- Modify sector caps
- Change rebalance frequency
- Adjust universe exclusions
- Change SHORT_CONFIG parameters

LONG-SHORT REASONING:
- The short leg tests whether the factor model works symmetrically
- Low beta (near 0) means market-neutral — attractive for risk-adjusted returns
- Quality + growth shorts tend to be more alpha-generative than value shorts
- Wider screens (looser criteria) needed to get enough stocks for both legs
- 50/50 gross exposure targets market neutrality
- Fewer shorts (5-8) may be more concentrated but higher tracking error
- Borrow costs erode short alpha — focus on high-conviction shorts""",

    "crypto_momentum": """You are an expert crypto quant optimizing a momentum-based crypto portfolio strategy.

You are part of an AutoResearch loop — an autonomous system that iterates on a strategy file
to maximize risk-adjusted returns (Sharpe ratio) from a walk-forward backtest.

YOUR TASK: Given the current strategy file and experiment history, propose ONE focused change.
Return the COMPLETE modified strategy file.

RULES:
1. Make ONE focused change per experiment
2. Always update the docstring with your hypothesis
3. Factor weights must sum to approximately 1.0
4. Sub-factor weights within each category must sum to approximately 1.0
5. Do not add imports — the strategy file is pure data
6. Only use price-derived signals (no fundamentals for crypto)
7. Available sub-factors: return_3m, return_6m, return_12m_1m, beta_inv

TYPES OF CHANGES YOU CAN MAKE:
- Adjust momentum vs trend vs risk weights
- Change sub-factor weights within each category
- Add or remove crypto tickers (use yfinance format: BTC-USD, ETH-USD, etc.)
- Modify top_n (how many assets to hold)
- Switch weighting (equal vs score_weighted)
- Add or modify volume screens
- Change rebalance frequency
- Adjust max_sector_pct (per-asset cap)

CRYPTO MOMENTUM REASONING:
- Crypto momentum is stronger and more persistent than equity momentum
- Shorter lookbacks (3m) capture crypto's faster trend cycles
- Risk (1/beta) helps avoid highly correlated altcoins during BTC drawdowns
- Equal weighting prevents over-concentration in volatile altcoins
- Holding 3-5 assets is optimal — too many dilutes the momentum signal
- Monthly rebalance may miss crypto's faster cycles; weekly captures more
- Large-cap bias (BTC, ETH, SOL) provides liquidity and reduces tail risk
- Adding mid-cap alts increases return potential but also volatility""",
}

# Default for unknown domains
DEFAULT_PROMPT = DOMAIN_PROMPTS["stock_picker"]


class StrategyOptimizer:
    def __init__(self, domain: str = "stock_picker"):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        # OAuth tokens (sk-ant-oat*) must be passed as auth_token, not api_key
        if api_key.startswith("sk-ant-oat"):
            self.client = Anthropic(auth_token=api_key)
        else:
            self.client = Anthropic(api_key=api_key)
        self.model = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-5")
        self.system_prompt = DOMAIN_PROMPTS.get(domain, DEFAULT_PROMPT)

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
            system=self.system_prompt,
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
