"""Claude-powered deep-dive equity research agent."""

import os
import json
from anthropic import Anthropic
from data.edgar import get_filing_text, extract_section
from data.fundamentals import get_key_ratios


class ResearchAgent:
    def __init__(self):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env")
        # OAuth tokens (sk-ant-oat*) must be passed as auth_token, not api_key
        if api_key.startswith("sk-ant-oat"):
            self.client = Anthropic(auth_token=api_key)
        else:
            self.client = Anthropic(api_key=api_key)
        self.model = os.environ.get("AUTORESEARCH_MODEL", "claude-sonnet-4-6")

    def analyze_stock(self, ticker: str, quantitative_data: dict, config: dict) -> dict:
        """Produce a structured equity research analysis for a single stock."""
        # Gather qualitative data
        filing_context = ""
        if config.get("analyze_10k", True):
            full_text = get_filing_text(ticker, "10-K")
            if full_text:
                risk = extract_section(full_text, "risk_factors")
                mdna = extract_section(full_text, "mdna")
                if risk:
                    filing_context += f"\n--- 10-K RISK FACTORS (excerpt) ---\n{risk[:5000]}\n"
                if mdna:
                    filing_context += f"\n--- 10-K MD&A (excerpt) ---\n{mdna[:5000]}\n"

        if config.get("analyze_recent_8k", True):
            from data.edgar import get_recent_filings
            filings_8k = get_recent_filings(ticker, "8-K", count=3)
            if filings_8k:
                filing_context += f"\n--- Recent 8-K Filings ---\n"
                for f in filings_8k:
                    filing_context += f"  {f['date']}: {f['form']} (Accession: {f['accession_raw']})\n"

        # Build ratios summary
        ratios = quantitative_data if isinstance(quantitative_data, dict) else {}
        if not ratios or "ticker" not in ratios:
            ratios = get_key_ratios(ticker)

        ratios_text = self._format_ratios(ratios)
        focus_areas = config.get("focus_areas", [
            "competitive_moat", "management_quality", "risk_factors", "growth_catalysts"
        ])

        prompt = f"""You are a senior equity research analyst at a top-tier institutional investor.
Analyze {ticker} and provide a structured investment research note.

QUANTITATIVE DATA:
{ratios_text}

{filing_context if filing_context else "No SEC filing data available."}

Provide your analysis as a JSON object with these fields:
- "ticker": "{ticker}"
- "summary": A 2-3 paragraph investment thesis summary
- "competitive_moat": Assessment of the company's competitive moat (none/narrow/wide) with reasoning
- "key_risks": Array of 3-5 key risks
- "growth_catalysts": Array of 3-5 growth catalysts
- "management_assessment": Brief assessment of management quality based on available data
- "conviction": "high", "medium", or "low" — your conviction level for this as an investment
- "fair_value_signal": "undervalued", "fairly_valued", or "overvalued" based on quantitative metrics

Focus your analysis on: {', '.join(focus_areas)}

Respond ONLY with the JSON object, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Parse JSON from response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "ticker": ticker,
                "summary": text if "text" in dir() else "Analysis failed",
                "competitive_moat": "Unable to parse",
                "key_risks": [],
                "growth_catalysts": [],
                "conviction": "low",
            }
        except Exception as e:
            return {
                "ticker": ticker,
                "summary": f"Analysis error: {str(e)}",
                "competitive_moat": "N/A",
                "key_risks": [],
                "growth_catalysts": [],
                "conviction": "low",
            }

    def _format_ratios(self, ratios: dict) -> str:
        """Format ratios dict into readable text."""
        if not ratios:
            return "No ratio data available."

        lines = []
        display = {
            "sector": "Sector",
            "industry": "Industry",
            "market_cap": "Market Cap",
            "pe_ratio": "P/E Ratio",
            "forward_pe": "Forward P/E",
            "pb_ratio": "P/B Ratio",
            "ps_ratio": "P/S Ratio",
            "ev_to_ebitda": "EV/EBITDA",
            "profit_margin": "Profit Margin",
            "operating_margin": "Operating Margin",
            "gross_margin": "Gross Margin",
            "roe": "ROE",
            "roa": "ROA",
            "debt_to_equity": "Debt/Equity",
            "current_ratio": "Current Ratio",
            "revenue_growth": "Revenue Growth (YoY)",
            "earnings_growth": "Earnings Growth",
            "fcf_yield": "FCF Yield",
            "earnings_yield": "Earnings Yield",
            "dividend_yield": "Dividend Yield",
            "beta": "Beta",
        }

        for key, label in display.items():
            val = ratios.get(key)
            if val is not None:
                if isinstance(val, float):
                    if "margin" in key or "yield" in key or "growth" in key or key in ("roe", "roa"):
                        lines.append(f"  {label}: {val:.1%}")
                    elif key == "market_cap":
                        lines.append(f"  {label}: ${val:,.0f}")
                    else:
                        lines.append(f"  {label}: {val:.2f}")
                else:
                    lines.append(f"  {label}: {val}")

        return "\n".join(lines) if lines else "No ratio data available."
