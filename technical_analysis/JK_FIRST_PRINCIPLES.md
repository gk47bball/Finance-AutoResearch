# JK Indicators: First Principles Analysis & Strategy Design

## Part 1: What Are These Indicators *Really* Doing?

Strip away the names and parameters. At their core, the 16 JK indicators reduce to **4 independent information sources**:

### Source 1: Multi-Timeframe Trend Alignment (MultiMAC family)
**Core idea:** Subtract fast EMAs from slow EMAs across 5 cascading timeframes. When all 5 differences are positive, every timeframe agrees the trend is up.

| Variant | Twist | Rating |
|---------|-------|--------|
| **multimac** | Raw EMA diffs / price avg | B |
| **multimac_fib** | Fibonacci periods + pct normalization | B+ |
| **multimac_rsi** | Adds RSI overlay for mean-reversion | C (disabled) |
| **multimac_dampened** | Caps long-term diffs to prevent domination | C (disabled) |

**First principles verdict:** This is a **trend strength thermometer**. +100 = perfect alignment across all timeframes. -100 = total breakdown. The Fibonacci variant is slightly better because percentage normalization makes it comparable across price levels. The RSI and dampened variants over-engineer a clean concept — disabled for good reason.

**Unique information:** How aligned are moving averages from 7-day to 89-day? No other indicator captures this multi-timeframe consensus.

---

### Source 2: Intraday Buying Pressure (Z-Factor family)
**Core idea:** Where does price close within the day's range? `(Close - Low) / TrueRange`. Smoothed over fast and slow windows, then differenced.

| Variant | Twist | Rating |
|---------|-------|--------|
| **z_factor** | Fast(10) - Slow(21) SMA of daily Z | A- |
| **z_hybrid** | Adds Z bias term (like hybrid oscillator) | A |

**First principles verdict:** This is the **only indicator using High/Low data** — everything else is close-only or close+volume. Z-Factor captures information the others literally cannot see: are buyers stepping in at the lows (bullish) or are sellers smashing it into the close (bearish)?

**z_hybrid wins** because the bias term `(z_slow - 50) / 5` acts as an anchor — when the slow Z is below 50 (persistent weak closes), even a fast Z recovery gets dampened. This prevents false signals in downtrends.

**Unique information:** Intraday price action. Completely orthogonal to MA-based and RSI-based indicators.

**Why the extreme spread is largest (29.9%):** When z_hybrid hits bottom decile, it means institutional selling is driving closes to the low of the range across multiple timeframes simultaneously — a capitulation pattern that reliably snaps back.

---

### Source 3: Momentum Oscillators (RSI family)
**Core idea:** Rate of change in price, smoothed to oscillate between overbought/oversold.

| Variant | Twist | Rating |
|---------|-------|--------|
| **ve_rsi** | Standard RSI weighted by volume ratio | B |
| **hybrid_osc** | RSI(34)-RSI(55) differential + bias | B+ |
| **rsi_diff** | Raw RSI(34)-RSI(55) | C (disabled) |

**First principles verdict:** VE-RSI's volume weighting is its unique edge — a 2% drop on 3x volume means more than a 2% drop on normal volume. The hybrid oscillator adds the same bias trick as z_hybrid, giving it better regime awareness. RSI_diff is too noisy alone (rightfully disabled).

**Unique information:** VE-RSI is the only indicator using volume data. Volume confirms conviction. A selloff on low volume is noise; on high volume it's meaningful.

---

### Source 4: Mean-Reversion Distance (OBOS)
**Core idea:** How far is price from its moving average, normalized by the historical maximum distance? A "rubber band" indicator.

| Variant | Twist | Rating |
|---------|-------|--------|
| **obos** | Distance from SMA(17) / max historical distance | B- |
| **mfoo** | Average of VE-RSI momentum + OBOS reversion | B- |

**First principles verdict:** OBOS is the purest mean-reversion signal — it literally measures how stretched the rubber band is. But it's also the most dangerous: in trending markets, the rubber band keeps stretching. MFOO's blend with VE-RSI partially solves this, but both have the weakest extreme spreads (13.5-20%).

**Unique information:** Normalized distance from MA. Other indicators measure direction or momentum, not raw displacement.

---

### Source 5: Regime Classification (Trend Score)
**Core idea:** Count how many of 10 boolean conditions are true (price vs 4 MAs, plus all 6 MA pair comparisons). Score from -5 to +5.

| Variant | Twist | Rating |
|---------|-------|--------|
| **trend_score** | Discrete vote count across 4 MA timeframes | A |

**First principles verdict:** This is fundamentally different from the others. It's not an oscillator — it's a **regime classifier**. A score of +5 means "perfect bull" and -5 means "perfect bear." It doesn't try to predict direction; it describes the current state. That makes it the ideal **context layer** for the other indicators.

**Why the bottom decile is explosive (+36.5%):** A trend score of -5 means ALL four moving averages are inverted (every fast MA below every slow MA, price below all of them). This is maximum bearish alignment — but by the time all MAs agree, the move is usually exhausted. The snapback from -5 is the strongest single signal in the entire system.

**Unique information:** Discrete regime state. Binary yes/no votes across timeframes, not a continuous oscillator.

---

## Part 2: Indicator Ratings

### Tier System

**Tier 1 — Core (must have, unique information):**
| Indicator | Rating | Why |
|-----------|--------|-----|
| **z_hybrid** | **A** | Largest extreme spread (29.9%). Only uses H/L/C. Orthogonal to everything else. |
| **trend_score** | **A** | Best regime classifier. Discrete voting prevents curve-fitting. Explosive bottom decile. |
| **z_factor** | **A-** | IC leader (-0.135 SPY). Fastest mean-reversion (20d optimal). Partially redundant with z_hybrid. |

**Tier 2 — Valuable (add unique info, keep):**
| Indicator | Rating | Why |
|-----------|--------|-----|
| **hybrid_osc** | **B+** | Best long-horizon IC (-0.196 at 60d). Smooth. Second-largest spread (27.1%). |
| **multimac_fib** | **B+** | Multi-timeframe trend. Fibonacci periods give natural spacing. DIA IC leader (-0.162). |
| **ve_rsi** | **B** | Only volume-aware indicator. Unique signal. Long-term IC builds to -0.179. |

**Tier 3 — Redundant (can drop without losing edge):**
| Indicator | Rating | Why |
|-----------|--------|-----|
| **multimac** | **B** | Good but ~95% correlated with multimac_fib. Keep one, drop the other. |
| **mfoo** | **B-** | Composite of ve_rsi + obos. Weakest spread (13.5%). Adds complexity not alpha. |
| **obos** | **B-** | Purest mean-reversion but redundant with the Z family. Dangerous in trends. |

**Tier 4 — Disabled (correctly so):**
| Indicator | Rating | Why |
|-----------|--------|-----|
| **rsi_diff** | **C** | Noisy. hybrid_osc does the same thing better. |
| **multimac_rsi** | **C** | Over-parameterized. Blending trend + RSI this way just muddies both signals. |
| **multimac_dampened** | **C** | Solving a problem that doesn't exist. Capping long-term diffs loses real signal. |

---

## Part 3: The "Four Pillars" Strategy

Instead of averaging 9 correlated indicators into mush, use **4 orthogonal pillars** with distinct roles:

```
┌──────────────────────────────────────────────────────┐
│                  JK FOUR PILLARS                     │
│                                                      │
│  PILLAR 1: REGIME    ─── trend_score ───────────┐    │
│  "Where are we?"     -5 to +5 discrete state    │    │
│                                                  │    │
│  PILLAR 2: TIMING    ─── z_hybrid ──────────┐   │    │
│  "When to enter?"    Intraday buying pressure│   │    │
│                                              │   │    │
│  PILLAR 3: MOMENTUM  ─── hybrid_osc ────┐   │   │    │
│  "Is the move real?" RSI differential    │   │   │    │
│                                          ▼   ▼   ▼    │
│  PILLAR 4: VOLUME    ─── ve_rsi ────►  SIGNAL  ◄──   │
│  "Is there conviction?" Vol-weighted RSI              │
│                                                      │
│  CONFLUENCE: Count pillars agreeing (0-4)            │
└──────────────────────────────────────────────────────┘
```

### How It Works:

**Step 1: Regime Check (trend_score)**
- Score ≥ +3: **BULL regime** → Enable contrarian dip-buying. Look for oversold signals.
- Score ≤ -3: **BEAR regime** → Go defensive. Only trade with extreme oversold + volume confirmation.
- Score -2 to +2: **CHOP zone** → Half position size. Signals are unreliable here.

**Step 2: Entry Signal (z_hybrid)**
- z_hybrid z-score drops below -1.5: **Oversold trigger**. Start watching for entry.
- z_hybrid z-score drops below -2.0: **Deep oversold**. Strong entry signal.
- This is the primary timing tool because it has the fastest response (20d optimal hold) and biggest extreme spread.

**Step 3: Momentum Confirmation (hybrid_osc)**
- hybrid_osc below its signal line AND declining: Confirms the dip is real, not just noise.
- hybrid_osc below signal line BUT turning up: **Buy signal** — momentum is shifting while still oversold.
- This prevents buying into a trending decline (where z_hybrid keeps screaming "oversold" but price keeps falling).

**Step 4: Volume Confirmation (ve_rsi)**
- ve_rsi < 35 with rising volume: **Capitulation** — high-conviction selling exhaustion.
- ve_rsi < 35 with falling volume: **Drift** — weak sellers, less reliable bounce.
- ve_rsi divergence (price making new low, ve_rsi making higher low): **Strong buy**.

### Position Sizing Matrix:

| Regime (trend_score) | Signal (z_hybrid) | Confirmation (hybrid_osc + ve_rsi) | Position |
|----------------------|--------------------|------------------------------------|----------|
| Bull (≥+3) | Deep oversold (z < -2.0) | Both confirm | **100%** |
| Bull (≥+3) | Oversold (z < -1.5) | One confirms | **75%** |
| Bull (≥+3) | Oversold (z < -1.5) | Neither confirms | **50%** |
| Chop (-2 to +2) | Deep oversold | Both confirm | **50%** |
| Chop (-2 to +2) | Oversold | Any | **25%** |
| Bear (≤-3) | Deep oversold | Both confirm + volume surge | **50%** |
| Bear (≤-3) | Any other | — | **0%** (flat) |

### Exit Rules:
1. **Primary exit:** z_hybrid crosses back above 0 (mean-reversion complete). ~15-20 days typical.
2. **Extended hold:** If hybrid_osc is still rising when z_hybrid crosses 0, hold until hybrid_osc peaks. ~40-60 days.
3. **Stop loss:** -5% from entry. Non-negotiable. Protects against the tail risk of buying oversold in a crash.
4. **Trailing stop:** Once +3% in profit, trail at 2% below high water mark.
5. **Time stop:** Exit after 60 days regardless (diminishing IC beyond this).

### What Makes This Different from the Current Strategy:

| Current Approach | Four Pillars |
|------------------|-------------|
| 9 indicators averaged into one score | 4 indicators with distinct *roles* |
| Binary position (0% or 100%) | Graduated sizing (0/25/50/75/100%) |
| Single threshold (z > -0.3 → all-in) | Multi-gate entry (regime + timing + confirmation) |
| No exit logic (just when signal flips) | Explicit exits (z_hybrid cross, stop, trail, time) |
| ~85% time in market | ~30-50% time in market (higher quality trades) |
| Sharpe 0.965 | Target: Sharpe 1.2+ with lower drawdown |

### Assets to Trade:
- **Primary:** SPY, DIA (proven edge, IC -0.12 to -0.16)
- **Secondary:** XLU, XLV, XLF (defensive sectors, IC -0.10 to -0.20)
- **Never:** High-beta names (TSLA, AMD, COIN), QQQ (momentum-driven, signals invert)

---

## Part 4: Why This Should Work (and Where It Could Fail)

### Why it should work:
1. **Orthogonal information.** Each pillar sees something the others can't: MAs (trend state), H/L/C (intraday pressure), RSI differential (momentum divergence), volume (conviction). Averaging correlated signals adds noise; combining orthogonal signals adds information.

2. **Regime-appropriate behavior.** The current strategy trades the same way in bulls and bears. The Four Pillars strategy adapts: aggressive dip-buying in bulls, defensive in bears, cautious in chop. This aligns with the scenario analysis finding that bull_volatile has 2-3x the IC of other regimes.

3. **Quality over quantity.** Trading ~35% of the time at higher conviction should outperform trading ~85% of the time at mixed conviction. The scenario data shows the bottom 2 deciles capture 80%+ of the alpha.

4. **Explicit risk management.** The current strategy has no stops. A -5% stop + trailing stop structurally limits drawdowns while letting winners run.

### Where it could fail:
1. **Overfitting to SPY.** All parameters are calibrated on SPY. The strategy may not generalize even to DIA.
2. **Regime classification lag.** Trend score uses 13-55 day MAs. By the time it confirms "bear," you've already missed the first 10-15% of the decline.
3. **Stop loss whipsaw.** In volatile markets, a -5% stop could trigger right before a reversal. Need to consider volatility-adjusted stops.
4. **Reduced exposure = reduced compounding.** Being in market only 35% of the time means missing bull run days. The cash drag could offset the improved trade quality.
5. **10 years of data is one regime.** 2015-2025 was predominantly a bull market with 2 sharp bears (2020, 2022). The strategy hasn't seen a 2008-style multi-year bear.
