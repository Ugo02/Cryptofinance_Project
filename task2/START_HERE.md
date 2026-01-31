# 🚀 Quick Start Guide

## What is this?

Bitcoin mining strategies analysis for the **CryptoFinance course** (ESILV, Semester 9).

**Main finding:** Selfish mining is profitable with only **25% hash power** (not 51%!)

---

## Files Overview

| File | Purpose |
|------|---------|
| **📊 README.md** | **Complete results with graphs** ← Read this first! |
| **⚙️ TECHNICAL_GUIDE.md** | Setup and usage instructions |
| **results/** | Generated plots (3 PNG files) |

**Code files:**
- `quick_demo.py` - Generate plots quickly (30 sec)
- `profitability_heatmap.py` - Generate heatmap (2 min)
- `mining_strategies.py` - Core implementation
- `visualizations.py` - Plotting functions
- `main.py` - Full analysis (10-15 min)

---

## Quick Actions

### View Results

```bash
# Read the main document
open README.md

# View the plots
open results/task_2_1_strategy_comparison.png
open results/task_2_2_profitability_heatmap.png
open results/task_2_3_optimal_strategy.png
```

### Regenerate Plots

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Generate all plots (under 3 minutes)
python quick_demo.py
python profitability_heatmap.py
```

### Understand the Code

```bash
# See setup instructions
open TECHNICAL_GUIDE.md

# Read the implementation
open mining_strategies.py
```

---

## Three Tasks Completed

✅ **Task 2.1** - Honest vs Selfish mining simulation
✅ **Task 2.2** - Profitability analysis (q, γ) parameter space
✅ **Task 2.3** - Optimal selfish mining strategy matrix

---

## Key Results

- **Selfish mining profitable at q ≈ 0.25** (with γ = 0.5)
- **Only need 20-35% hash power** (depending on connectivity)
- **Clear optimal strategy:** 4 actions (Adopt/Wait/Match/Override)

See **README.md** for complete analysis with embedded graphics!

---

*ESILV - CryptoFinance - January 2026*
