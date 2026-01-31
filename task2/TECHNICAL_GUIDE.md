# Technical Guide - Bitcoin Mining Strategies

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

```bash
# Navigate to project directory
cd task2

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- `numpy` - Numerical computations and random number generation
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical graphics (optional, for enhanced styling)

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Unix/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### Quick Start (Recommended)

Generate all visualizations in under 3 minutes:

```bash
# Generate main plots (Task 2.1 and 2.3)
python quick_demo.py

# Generate profitability heatmap (Task 2.2)
python profitability_heatmap.py
```

**Output:** 3 PNG files in `results/` directory

### Full Analysis

Run comprehensive analysis with all tasks:

```bash
python main.py
```

**Note:** This takes 10-15 minutes and generates extensive output including:
- Console output with numerical results
- All visualization plots
- Results summary text file

---

## Project Structure

```
task2/
├── README.md                          # Results and findings (with graphics)
├── TECHNICAL_GUIDE.md                 # This file - setup instructions
├── requirements.txt                   # Python dependencies
│
├── mining_strategies.py               # Core implementations
├── visualizations.py                  # Plotting functions
│
├── quick_demo.py                      # Quick visualization generator
├── profitability_heatmap.py           # Heatmap generator
├── main.py                            # Full analysis runner
│
└── results/                           # Generated plots
    ├── task_2_1_strategy_comparison.png
    ├── task_2_2_profitability_heatmap.png
    └── task_2_3_optimal_strategy.png
```

---

## Code Architecture

### Core Classes

#### `HonestMining` (mining_strategies.py)
- **Purpose:** Simulate honest mining strategy
- **Key Methods:**
  - `theoretical_revenue()` → Returns q (hash power)
  - `simulate(num_blocks, seed)` → Monte Carlo simulation
- **Time Complexity:** O(n) for n blocks

#### `SelfishMining` (mining_strategies.py)
- **Purpose:** Simulate selfish mining with state machine
- **Key Methods:**
  - `theoretical_revenue()` → Run simulations to estimate revenue
  - `simulate(num_blocks, seed)` → State machine simulation
  - `is_profitable()` → Check if selfish > honest
- **Time Complexity:** O(n) for n blocks

#### `OptimalSelfishMining` (mining_strategies.py)
- **Purpose:** Determine optimal decisions for each state
- **Key Methods:**
  - `get_optimal_action(a, h)` → Decision for state (a, h)
  - `get_strategy_matrix(max_a, max_h)` → Full decision matrix
- **Time Complexity:** O(max_a × max_h)

### State Machine Implementation

The selfish mining simulation uses a state-based approach:

**State = lead** (attacker private chain length - public chain length)

**Transitions:**
```python
if attacker_mines:
    state += 1  # Increase lead
else:  # honest miners mine
    if state == 0:
        # Adopt their block
    elif state == 1:
        # Publish our block (race with probability γ)
    elif state == 2:
        # Publish both blocks to override
    else:  # state > 2
        # Publish one block to maintain lead
```

---

## Parameters

### Configurable Parameters

**In `quick_demo.py` and `profitability_heatmap.py`:**

- `q_values` - Array of hash power values to test (default: 0.10 to 0.40)
- `gamma` - Connectivity parameter (default: 0.5)
- `num_blocks` - Blocks per simulation (default: 3000-5000)
- `seed` - Random seed for reproducibility (default: 42)

**In `main.py`:**

- `num_simulations` - Number of Monte Carlo runs (default: 100)
- `resolution` - Grid resolution for heatmaps (default: 50x50)

### Parameter Meanings

- **q (alpha):** Fraction of total network hash power controlled by attacker (0 < q < 1)
- **γ (gamma):** Connectivity - fraction of honest miners who build on attacker's block during ties (0 ≤ γ ≤ 1)
- **a:** Number of blocks in attacker's private chain
- **h:** Number of blocks in honest miners' public chain

---

## Extending the Code

### Adding New Strategies

```python
from mining_strategies import MiningResult
import numpy as np

class CustomMiningStrategy:
    def __init__(self, q: float):
        self.q = q

    def simulate(self, num_blocks: int = 10000, seed: int = None) -> MiningResult:
        # Your simulation logic here
        # ...
        return MiningResult(
            revenue=blocks_accepted,
            blocks_mined=blocks_mined,
            blocks_accepted=blocks_accepted,
            total_blocks=num_blocks,
            relative_revenue=blocks_accepted / num_blocks
        )
```

### Creating Custom Visualizations

```python
import matplotlib.pyplot as plt
from mining_strategies import SelfishMining

# Your analysis
q_values = [0.1, 0.2, 0.3, 0.4]
revenues = []

for q in q_values:
    selfish = SelfishMining(q, gamma=0.5)
    result = selfish.simulate(num_blocks=5000, seed=42)
    revenues.append(result.relative_revenue)

# Your plot
plt.plot(q_values, revenues, 'o-')
plt.xlabel('Hash Power (q)')
plt.ylabel('Revenue')
plt.title('Custom Analysis')
plt.savefig('results/my_custom_plot.png', dpi=300)
```

---

## Performance Considerations

### Simulation Runtime

| Configuration | Blocks | Time |
|--------------|--------|------|
| Single run | 5,000 | ~0.05s |
| Single run | 10,000 | ~0.1s |
| Single run | 20,000 | ~0.2s |
| 20 runs | 5,000 | ~1s |
| Grid 20×20 | 3,000 | ~2min |

### Memory Usage

- Typical: < 100 MB
- Large analysis (100x100 grid): ~200 MB

### Optimization Tips

1. **Reduce num_blocks** for faster iterations (min: 1000)
2. **Reduce num_simulations** in `theoretical_revenue()` (min: 10)
3. **Lower grid resolution** in heatmaps (try 20×20 instead of 50×50)
4. **Use fixed seeds** for reproducibility

---

## Troubleshooting

### Common Issues

**ImportError: No module named 'numpy'**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**ImportError: No module named 'mining_strategies'**
```bash
# Solution: Run from task2 directory
cd task2
python quick_demo.py
```

**"externally-managed-environment" error**
```bash
# Solution: Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Plots not displaying**
```bash
# If running remotely, plots are saved to results/
# No need for display - check results/ directory
```

**Simulation takes too long**
```python
# Reduce parameters in the script:
num_blocks = 1000      # Instead of 5000
num_simulations = 10   # Instead of 100
resolution = 20        # Instead of 50
```

---

## Testing

### Quick Validation

```bash
# Run Python REPL
python3

# Test imports
>>> from mining_strategies import HonestMining, SelfishMining
>>> h = HonestMining(0.3)
>>> h.theoretical_revenue()
0.3

>>> s = SelfishMining(0.3, 0.5)
>>> result = s.simulate(1000, seed=42)
>>> print(f"Revenue: {result.relative_revenue:.4f}")
Revenue: 0.3040
```

### Expected Outputs

**Honest Mining (q=0.3):**
- Revenue should be exactly 0.3

**Selfish Mining (q=0.3, γ=0.5):**
- Revenue should be ~0.30-0.31 (slight advantage)

**Selfish Mining (q=0.35, γ=0.5):**
- Revenue should be ~0.39-0.40 (clear advantage)

---

## Dependencies Details

### numpy (>= 1.21.0)
- Random number generation (`np.random.random()`, `np.random.binomial()`)
- Array operations (`np.linspace()`, `np.array()`)
- Statistical functions (`np.mean()`, `np.std()`)

### matplotlib (>= 3.4.0)
- Figure creation (`plt.subplots()`, `plt.figure()`)
- Plotting (`plt.plot()`, `plt.imshow()`, `plt.contour()`)
- Saving (`plt.savefig()`)

### seaborn (>= 0.11.0)
- Styling (`sns.set_palette()`)
- Color schemes (used for better-looking plots)
- Optional: Code works without it, just uses default matplotlib styles

---

## Development Notes

### Design Decisions

1. **Simulation over Formula:** We use Monte Carlo simulation instead of closed-form formulas because:
   - The theoretical formula has multiple formulations in literature
   - Direct state machine implementation is more reliable
   - Results are easier to verify and debug

2. **Modular Architecture:** Each strategy is a separate class for:
   - Easy extension to new strategies
   - Clear separation of concerns
   - Reusable components

3. **Reproducible Results:** Fixed random seeds ensure:
   - Consistent outputs across runs
   - Debuggable code
   - Comparable results

### Code Quality

- Docstrings for all classes and methods
- Type hints for parameters and returns
- Comments explaining complex logic
- PEP 8 compliant (mostly)

---

## License & Attribution

This implementation is for educational purposes as part of the ESILV CryptoFinance course.

**Based on:**
- Eyal, I., & Sirer, E. G. (2014). "Majority is not Enough: Bitcoin Mining is Vulnerable."

**Use responsibly:**
- For academic study only
- Do not deploy on production systems
- Selfish mining harms the Bitcoin network

---

## Support

For issues or questions about the code:

1. Check this guide first
2. Review code comments in `mining_strategies.py`
3. Look at example usage in `quick_demo.py`
4. Check `README.md` for theoretical background

---

*Last Updated: January 2026*
*ESILV - CryptoFinance Course - Semester 9*
