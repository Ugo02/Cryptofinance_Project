"""
Conclusions: Repeated Double-Spending Attacks in Bitcoin

This module summarizes the key findings from our analysis of repeated
double-spending attacks with abandonment threshold strategy.

Author: Crypto Finance Research
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Import from our analysis modules
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from repeated_double_spending_analysis import (
    RepeatedDoubleSpendingAttack,
    calculate_expected_profit_per_attack
)


def generate_comprehensive_summary():
    """Generate comprehensive summary of all findings."""

    print("=" * 80)
    print("CONCLUSIONS: REPEATED DOUBLE-SPENDING ATTACKS IN BITCOIN")
    print("=" * 80)

    print("""
================================================================================
FRAMEWORK DEFINITION
================================================================================

We define a repeated double-spending attack with abandonment threshold as follows:

ATTACK PROTOCOL:
1. Attacker controls hash power fraction q (honest network has p = 1-q)
2. Attacker sends transaction for value v to merchant
3. Attacker immediately begins mining a secret fork (excluding their tx)
4. Merchant waits for n confirmations before releasing goods
5. After merchant releases goods, attacker races to extend secret fork
6. If attacker's fork becomes longer: SUCCESS - publish and double-spend
7. If attacker falls more than A blocks behind: ABANDON and restart

KEY PARAMETERS:
- q: Attacker's hash power fraction
- n: Number of confirmations required by merchant
- A: Abandonment threshold (maximum deficit before giving up)
- v: Value of double-spent goods (in BTC)
- R: Block reward (currently 6.25 BTC)

ECONOMIC CONSIDERATIONS:
- Opportunity cost: While attacking, attacker foregoes honest mining rewards
- The attack has positive expected value only if:
  E[Profit] = P(success) × v - E[Duration] × q × R > 0

================================================================================
KEY FINDINGS
================================================================================
""")

    # Run comprehensive numerical analysis
    n = 6  # Standard confirmations
    block_reward = 6.25

    # Finding 1: Success Probability
    print("\n1. SUCCESS PROBABILITY ANALYSIS")
    print("-" * 60)
    print("""
The abandonment threshold A fundamentally changes attack dynamics:

- Without abandonment (A = ∞): Attack has non-zero success probability for
  any q > 0, but may take infinite time. For q < 0.5, the expected duration
  is infinite (the random walk is transient).

- With abandonment (finite A): Attack always terminates in finite time.
  Success probability decreases but attack duration is bounded.
""")

    print("Success probabilities for n=6 confirmations:\n")
    print(f"{'q':>8} | {'A=5':>10} | {'A=10':>10} | {'A=20':>10} | {'A=50':>10}")
    print("-" * 55)

    for q in [0.1, 0.2, 0.3, 0.4, 0.45]:
        probs = []
        for A in [5, 10, 20, 50]:
            attack = RepeatedDoubleSpendingAttack(q, n, A)
            stats = attack.estimate_attack_statistics(80000)
            probs.append(stats['success_probability'])
        print(f"{q:>8.2f} | {probs[0]:>10.4f} | {probs[1]:>10.4f} | {probs[2]:>10.4f} | {probs[3]:>10.4f}")

    # Finding 2: Break-even Analysis
    print("\n\n2. BREAK-EVEN DOUBLE-SPEND VALUE")
    print("-" * 60)
    print("""
For an attack to be profitable on average, the double-spend value must
exceed a threshold that depends on (q, n, A).

Break-even value = (Opportunity Cost) / P(success)
                = (E[Duration] × q × R) / P(success)
""")

    print("\nMinimum profitable double-spend value (BTC) for n=6:\n")
    print(f"{'q':>8} | {'A=5':>10} | {'A=10':>10} | {'A=20':>10}")
    print("-" * 45)

    for q in [0.1, 0.2, 0.3, 0.4]:
        break_evens = []
        for A in [5, 10, 20]:
            stats = calculate_expected_profit_per_attack(q, n, A, 100, num_simulations=60000)
            break_evens.append(stats['break_even_value'])
        print(f"{q:>8.2f} | {break_evens[0]:>10.1f} | {break_evens[1]:>10.1f} | {break_evens[2]:>10.1f}")

    # Finding 3: Optimal Abandonment Threshold
    print("\n\n3. OPTIMAL ABANDONMENT THRESHOLD")
    print("-" * 60)
    print("""
The abandonment threshold A presents a trade-off:
- Low A: Faster attack cycles, but lower success probability
- High A: Higher success probability, but higher opportunity cost per attempt

There exists an optimal A* that maximizes expected profit rate.
""")

    print("\nOptimal A for different (q, v) combinations (n=6):\n")

    for q in [0.25, 0.30, 0.35, 0.40]:
        for v in [100, 500]:
            best_A = 1
            best_profit = float('-inf')

            for A in range(1, 41):
                stats = calculate_expected_profit_per_attack(q, n, A, v, num_simulations=30000)
                if stats['expected_profit'] > best_profit:
                    best_profit = stats['expected_profit']
                    best_A = A

            profit_check = calculate_expected_profit_per_attack(q, n, best_A, v, num_simulations=50000)
            print(f"q={q:.2f}, v={v:4d} BTC: Optimal A* = {best_A:2d}, "
                  f"E[Profit] = {profit_check['expected_profit']:7.2f} BTC")

    # Finding 4: Performance vs Double-Spend Amount
    print("\n\n4. ATTACK PERFORMANCE VS DOUBLE-SPEND AMOUNT")
    print("-" * 60)
    print("""
The relationship between double-spend value and attack profitability:

- Below break-even: Attack has negative expected value (lose money on average)
- Above break-even: Attack has positive expected value, profit scales linearly
- Key insight: For repeated attacks, even small positive expectation compounds
""")

    print("\nExpected profit for q=0.30, n=6, A=15:\n")
    print(f"{'Value (BTC)':>12} | {'P(success)':>10} | {'E[Gain]':>10} | {'Opp.Cost':>10} | {'E[Profit]':>10} | {'Status':>12}")
    print("-" * 75)

    for v in [25, 50, 75, 100, 150, 200, 500]:
        stats = calculate_expected_profit_per_attack(0.30, n, 15, v, num_simulations=60000)
        status = "PROFITABLE" if stats['expected_profit'] > 0 else "UNPROFITABLE"
        print(f"{v:>12.0f} | {stats['success_probability']:>10.4f} | "
              f"{stats['expected_gain']:>10.2f} | {stats['opportunity_cost']:>10.2f} | "
              f"{stats['expected_profit']:>10.2f} | {status:>12}")

    # Finding 5: Long-term Attack Strategy
    print("\n\n5. LONG-TERM ATTACK STRATEGY ANALYSIS")
    print("-" * 60)
    print("""
For repeated attacks, the relevant metric is the profit RATE (profit per unit time),
not just profit per attack. An attacker running continuous attacks cares about
maximizing their long-term expected return.

Profit Rate = E[Profit per attack] / E[Duration per attack]

This should be compared with honest mining rate = q × R
""")

    print("\nProfit rates comparison (n=6, A=15):\n")
    print(f"{'q':>6} | {'v (BTC)':>8} | {'Attack Rate':>12} | {'Honest Rate':>12} | {'Advantage':>10}")
    print("-" * 60)

    for q in [0.25, 0.30, 0.35, 0.40]:
        honest_rate = q * block_reward
        for v in [100, 500]:
            stats = calculate_expected_profit_per_attack(q, n, 15, v, num_simulations=50000)
            attack_rate = stats['expected_profit'] / stats['avg_attack_duration_blocks']
            advantage = (attack_rate - honest_rate) / honest_rate * 100 if honest_rate > 0 else 0
            print(f"{q:>6.2f} | {v:>8d} | {attack_rate:>12.4f} | {honest_rate:>12.4f} | {advantage:>9.1f}%")

    # Conclusions
    print("\n\n" + "=" * 80)
    print("MAIN CONCLUSIONS")
    print("=" * 80)

    print("""
1. VIABILITY OF REPEATED ATTACKS:
   - Repeated double-spending attacks with abandonment threshold are viable
     for attackers with sufficient hash power (roughly q > 0.20-0.25)
   - The abandonment strategy makes attacks practical by ensuring finite duration
   - Attackers must target high-value transactions to overcome opportunity costs

2. CRITICAL THRESHOLDS:
   - For q < 0.20: Attacks are rarely profitable, regardless of double-spend value
   - For q = 0.30: Break-even value is approximately 50-100 BTC (depending on A)
   - For q = 0.40: Break-even value drops to approximately 20-40 BTC
   - For q ≥ 0.50: Attacker always wins eventually (but this contradicts Nakamoto consensus)

3. OPTIMAL ABANDONMENT THRESHOLD:
   - A* typically ranges from 10-25 blocks depending on parameters
   - Too low A: High failure rate, many wasted attack attempts
   - Too high A: Excessive opportunity cost per attempt
   - The optimal A* increases with both q and v

4. ECONOMIC IMPLICATIONS:
   - High-value transactions (exchanges, large purchases) are primary targets
   - Standard 6 confirmations provide reasonable security for moderate values
   - For very high values (>1000 BTC), more confirmations may be warranted

5. DEFENSE RECOMMENDATIONS:
   - Merchants should calibrate confirmation requirements to transaction value
   - Rule of thumb: Wait for confirmations until opportunity cost exceeds gain
   - For exchanges: Consider additional verification for large withdrawals
   - Network should maintain decentralization to prevent hash power concentration

6. THEORETICAL INSIGHTS:
   - The abandonment threshold transforms an infinite-variance problem into
     a tractable repeated game
   - Repeated attacks allow even low-probability strategies to be profitable
   - The key metric for attackers is profit RATE, not profit per attack

7. COMPARISON WITH HONEST MINING:
   - For most realistic scenarios (q < 0.40, moderate v), honest mining is
     often more profitable than attacking
   - Attacks only dominate for high hash power AND high-value targets
   - This provides economic incentive for miners to remain honest
""")

    return


def plot_final_summary(save_path: str = None):
    """Create final summary visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    n = 6
    num_sims = 40000

    # Plot 1: Success probability surface
    ax1 = axes[0, 0]
    q_vals = np.linspace(0.1, 0.45, 10)
    A_vals = [5, 10, 15, 20, 30]

    for A in A_vals:
        probs = []
        for q in q_vals:
            attack = RepeatedDoubleSpendingAttack(q, n, A)
            stats = attack.estimate_attack_statistics(num_sims)
            probs.append(stats['success_probability'])
        ax1.plot(q_vals, probs, 'o-', label=f'A={A}')

    ax1.set_xlabel('Attacker Hash Power (q)', fontsize=11)
    ax1.set_ylabel('Success Probability', fontsize=11)
    ax1.set_title('Attack Success Probability vs Hash Power', fontsize=12)
    ax1.legend(title='Abandon Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.1, 0.45])

    # Plot 2: Break-even value
    ax2 = axes[0, 1]
    q_vals = np.linspace(0.15, 0.45, 10)

    for A in [5, 10, 20]:
        break_evens = []
        for q in q_vals:
            stats = calculate_expected_profit_per_attack(q, n, A, 100, num_simulations=num_sims)
            be = min(stats['break_even_value'], 5000)
            break_evens.append(be)
        ax2.plot(q_vals, break_evens, 'o-', label=f'A={A}')

    ax2.set_xlabel('Attacker Hash Power (q)', fontsize=11)
    ax2.set_ylabel('Break-even Value (BTC)', fontsize=11)
    ax2.set_title('Minimum Profitable Double-Spend Value', fontsize=12)
    ax2.legend(title='Abandon Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Profit vs double-spend value
    ax3 = axes[1, 0]
    v_vals = np.linspace(10, 500, 15)

    for q in [0.25, 0.30, 0.35, 0.40]:
        profits = []
        for v in v_vals:
            stats = calculate_expected_profit_per_attack(q, n, 15, v, num_simulations=num_sims)
            profits.append(stats['expected_profit'])
        ax3.plot(v_vals, profits, 'o-', label=f'q={q}')

    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Double-Spend Value (BTC)', fontsize=11)
    ax3.set_ylabel('Expected Profit per Attack (BTC)', fontsize=11)
    ax3.set_title('Expected Profit vs Double-Spend Value (n=6, A=15)', fontsize=12)
    ax3.legend(title='Hash Power')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Attack vs Honest Mining
    ax4 = axes[1, 1]
    q_vals = np.linspace(0.1, 0.45, 12)

    for v in [100, 250, 500]:
        attack_rates = []
        for q in q_vals:
            stats = calculate_expected_profit_per_attack(q, n, 15, v, num_simulations=num_sims)
            rate = stats['expected_profit'] / stats['avg_attack_duration_blocks']
            attack_rates.append(rate)
        ax4.plot(q_vals, attack_rates, 'o-', label=f'Attack v={v}')

    honest_rates = [q * 6.25 for q in q_vals]
    ax4.plot(q_vals, honest_rates, 'k--', linewidth=2, label='Honest Mining')
    ax4.axhline(y=0, color='red', linestyle=':', alpha=0.5)

    ax4.set_xlabel('Hash Power (q)', fontsize=11)
    ax4.set_ylabel('Profit Rate (BTC/block)', fontsize=11)
    ax4.set_title('Attack Profit Rate vs Honest Mining', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Repeated Double-Spending Attack Analysis Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    plt.show()
    return fig


if __name__ == "__main__":
    generate_comprehensive_summary()

    print("\n" + "=" * 80)
    print("GENERATING SUMMARY FIGURES")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_final_summary(
        save_path=os.path.join(script_dir, "conclusions_summary.png")
    )
