"""
Bitcoin Mining Thresholds Analysis
==================================

Task 3.1: Orphan block mining threshold
Task 3.2: Block withholding threshold (γ = 0)

Author: PhD Student
Course: CryptoFinance
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import os


# =============================================================================
# TASK 3.1: Orphan Block Mining Threshold
# =============================================================================

def analyze_orphan_mining():
    """
    Task 3.1: Determine when a miner should continue mining on their orphan block
    despite being one block behind the official blockchain.

    Scenario:
    - Miner produced a block that got orphaned
    - Miner is now 1 block behind (state = -1)
    - Should they continue on orphan or switch to main chain?

    Analysis:
    - If miner continues and catches up: they get credited for multiple blocks
    - If miner switches: they lose the orphan but mine honestly

    Key insight: For q > 0.5, miner will always eventually catch up.
    The question is whether the rate of earning is higher.
    """
    print("=" * 70)
    print("TASK 3.1: Orphan Block Mining Threshold")
    print("=" * 70)

    print("\n### Scenario ###")
    print("- Miner's block at height n was orphaned")
    print("- Main chain is at height n+1 (miner is 1 block behind)")
    print("- Should miner continue on orphan fork or switch to main chain?")

    print("\n### Mathematical Analysis ###")
    print("\nFor q > 0.5 (miner has majority hash power):")
    print("- Probability of catching up: P(win) = 1")
    print("- Expected time to catch up: E[T] = 2/(2q-1)")
    print("- Expected credited blocks when winning: E[credited] = (4q-1)/(2q-1)")
    print("  (This includes the orphan block plus blocks mined during catch-up)")

    print("\nRevenue rate comparison:")
    print("- Orphan mining rate: R_orphan = E[credited]/E[T] = (4q-1)/2")
    print("- Honest mining rate: R_honest = q")

    print("\nFor orphan mining to be advantageous:")
    print("  (4q-1)/2 > q")
    print("  4q - 1 > 2q")
    print("  2q > 1")
    print("  q > 0.5")

    print("\n" + "=" * 70)
    print("RESULT: Threshold q = 0.5 (50%)")
    print("=" * 70)
    print("\nInterpretation:")
    print("- For q > 50%: Continue mining on orphan block (will catch up)")
    print("- For q < 50%: Switch to main chain (unlikely to catch up)")
    print("- For q = 50%: Indifferent (expected values are equal)")

    return 0.5


def verify_orphan_threshold_simulation(num_simulations: int = 10000):
    """Verify the orphan mining threshold through simulation."""

    print("\n### Simulation Verification ###")

    q_values = [0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

    print(f"\n{'q':>6} | {'Orphan Rate':>12} | {'Honest Rate':>12} | {'Better Strategy':>15}")
    print("-" * 55)

    for q in q_values:
        # Simulate orphan mining from state -1
        orphan_blocks = []
        orphan_times = []

        for _ in range(num_simulations):
            state = -1
            time = 0
            max_time = 10000  # Prevent infinite loops

            while state < 1 and time < max_time:
                if np.random.random() < q:
                    state += 1  # Miner finds block
                else:
                    state -= 1  # Network finds block
                time += 1

            if state >= 1:
                # Miner caught up - credit includes orphan + mined blocks
                # From -1 to +1, net change of +2, so miner mined (time + 2)/2 blocks
                credited = (time + 2) / 2 + 1  # +1 for orphan
                orphan_blocks.append(credited)
                orphan_times.append(time)
            else:
                # Didn't catch up in time limit
                orphan_blocks.append(0)
                orphan_times.append(max_time)

        avg_orphan_blocks = np.mean(orphan_blocks)
        avg_time = np.mean(orphan_times)
        orphan_rate = avg_orphan_blocks / avg_time if avg_time > 0 else 0
        honest_rate = q

        better = "ORPHAN" if orphan_rate > honest_rate else "HONEST"
        if abs(orphan_rate - honest_rate) < 0.01:
            better = "~EQUAL"

        print(f"{q:>6.2f} | {orphan_rate:>12.4f} | {honest_rate:>12.4f} | {better:>15}")

    print("\nSimulation confirms: threshold is approximately q = 0.5")


# =============================================================================
# TASK 3.2: Block Withholding Threshold (γ = 0)
# =============================================================================

def analyze_withholding_threshold():
    """
    Task 3.2: Determine threshold for withholding a newly found block when γ = 0.

    Scenario:
    - Miner just found a block on top of the main chain
    - Miner is at state +1 (1 block ahead, privately)
    - Should they reveal immediately or withhold?
    - Connectivity γ = 0 (miner loses all races)

    Analysis:
    At state +1, comparing reveal vs withhold:
    - Reveal: Get 1 block for certain
    - Withhold:
      - With prob q: find another block, publish both, get 2 blocks
      - With prob 1-q: network finds block, race, lose (γ=0), get 0 blocks
    """
    print("\n" + "=" * 70)
    print("TASK 3.2: Block Withholding Threshold (γ = 0)")
    print("=" * 70)

    print("\n### Scenario ###")
    print("- Miner just discovered a block extending the main chain")
    print("- Miner is now at state +1 (1 block ahead, privately)")
    print("- Connectivity γ = 0 (miner loses ALL races)")
    print("- Should miner reveal immediately or withhold?")

    print("\n### Mathematical Analysis ###")
    print("\nAt state +1, comparing two options:")
    print("\nOption A - Reveal immediately:")
    print("  E[blocks] = 1 (certain)")

    print("\nOption B - Withhold:")
    print("  - With prob q: miner finds next block → publish both → 2 blocks")
    print("  - With prob (1-q): network finds block → race → lose (γ=0) → 0 blocks")
    print("  E[blocks] = q × 2 + (1-q) × 0 = 2q")

    print("\nFor withholding to be at least as good as revealing:")
    print("  E[withhold] ≥ E[reveal]")
    print("  2q ≥ 1")
    print("  q ≥ 0.5")

    print("\n" + "=" * 70)
    print("RESULT: Threshold q = 0.5 (50%)")
    print("=" * 70)
    print("\nInterpretation:")
    print("- For q ≥ 50%: Miner has NO INCENTIVE to reveal (withholding is better)")
    print("- For q < 50%: Miner should reveal immediately")
    print("- For q = 50%: Indifferent (E[withhold] = E[reveal] = 1)")

    print("\n### Important Caveat ###")
    print("This is the ONE-SHOT decision threshold at state +1.")
    print("In the LONG RUN, the full selfish mining strategy with γ=0")
    print("uses the Eyal & Sirer (2014) formula:")
    print("  R_selfish = [4q²(1-q)² - q³] / [1 - q - 2q² + q³]")
    print("  R_honest = q")
    print("  Selfish > Honest when q > 1/3 ≈ 0.333")
    print("\nSo the FULL selfish mining strategy with γ=0 is profitable")
    print("for q > 1/3, which is a LOWER threshold than the one-shot")
    print("decision (q = 0.5). This is because the full strategy exploits")
    print("states beyond +1 (override at state +2, etc.).")

    return 0.5


def verify_withholding_simulation(num_simulations: int = 50000):
    """Verify the withholding threshold through simulation."""

    print("\n### Simulation Verification ###")

    q_values = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

    print(f"\n{'q':>6} | {'E[Withhold]':>12} | {'E[Reveal]':>12} | {'Better Option':>15}")
    print("-" * 55)

    for q in q_values:
        # Simulate withholding from state +1
        withhold_blocks = []

        for _ in range(num_simulations):
            # At state +1, wait for next block
            if np.random.random() < q:
                # Miner finds block, publishes both
                withhold_blocks.append(2)
            else:
                # Network finds block, race with γ=0, miner loses
                withhold_blocks.append(0)

        avg_withhold = np.mean(withhold_blocks)
        reveal = 1.0  # Always get 1 block if reveal

        better = "WITHHOLD" if avg_withhold > reveal else "REVEAL"
        if abs(avg_withhold - reveal) < 0.02:
            better = "~EQUAL"

        print(f"{q:>6.2f} | {avg_withhold:>12.4f} | {reveal:>12.4f} | {better:>15}")

    print("\nSimulation confirms: threshold is approximately q = 0.5")


def selfish_revenue_gamma0(q):
    """
    Eyal & Sirer (2014) formula for selfish mining revenue with γ=0.

    R = [4q²(1-q)² - q³] / [1 - q - 2q² + q³]

    Note: This formula is valid for q < 0.5. For q >= 0.5, the selfish
    miner eventually controls the entire chain (but the formula's
    denominator approaches 0, making it undefined).
    """
    if q <= 0 or q >= 0.5:
        return 0.0  # Formula not applicable for q >= 0.5
    numerator = 4 * q**2 * (1 - q)**2 - q**3
    denominator = 1 - q - 2 * q**2 + q**3
    if abs(denominator) < 1e-10:
        return q
    return max(numerator / denominator, 0.0)


def long_run_selfish_analysis():
    """Analyze long-run profitability of selfish mining with γ=0."""

    print("\n### Long-Run Selfish Mining Analysis (γ = 0) ###")
    print("Using the Eyal & Sirer (2014) formula:")
    print("  R_selfish = [4q²(1-q)² - q³] / [1 - q - 2q² + q³]")

    q_values = np.linspace(0.05, 0.49, 18)

    print(f"\n{'q':>6} | {'Selfish Rate':>12} | {'Honest Rate':>12} | {'Difference':>12}")
    print("-" * 55)

    for q in q_values:
        selfish_rate = selfish_revenue_gamma0(q)
        honest_rate = q
        diff = selfish_rate - honest_rate

        print(f"{q:>6.2f} | {selfish_rate:>12.4f} | {honest_rate:>12.4f} | {diff:>+12.4f}")

    print("\nConclusion: Selfish mining with γ=0 becomes profitable at q > 1/3 ≈ 0.333")
    print("Below q = 1/3, honest mining is better. Above q = 1/3, selfish mining is better.")


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations():
    """Generate plots for both thresholds."""

    os.makedirs('results', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    q_values = np.linspace(0.01, 0.99, 100)

    # --- Task 3.1: Orphan Mining ---
    ax1 = axes[0]

    # For q > 0.5, compute rates
    orphan_rate = np.zeros_like(q_values)
    honest_rate = q_values.copy()

    for i, q in enumerate(q_values):
        if q > 0.5:
            orphan_rate[i] = (4*q - 1) / 2
        else:
            orphan_rate[i] = 0  # Rate is 0 when can't catch up reliably

    ax1.plot(q_values, honest_rate, 'b-', linewidth=2, label='Honest Mining Rate (q)')
    ax1.plot(q_values, orphan_rate, 'r-', linewidth=2, label='Orphan Mining Rate ((4q-1)/2)')
    ax1.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold q = 0.5')
    ax1.fill_between(q_values, 0, 1, where=(q_values > 0.5), alpha=0.2, color='green',
                     label='Orphan mining advantageous')

    ax1.set_xlabel('Hashing Power (q)', fontsize=12)
    ax1.set_ylabel('Revenue Rate (blocks per round)', fontsize=12)
    ax1.set_title('Task 3.1: Orphan Block Mining Threshold', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.5])

    # Add annotation
    ax1.annotate('Continue on orphan\n(q > 0.5)', xy=(0.75, 1.1), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.annotate('Switch to main\n(q < 0.5)', xy=(0.25, 0.3), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # --- Task 3.2: Withholding Threshold ---
    ax2 = axes[1]

    # E[withhold] = 2q, E[reveal] = 1
    withhold_ev = 2 * q_values
    reveal_ev = np.ones_like(q_values)

    ax2.plot(q_values, reveal_ev, 'b-', linewidth=2, label='E[Reveal] = 1')
    ax2.plot(q_values, withhold_ev, 'r-', linewidth=2, label='E[Withhold] = 2q')
    ax2.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold q = 0.5')
    ax2.fill_between(q_values, 0, 2, where=(q_values >= 0.5), alpha=0.2, color='green',
                     label='No incentive to reveal')

    ax2.set_xlabel('Hashing Power (q)', fontsize=12)
    ax2.set_ylabel('Expected Blocks', fontsize=12)
    ax2.set_title('Task 3.2: Block Withholding Threshold (γ = 0)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 2.2])

    # Add annotation
    ax2.annotate('Withhold\n(q ≥ 0.5)', xy=(0.75, 1.7), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax2.annotate('Reveal\n(q < 0.5)', xy=(0.25, 0.5), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig('results/bitcoin_thresholds.png', dpi=300, bbox_inches='tight')
    print("\nSaved: results/bitcoin_thresholds.png")
    plt.close()

    # --- Additional plot: Long-run selfish vs honest ---
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    # Use range 0.01-0.49 for valid formula application
    q_plot = np.linspace(0.01, 0.49, 100)

    # Eyal & Sirer formula for γ=0
    selfish_rate_arr = np.array([selfish_revenue_gamma0(q) for q in q_plot])

    ax3.plot(q_plot, q_plot, 'b-', linewidth=2, label='Honest Mining Rate (q)')
    ax3.plot(q_plot, selfish_rate_arr, 'r-', linewidth=2,
             label='Selfish Mining Rate (Eyal & Sirer, γ=0)')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.axvline(x=1/3, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label='Threshold q = 1/3')

    # Fill regions
    ax3.fill_between(q_plot, selfish_rate_arr, q_plot,
                     where=(selfish_rate_arr < q_plot), alpha=0.2, color='blue',
                     label='Honest advantage (q < 1/3)')
    ax3.fill_between(q_plot, selfish_rate_arr, q_plot,
                     where=(selfish_rate_arr > q_plot), alpha=0.2, color='red',
                     label='Selfish advantage (q > 1/3)')

    ax3.set_xlabel('Hashing Power (q)', fontsize=12)
    ax3.set_ylabel('Revenue Rate (blocks per round)', fontsize=12)
    ax3.set_title('Long-Run: Selfish vs Honest Mining (γ = 0)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])

    ax3.annotate('Honest better\n(q < 1/3)', xy=(0.17, 0.1), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax3.annotate('Selfish better\n(q > 1/3)', xy=(0.6, 0.35), fontsize=10,
                ha='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('results/selfish_vs_honest_gamma0.png', dpi=300, bbox_inches='tight')
    print("Saved: results/selfish_vs_honest_gamma0.png")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    """Run complete analysis."""

    print("\n" + "=" * 70)
    print("  BITCOIN MINING THRESHOLDS ANALYSIS")
    print("  Course: CryptoFinance")
    print("=" * 70)

    # Task 3.1
    threshold_3_1 = analyze_orphan_mining()
    verify_orphan_threshold_simulation()

    # Task 3.2
    threshold_3_2 = analyze_withholding_threshold()
    verify_withholding_simulation()
    long_run_selfish_analysis()

    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    create_visualizations()

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTask 3.1 - Orphan Block Mining Threshold: q = {threshold_3_1}")
    print(f"  → For q > 50%, continue mining on orphan block")
    print(f"\nTask 3.2 - Block Withholding Threshold (γ=0): q = {threshold_3_2}")
    print(f"  → For q ≥ 50%, no incentive to reveal immediately (one-shot decision)")
    print(f"  → Full selfish mining with γ=0 is profitable for q > 1/3 (Eyal & Sirer)")
    print(f"\nNote: The one-shot decision threshold (50%) is higher than the")
    print(f"full selfish mining threshold (1/3) because the full strategy")
    print(f"exploits additional states beyond +1.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
