"""
Repeated Double-Spending Attack Analysis in Bitcoin

This module analyzes the profitability and dynamics of repeated double-spending attacks
where an attacker uses an abandonment threshold strategy.

Framework:
- Attacker has hash power fraction q (honest network has p = 1-q)
- Merchant requires n confirmations before releasing goods
- Attacker abandons attack if they fall more than A blocks behind
- Attack is repeated indefinitely with fresh attempts

Author: Crypto Finance Research
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


class RepeatedDoubleSpendingAttack:
    """
    Framework for analyzing repeated double-spending attacks with abandonment threshold.

    The attacker:
    1. Sends payment to merchant and starts mining secretly from the block before the transaction
    2. Waits for merchant to see n confirmations on honest chain
    3. Tries to catch up and surpass the honest chain
    4. Abandons if falling more than A blocks behind the honest chain tip
    5. If successful, publishes secret chain to double-spend; otherwise restarts
    """

    def __init__(self, q: float, n: int, A: int):
        """
        Initialize attack parameters.

        Args:
            q: Attacker's hash power fraction (0 < q < 0.5 for interesting cases)
            n: Number of confirmations merchant waits for
            A: Abandonment threshold (attacker gives up if A blocks behind)
        """
        self.q = q
        self.p = 1 - q
        self.n = n
        self.A = A

    def simulate_single_attack(self) -> Tuple[bool, int, int]:
        """
        Simulate a single double-spending attack attempt.

        Returns:
            (success, attacker_blocks_mined, honest_blocks_mined)
        """
        # Phase 1: While merchant waits for n confirmations
        # Attacker mines secretly, honest network mines on public chain

        attacker_blocks = 0
        honest_blocks = 0

        # Mine until honest chain has n blocks (confirmations)
        while honest_blocks < self.n:
            if np.random.random() < self.q:
                attacker_blocks += 1
            else:
                honest_blocks += 1

        # Phase 2: Race to catch up
        # Attacker needs to have more blocks than honest chain
        # The gap is: honest_blocks - attacker_blocks

        while True:
            gap = honest_blocks - attacker_blocks

            # Success: attacker has longer chain
            if gap < 0:
                return True, attacker_blocks, honest_blocks

            # Failure: gap exceeds abandonment threshold
            if gap > self.A:
                return False, attacker_blocks, honest_blocks

            # Continue mining
            if np.random.random() < self.q:
                attacker_blocks += 1
            else:
                honest_blocks += 1

    def estimate_attack_statistics(self, num_simulations: int = 100000) -> Dict:
        """
        Estimate attack statistics through Monte Carlo simulation.

        Returns:
            Dictionary with success probability, expected blocks mined, etc.
        """
        successes = 0
        total_attacker_blocks = 0
        total_honest_blocks = 0
        attack_durations = []

        for _ in range(num_simulations):
            success, att_blocks, hon_blocks = self.simulate_single_attack()
            if success:
                successes += 1
            total_attacker_blocks += att_blocks
            total_honest_blocks += hon_blocks
            attack_durations.append(att_blocks + hon_blocks)

        success_prob = successes / num_simulations
        avg_attacker_blocks = total_attacker_blocks / num_simulations
        avg_honest_blocks = total_honest_blocks / num_simulations
        avg_duration = np.mean(attack_durations)

        return {
            'success_probability': success_prob,
            'avg_attacker_blocks': avg_attacker_blocks,
            'avg_honest_blocks': avg_honest_blocks,
            'avg_total_blocks': avg_duration,
            'std_duration': np.std(attack_durations)
        }

    def theoretical_success_probability_no_abandon(self) -> float:
        """
        Calculate theoretical success probability without abandonment (A = infinity).
        Based on Nakamoto's original analysis.
        """
        if self.q >= 0.5:
            return 1.0

        # Probability attacker catches up from z blocks behind
        def catch_up_prob(z):
            return (self.q / self.p) ** z

        # Sum over Poisson distribution of attacker's head start
        lambda_param = self.n * (self.q / self.p)
        prob = 0

        for k in range(self.n):
            # Probability honest mined n blocks while attacker mined k
            poisson_prob = stats.poisson.pmf(k, lambda_param * self.p / self.q)
            # Actually use negative binomial
            pass

        # Simpler formula from Nakamoto paper
        prob = 1.0
        for k in range(self.n):
            lambda_val = self.n * self.q / self.p
            poisson_term = np.exp(-lambda_val) * (lambda_val ** k) / math.factorial(k)
            prob -= poisson_term * (1 - (self.q / self.p) ** (self.n - k))

        return max(0, prob)


def calculate_expected_profit_per_attack(
    q: float, n: int, A: int,
    double_spend_value: float,
    block_reward: float = 6.25,
    include_opportunity_cost: bool = True,
    num_simulations: int = 50000
) -> Dict:
    """
    Calculate expected profit per attack attempt.

    Args:
        q: Attacker's hash power fraction
        n: Number of confirmations
        A: Abandonment threshold
        double_spend_value: Value of goods obtained through double-spend (in BTC)
        block_reward: Current block reward (BTC)
        include_opportunity_cost: Whether to include foregone honest mining rewards

    Returns:
        Dictionary with profit analysis
    """
    attack = RepeatedDoubleSpendingAttack(q, n, A)
    stats = attack.estimate_attack_statistics(num_simulations)

    # Expected gain from successful double-spend
    expected_gain = stats['success_probability'] * double_spend_value

    # Opportunity cost: blocks attacker could have mined honestly
    # During attack, attacker mines avg_attacker_blocks secretly
    # Expected honest blocks attacker would have found = q * avg_total_blocks
    expected_honest_reward = q * stats['avg_total_blocks'] * block_reward

    if include_opportunity_cost:
        expected_profit = expected_gain - expected_honest_reward
    else:
        expected_profit = expected_gain

    # Also consider: on success, attacker gets their secretly mined blocks
    # On failure, those blocks are wasted
    # This is implicitly captured in opportunity cost

    return {
        'success_probability': stats['success_probability'],
        'expected_gain': expected_gain,
        'opportunity_cost': expected_honest_reward,
        'expected_profit': expected_profit,
        'avg_attack_duration_blocks': stats['avg_total_blocks'],
        'break_even_value': expected_honest_reward / stats['success_probability'] if stats['success_probability'] > 0 else float('inf')
    }


def analyze_optimal_abandonment_threshold(
    q: float, n: int, double_spend_value: float,
    max_A: int = 50, num_simulations: int = 30000
) -> Tuple[int, Dict]:
    """
    Find the optimal abandonment threshold that maximizes expected profit.
    """
    best_A = 1
    best_profit = float('-inf')
    results = {}

    for A in range(1, max_A + 1):
        profit_stats = calculate_expected_profit_per_attack(
            q, n, A, double_spend_value, num_simulations=num_simulations
        )
        results[A] = profit_stats

        if profit_stats['expected_profit'] > best_profit:
            best_profit = profit_stats['expected_profit']
            best_A = A

    return best_A, results


def plot_attack_analysis(q_values: List[float], n: int = 6, A: int = 10,
                         double_spend_values: List[float] = None,
                         save_path: str = None):
    """
    Create comprehensive visualization of attack analysis.
    """
    if double_spend_values is None:
        double_spend_values = [10, 50, 100, 500, 1000]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: Success probability vs hash power
    ax1 = axes[0, 0]
    A_values = [5, 10, 20, 50]
    for A in A_values:
        probs = []
        for q in q_values:
            attack = RepeatedDoubleSpendingAttack(q, n, A)
            stats = attack.estimate_attack_statistics(20000)
            probs.append(stats['success_probability'])
        ax1.plot(q_values, probs, label=f'A={A}', marker='o', markersize=3)

    ax1.set_xlabel('Attacker Hash Power (q)')
    ax1.set_ylabel('Success Probability')
    ax1.set_title(f'Attack Success Probability (n={n} confirmations)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.5])

    # Plot 2: Expected profit vs hash power for different double-spend values
    ax2 = axes[0, 1]
    A = 10
    for v in [50, 100, 500]:
        profits = []
        for q in q_values:
            if q > 0:
                profit_stats = calculate_expected_profit_per_attack(
                    q, n, A, v, num_simulations=15000
                )
                profits.append(profit_stats['expected_profit'])
            else:
                profits.append(0)
        ax2.plot(q_values, profits, label=f'v={v} BTC', marker='o', markersize=3)

    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Attacker Hash Power (q)')
    ax2.set_ylabel('Expected Profit per Attack (BTC)')
    ax2.set_title(f'Expected Profit (n={n}, A={A})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Break-even double-spend value vs hash power
    ax3 = axes[0, 2]
    A_values = [5, 10, 20]
    for A in A_values:
        break_evens = []
        for q in q_values:
            if q > 0:
                profit_stats = calculate_expected_profit_per_attack(
                    q, n, A, 100, num_simulations=15000
                )
                break_evens.append(profit_stats['break_even_value'])
            else:
                break_evens.append(float('inf'))
        # Cap for visualization
        break_evens = [min(b, 10000) for b in break_evens]
        ax3.plot(q_values, break_evens, label=f'A={A}', marker='o', markersize=3)

    ax3.set_xlabel('Attacker Hash Power (q)')
    ax3.set_ylabel('Break-even Value (BTC)')
    ax3.set_title(f'Minimum Profitable Double-Spend Value (n={n})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Effect of confirmations on success probability
    ax4 = axes[1, 0]
    n_values = range(1, 13)
    q_test_values = [0.1, 0.2, 0.3, 0.4]
    A = 15

    for q in q_test_values:
        probs = []
        for n_conf in n_values:
            attack = RepeatedDoubleSpendingAttack(q, n_conf, A)
            stats = attack.estimate_attack_statistics(15000)
            probs.append(stats['success_probability'])
        ax4.plot(n_values, probs, label=f'q={q}', marker='o', markersize=4)

    ax4.set_xlabel('Number of Confirmations (n)')
    ax4.set_ylabel('Success Probability')
    ax4.set_title(f'Effect of Confirmations (A={A})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Optimal abandonment threshold
    ax5 = axes[1, 1]
    q = 0.3
    n = 6
    A_range = range(1, 31)

    for v in [50, 100, 200]:
        profits = []
        for A in A_range:
            profit_stats = calculate_expected_profit_per_attack(
                q, n, A, v, num_simulations=10000
            )
            profits.append(profit_stats['expected_profit'])
        ax5.plot(A_range, profits, label=f'v={v} BTC', marker='o', markersize=3)

    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Abandonment Threshold (A)')
    ax5.set_ylabel('Expected Profit per Attack (BTC)')
    ax5.set_title(f'Optimal Abandonment Threshold (q={q}, n={n})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Long-term profitability (attacks per time unit)
    ax6 = axes[1, 2]
    q_values_profit = np.linspace(0.05, 0.45, 15)
    n = 6
    v = 100

    profit_rates = []
    success_rates = []

    for q in q_values_profit:
        profit_stats = calculate_expected_profit_per_attack(
            q, n, 15, v, num_simulations=15000
        )
        # Profit rate = profit per attack / expected duration
        if profit_stats['avg_attack_duration_blocks'] > 0:
            profit_rate = profit_stats['expected_profit'] / profit_stats['avg_attack_duration_blocks']
        else:
            profit_rate = 0
        profit_rates.append(profit_rate)
        success_rates.append(profit_stats['success_probability'] / profit_stats['avg_attack_duration_blocks'])

    ax6.plot(q_values_profit, profit_rates, 'b-o', markersize=4, label='Profit Rate')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Attacker Hash Power (q)')
    ax6.set_ylabel('Expected Profit Rate (BTC/block)')
    ax6.set_title(f'Long-term Profit Rate (v={v} BTC, n={n})')
    ax6.grid(True, alpha=0.3)

    # Add honest mining comparison
    honest_profit_rate = [q * 6.25 for q in q_values_profit]  # Expected honest reward
    ax6_twin = ax6.twinx()
    ax6_twin.plot(q_values_profit, honest_profit_rate, 'g--', alpha=0.7, label='Honest Mining')
    ax6_twin.set_ylabel('Honest Mining Rate (BTC/block)', color='green')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig


def run_comprehensive_analysis():
    """
    Run comprehensive analysis of repeated double-spending attacks.
    """
    print("=" * 80)
    print("REPEATED DOUBLE-SPENDING ATTACK ANALYSIS")
    print("=" * 80)

    # Analysis parameters
    n = 6  # Standard Bitcoin confirmation requirement
    block_reward = 6.25  # Current BTC block reward

    print("\n" + "=" * 80)
    print("1. SUCCESS PROBABILITY ANALYSIS")
    print("=" * 80)

    print(f"\nFixed parameters: n={n} confirmations")
    print("\nSuccess probability for different (q, A) combinations:")
    print("-" * 60)
    print(f"{'q':>8} | {'A=5':>10} | {'A=10':>10} | {'A=20':>10} | {'A=∞':>10}")
    print("-" * 60)

    for q in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        probs = []
        for A in [5, 10, 20, 100]:  # A=100 approximates infinity
            attack = RepeatedDoubleSpendingAttack(q, n, A)
            stats = attack.estimate_attack_statistics(50000)
            probs.append(stats['success_probability'])
        print(f"{q:>8.2f} | {probs[0]:>10.4f} | {probs[1]:>10.4f} | {probs[2]:>10.4f} | {probs[3]:>10.4f}")

    print("\n" + "=" * 80)
    print("2. PROFITABILITY ANALYSIS")
    print("=" * 80)

    print("\nFor q=0.3 (30% hash power), n=6 confirmations:")
    print("\nDouble-spend value vs Expected profit (A=10):")
    print("-" * 70)
    print(f"{'Value (BTC)':>12} | {'P(success)':>10} | {'E[Gain]':>10} | {'Opp.Cost':>10} | {'E[Profit]':>10}")
    print("-" * 70)

    q = 0.3
    A = 10
    for v in [10, 25, 50, 100, 200, 500, 1000]:
        profit_stats = calculate_expected_profit_per_attack(q, n, A, v, num_simulations=50000)
        print(f"{v:>12.1f} | {profit_stats['success_probability']:>10.4f} | "
              f"{profit_stats['expected_gain']:>10.2f} | {profit_stats['opportunity_cost']:>10.2f} | "
              f"{profit_stats['expected_profit']:>10.2f}")

    print("\n" + "=" * 80)
    print("3. BREAK-EVEN ANALYSIS")
    print("=" * 80)

    print("\nMinimum double-spend value for profitable attack:")
    print("-" * 50)
    print(f"{'q':>8} | {'A=5':>10} | {'A=10':>10} | {'A=20':>10}")
    print("-" * 50)

    for q in [0.1, 0.2, 0.3, 0.4]:
        break_evens = []
        for A in [5, 10, 20]:
            profit_stats = calculate_expected_profit_per_attack(q, n, A, 100, num_simulations=50000)
            break_evens.append(profit_stats['break_even_value'])
        print(f"{q:>8.2f} | {break_evens[0]:>10.1f} | {break_evens[1]:>10.1f} | {break_evens[2]:>10.1f}")

    print("\n" + "=" * 80)
    print("4. OPTIMAL ABANDONMENT THRESHOLD")
    print("=" * 80)

    print("\nFinding optimal A for different (q, v) combinations:")

    for q in [0.2, 0.3, 0.4]:
        for v in [100, 500]:
            best_A, results = analyze_optimal_abandonment_threshold(
                q, n, v, max_A=30, num_simulations=20000
            )
            best_profit = results[best_A]['expected_profit']
            print(f"\nq={q}, v={v} BTC: Optimal A = {best_A}, E[Profit] = {best_profit:.2f} BTC")

    print("\n" + "=" * 80)
    print("5. EFFECT OF CONFIRMATION COUNT")
    print("=" * 80)

    print("\nSuccess probability vs number of confirmations (q=0.3, A=15):")
    print("-" * 40)

    q = 0.3
    A = 15
    for n_conf in [1, 2, 3, 6, 10, 12]:
        attack = RepeatedDoubleSpendingAttack(q, n_conf, A)
        stats = attack.estimate_attack_statistics(50000)
        print(f"n={n_conf:>2}: P(success) = {stats['success_probability']:.4f}")

    print("\n" + "=" * 80)
    print("6. REPEATED ATTACK LONG-TERM ANALYSIS")
    print("=" * 80)

    print("\nExpected outcomes over 1000 attack attempts:")
    print("(q=0.3, n=6, A=10, v=100 BTC)")

    q, n, A, v = 0.3, 6, 10, 100
    num_attacks = 1000

    attack = RepeatedDoubleSpendingAttack(q, n, A)

    total_successes = 0
    total_attacker_blocks = 0
    total_honest_blocks = 0

    for _ in range(num_attacks):
        success, att_blocks, hon_blocks = attack.simulate_single_attack()
        if success:
            total_successes += 1
        total_attacker_blocks += att_blocks
        total_honest_blocks += hon_blocks

    total_blocks = total_attacker_blocks + total_honest_blocks

    # Gains and costs
    total_double_spend_gain = total_successes * v
    opportunity_cost = q * total_blocks * block_reward
    net_profit = total_double_spend_gain - opportunity_cost

    # What honest mining would have yielded
    honest_mining_reward = q * total_blocks * block_reward

    print(f"\nTotal attack attempts: {num_attacks}")
    print(f"Successful attacks: {total_successes}")
    print(f"Success rate: {total_successes/num_attacks:.2%}")
    print(f"\nTotal blocks during attacks: {total_blocks}")
    print(f"Total double-spend gains: {total_double_spend_gain:.2f} BTC")
    print(f"Opportunity cost (foregone honest mining): {opportunity_cost:.2f} BTC")
    print(f"Net profit from attacking: {net_profit:.2f} BTC")
    print(f"\nComparison: Honest mining would have earned: {honest_mining_reward:.2f} BTC")
    print(f"Attack strategy advantage: {net_profit:.2f} BTC")

    return


if __name__ == "__main__":
    print("\nRunning Repeated Double-Spending Attack Analysis...")
    print("This analysis examines the profitability of repeated double-spending")
    print("attacks with an abandonment threshold strategy.\n")

    run_comprehensive_analysis()

    # Generate visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    q_values = np.linspace(0.05, 0.45, 15)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig = plot_attack_analysis(
        q_values, n=6, A=10,
        save_path=os.path.join(script_dir, "repeated_double_spending_analysis.png")
    )
