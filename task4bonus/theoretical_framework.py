"""
Theoretical Framework for Repeated Double-Spending Attacks

This module provides the mathematical foundations and analytical results
for repeated double-spending attacks in Bitcoin with abandonment threshold.

Mathematical Framework:
- Random walk model for the blockchain race
- Markov chain analysis for success probability
- Expected value calculations for profitability

Author: Crypto Finance Research
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.stats import nbinom
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TheoreticalDoubleSpendAnalysis:
    """
    Theoretical analysis of double-spending attacks using Markov chain theory.

    The blockchain race can be modeled as a random walk:
    - State s = (attacker_blocks - honest_blocks) represents the lead/deficit
    - At each step: s -> s+1 with prob q, s -> s-1 with prob p=1-q
    - Attacker wins if s > 0 after honest chain has n blocks
    - Attacker abandons if s < -A
    """

    def __init__(self, q: float, n: int, A: int):
        """
        Initialize theoretical analysis.

        Args:
            q: Attacker's hash power fraction
            n: Number of confirmations
            A: Abandonment threshold
        """
        self.q = q
        self.p = 1 - q
        self.n = n
        self.A = A

    def catch_up_probability_infinite(self, z: int) -> float:
        """
        Probability of catching up from z blocks behind (no abandonment).
        Classic Gambler's ruin formula.

        P(catch up from z behind) = (q/p)^z if q < p, else 1
        """
        if z <= 0:
            return 1.0
        if self.q >= self.p:
            return 1.0
        return (self.q / self.p) ** z

    def success_probability_nakamoto(self) -> float:
        """
        Nakamoto's original formula for attack success probability.
        This assumes no abandonment (A = infinity).

        From Bitcoin whitepaper, simplified formula.
        """
        if self.q >= 0.5:
            return 1.0

        lambda_param = self.n * (self.q / self.p)
        prob = 1.0

        for k in range(self.n):
            poisson_term = np.exp(-lambda_param) * (lambda_param ** k) / math.factorial(k)
            catch_up = (self.q / self.p) ** (self.n - k)
            prob -= poisson_term * (1 - catch_up)

        return max(0, prob)

    def transition_probability_matrix(self, max_states: int = None) -> np.ndarray:
        """
        Build transition probability matrix for the Markov chain.

        States: -A-1 (absorbing failure), -A, -A+1, ..., -1, 0, 1, ..., max_lead (absorbing success)
        """
        if max_states is None:
            max_states = self.A + self.n + 10

        # States indexed from 0: state i corresponds to lead = i - A - 1
        # State 0 = absorbing failure (lead < -A)
        # State max_states = absorbing success (lead > some threshold)

        num_states = max_states + 2
        P = np.zeros((num_states, num_states))

        # Absorbing states
        P[0, 0] = 1  # Failure absorbing
        P[num_states - 1, num_states - 1] = 1  # Success absorbing

        # Transition probabilities for non-absorbing states
        for i in range(1, num_states - 1):
            lead = i - self.A - 1
            # Move right (attacker finds block): lead increases
            next_right = i + 1
            # Move left (honest finds block): lead decreases
            next_left = i - 1

            if next_left <= 0:
                P[i, 0] = self.p  # Failure
            else:
                P[i, next_left] = self.p

            if next_right >= num_states - 1:
                P[i, num_states - 1] = self.q  # Success
            else:
                P[i, next_right] = self.q

        return P

    def calculate_success_probability_markov(self, initial_lead: int = 0) -> float:
        """
        Calculate success probability using Markov chain absorption probabilities.
        """
        # We need to track during the "waiting phase" and "catching up phase"
        # This is complex because during waiting, honest chain grows deterministically to n

        # Simplified approach: simulate the Markov chain
        # Starting from lead = -n (attacker starts n behind after waiting phase)

        # Actually, let's compute it directly using the Gambler's ruin with boundaries

        # The problem: starting from state z (lead), what's the probability of
        # reaching +1 before reaching -A-1?

        z = initial_lead

        if z > 0:
            return 1.0  # Already winning
        if z < -self.A:
            return 0.0  # Already lost

        if self.q == self.p:
            # Fair random walk
            return (self.A + 1 + z) / (self.A + 2)

        # Biased random walk: use Gambler's ruin formula with two barriers
        # P(reach +1 before -A-1 | start at z) =
        # [(q/p)^(A+1+z) - 1] / [(q/p)^(A+2) - 1]

        r = self.q / self.p

        if abs(r - 1) < 1e-10:
            return (self.A + 1 + z) / (self.A + 2)

        numerator = r ** (self.A + 1 + z) - 1
        denominator = r ** (self.A + 2) - 1

        return numerator / denominator

    def expected_duration_until_absorption(self, initial_lead: int = 0) -> float:
        """
        Calculate expected number of steps until absorption (success or failure).
        Uses Gambler's ruin expected duration formula.
        """
        z = initial_lead

        if z > 0 or z < -self.A:
            return 0

        if self.q == self.p:
            # For fair random walk between -A-1 and +1
            # Starting from z
            a = self.A + 1  # distance to lower barrier
            b = 1 - z  # distance to upper barrier (win at +1, so distance is 1-z)
            # Actually for asymmetric barriers this is more complex
            # E[T] = (a + z)(b - z) for fair walk with barriers at -a and b
            return (self.A + 1 + z) * (1 - z)

        r = self.q / self.p

        # For biased random walk
        # Complex formula - using simulation is more reliable
        return self._simulate_expected_duration(initial_lead)

    def _simulate_expected_duration(self, initial_lead: int, num_sims: int = 50000) -> float:
        """Simulate to estimate expected duration."""
        durations = []

        for _ in range(num_sims):
            lead = initial_lead
            steps = 0
            while -self.A <= lead <= 0:
                if np.random.random() < self.q:
                    lead += 1
                else:
                    lead -= 1
                steps += 1

            durations.append(steps)

        return np.mean(durations)


def analyze_repeated_attack_profitability(
    q: float, n: int, A: int, v: float,
    block_reward: float = 6.25
) -> Dict:
    """
    Complete profitability analysis for repeated attacks.

    Key insight: The attack can be repeated indefinitely.
    Expected profit per unit time matters more than per-attack profit.

    Args:
        q: Hash power fraction
        n: Confirmations required
        A: Abandonment threshold
        v: Double-spend value in BTC
        block_reward: Mining reward

    Returns:
        Comprehensive profitability analysis
    """
    analysis = TheoreticalDoubleSpendAnalysis(q, n, A)

    # Simulate to get accurate statistics
    num_sims = 100000
    successes = 0
    total_attacker_blocks = 0
    total_blocks = 0

    for _ in range(num_sims):
        att_blocks = 0
        hon_blocks = 0

        # Phase 1: Wait for n confirmations
        while hon_blocks < n:
            if np.random.random() < q:
                att_blocks += 1
            else:
                hon_blocks += 1

        # Phase 2: Race to catch up
        while True:
            gap = hon_blocks - att_blocks
            if gap < 0:  # Success
                successes += 1
                break
            if gap > A:  # Failure
                break
            if np.random.random() < q:
                att_blocks += 1
            else:
                hon_blocks += 1

        total_attacker_blocks += att_blocks
        total_blocks += att_blocks + hon_blocks

    success_prob = successes / num_sims
    avg_duration = total_blocks / num_sims
    avg_attacker_blocks = total_attacker_blocks / num_sims

    # Economic analysis
    expected_gain_per_attack = success_prob * v

    # Opportunity cost: what the attacker could have earned mining honestly
    # During attack duration, attacker would have found q * avg_duration blocks
    opportunity_cost_per_attack = q * avg_duration * block_reward

    # Net expected profit per attack
    net_profit_per_attack = expected_gain_per_attack - opportunity_cost_per_attack

    # Profit rate (per block time)
    profit_rate = net_profit_per_attack / avg_duration if avg_duration > 0 else 0

    # Honest mining rate for comparison
    honest_rate = q * block_reward

    # Break-even double-spend value
    if success_prob > 0:
        break_even_value = opportunity_cost_per_attack / success_prob
    else:
        break_even_value = float('inf')

    return {
        'success_probability': success_prob,
        'avg_attack_duration': avg_duration,
        'expected_gain': expected_gain_per_attack,
        'opportunity_cost': opportunity_cost_per_attack,
        'net_profit_per_attack': net_profit_per_attack,
        'profit_rate_per_block': profit_rate,
        'honest_mining_rate': honest_rate,
        'relative_advantage': profit_rate / honest_rate if honest_rate > 0 else 0,
        'break_even_value': break_even_value,
        'is_profitable': net_profit_per_attack > 0
    }


def generate_profitability_heatmap(save_path: str = None):
    """
    Generate heatmap showing when repeated attacks are profitable.
    """
    q_values = np.linspace(0.05, 0.45, 15)
    v_values = np.array([10, 25, 50, 100, 200, 500, 1000, 2000])

    n = 6  # Standard confirmations
    A = 15  # Reasonable abandonment threshold

    profit_matrix = np.zeros((len(v_values), len(q_values)))

    for i, v in enumerate(v_values):
        for j, q in enumerate(q_values):
            result = analyze_repeated_attack_profitability(q, n, A, v)
            profit_matrix[i, j] = result['net_profit_per_attack']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap of net profit
    ax1 = axes[0]
    im1 = ax1.imshow(profit_matrix, aspect='auto', cmap='RdYlGn',
                     extent=[q_values[0], q_values[-1], 0, len(v_values)])

    ax1.set_xlabel('Attacker Hash Power (q)')
    ax1.set_ylabel('Double-Spend Value (BTC)')
    ax1.set_yticks(np.arange(len(v_values)) + 0.5)
    ax1.set_yticklabels([str(v) for v in v_values])
    ax1.set_title(f'Net Expected Profit per Attack (n={n}, A={A})')

    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('E[Profit] (BTC)')

    # Add contour line at profit = 0
    ax1.contour(q_values, np.arange(len(v_values)) + 0.5, profit_matrix,
                levels=[0], colors='black', linewidths=2)

    # Second plot: Profit rate comparison with honest mining
    ax2 = axes[1]

    for v in [50, 100, 500]:
        rates = []
        for q in q_values:
            result = analyze_repeated_attack_profitability(q, n, A, v)
            rates.append(result['profit_rate_per_block'])
        ax2.plot(q_values, rates, marker='o', markersize=4, label=f'v={v} BTC')

    # Honest mining line
    honest_rates = [q * 6.25 for q in q_values]
    ax2.plot(q_values, honest_rates, 'k--', linewidth=2, label='Honest mining')
    ax2.axhline(y=0, color='red', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Attacker Hash Power (q)')
    ax2.set_ylabel('Profit Rate (BTC per block)')
    ax2.set_title('Profit Rate: Attack vs Honest Mining')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()
    return fig


def analyze_optimal_strategy():
    """
    Analyze the optimal attack strategy as a function of double-spend value.
    """
    print("=" * 80)
    print("OPTIMAL STRATEGY ANALYSIS")
    print("=" * 80)

    n = 6  # Standard 6 confirmations

    print("\n1. For what double-spend values is attacking ever profitable?")
    print("-" * 60)

    for q in [0.1, 0.2, 0.3, 0.4]:
        print(f"\nHash power q = {q}:")

        # Find break-even value for different A
        for A in [5, 10, 20, 50]:
            result = analyze_repeated_attack_profitability(q, n, A, 1000)
            print(f"  A={A:2d}: Break-even value = {result['break_even_value']:8.1f} BTC, "
                  f"Success prob = {result['success_probability']:.4f}")

    print("\n2. Optimal abandonment threshold analysis")
    print("-" * 60)

    for q in [0.2, 0.3, 0.4]:
        for v in [100, 500]:
            print(f"\nq={q}, v={v} BTC:")

            best_A = 1
            best_profit = float('-inf')
            best_rate = float('-inf')

            for A in range(1, 51):
                result = analyze_repeated_attack_profitability(q, n, A, v)

                if result['net_profit_per_attack'] > best_profit:
                    best_profit = result['net_profit_per_attack']
                    best_A = A
                    best_rate = result['profit_rate_per_block']

            print(f"  Optimal A = {best_A}, E[Profit]/attack = {best_profit:.2f} BTC, "
                  f"Profit rate = {best_rate:.4f} BTC/block")

            # Compare with honest mining
            honest_rate = q * 6.25
            print(f"  Honest mining rate = {honest_rate:.4f} BTC/block")
            print(f"  Relative advantage = {(best_rate/honest_rate - 1)*100:.1f}%")

    print("\n3. Critical hash power threshold")
    print("-" * 60)
    print("\nMinimum hash power for profitable repeated attacks:")

    for v in [50, 100, 500, 1000]:
        A = 15
        critical_q = None

        for q in np.linspace(0.01, 0.49, 50):
            result = analyze_repeated_attack_profitability(q, n, A, v)
            if result['net_profit_per_attack'] > 0:
                critical_q = q
                break

        if critical_q:
            print(f"  v = {v:4d} BTC: Minimum q = {critical_q:.2f} ({critical_q*100:.0f}%)")
        else:
            print(f"  v = {v:4d} BTC: Attack never profitable for q < 50%")


def main():
    """Main analysis routine."""
    print("=" * 80)
    print("THEORETICAL FRAMEWORK FOR REPEATED DOUBLE-SPENDING ATTACKS")
    print("=" * 80)

    print("""
MATHEMATICAL MODEL
==================

The repeated double-spending attack can be modeled as follows:

1. ATTACK PHASES:
   - Phase 1 (Waiting): Attacker sends payment, waits for n confirmations
   - Phase 2 (Racing): Attacker tries to build longer secret chain
   - Phase 3 (Resolution): Success (publish chain) or Failure (abandon)

2. RANDOM WALK MODEL:
   Let s_t = (attacker_blocks - honest_blocks) at time t

   The process {s_t} is a random walk with:
   - P(s_t+1 = s_t + 1) = q  (attacker finds block)
   - P(s_t+1 = s_t - 1) = p = 1-q  (honest network finds block)

3. ABANDONMENT STRATEGY:
   - Attacker abandons if s_t < -A (falls A blocks behind)
   - This converts infinite variance random walk to finite expected duration

4. SUCCESS CONDITION:
   - Attack succeeds if attacker's chain becomes longer (s > 0)
   - After merchant releases goods (n confirmations seen)

5. KEY QUANTITIES:
   - P_success(q, n, A) = probability of successful attack
   - E[T](q, n, A) = expected duration of one attack attempt
   - E[Profit] = P_success × v - E[T] × q × block_reward

""")

    # Run detailed analysis
    analyze_optimal_strategy()

    # Generate visualization
    print("\n" + "=" * 80)
    print("GENERATING PROFITABILITY HEATMAP")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    generate_profitability_heatmap(
        save_path=os.path.join(script_dir, "profitability_heatmap.png")
    )


if __name__ == "__main__":
    main()
