"""
Mining Strategies Implementation for Bitcoin
============================================

This module implements various Bitcoin mining strategies:
1. Honest Mining (Strategy 1)
2. Selfish Mining (Strategy 2)
3. Optimal Selfish Mining

Author: PhD Student
Course: CryptoFinance
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class MiningResult:
    """Data class to store mining simulation results."""
    revenue: float
    blocks_mined: int
    blocks_accepted: int
    total_blocks: int
    relative_revenue: float


class HonestMining:
    """
    Implementation of honest mining strategy.

    In honest mining, miners immediately publish blocks when found and
    always build on the longest chain they observe.
    """

    def __init__(self, q: float):
        """
        Initialize honest miner.

        Args:
            q: Hashing power (fraction of total network, 0 < q < 1)
        """
        self.q = q

    def theoretical_revenue(self) -> float:
        """
        Calculate theoretical expected revenue.

        For honest mining, the expected revenue equals the hashing power.

        Returns:
            Expected relative revenue (fraction of all blocks)
        """
        return self.q

    def simulate(self, num_blocks: int = 10000, seed: int = None) -> MiningResult:
        """
        Simulate honest mining strategy.

        Args:
            num_blocks: Total number of blocks to mine in the network
            seed: Random seed for reproducibility

        Returns:
            MiningResult object with simulation statistics
        """
        if seed is not None:
            np.random.seed(seed)

        # Each block is mined by our miner with probability q
        blocks_mined = np.random.binomial(num_blocks, self.q)

        # In honest mining, all mined blocks are accepted
        blocks_accepted = blocks_mined

        relative_revenue = blocks_accepted / num_blocks

        return MiningResult(
            revenue=blocks_accepted,
            blocks_mined=blocks_mined,
            blocks_accepted=blocks_accepted,
            total_blocks=num_blocks,
            relative_revenue=relative_revenue
        )


class SelfishMining:
    """
    Implementation of selfish mining strategy.

    Selfish miners withhold blocks to create private forks, attempting to
    waste honest miners' work and capture more than their fair share of rewards.

    Reference: Eyal & Sirer (2014) "Majority is not Enough: Bitcoin Mining is Vulnerable"
    """

    def __init__(self, q: float, gamma: float):
        """
        Initialize selfish miner.

        Args:
            q: Hashing power (fraction of total network, 0 < q < 1)
            gamma: Connectivity parameter (0 <= gamma <= 1)
                  Fraction of honest miners who mine on attacker's fork in case of tie
        """
        self.q = q
        self.gamma = gamma

    def theoretical_revenue(self) -> float:
        """
        Calculate theoretical expected revenue using the closed-form formula
        from Eyal & Sirer (2014), "Majority is not Enough."

        The revenue fraction for the selfish mining pool is:

            R = [q(1-q)^2(4q + gamma(1-2q)) - q^3]
                / [1 - q(1 + (2-q)*q)]

        where q = attacker hash power, gamma = connectivity parameter.

        Returns:
            Expected relative revenue (fraction of all blocks)
        """
        q = self.q
        gamma = self.gamma

        numerator = q * (1 - q)**2 * (4 * q + gamma * (1 - 2 * q)) - q**3
        denominator = 1 - q * (1 + (2 - q) * q)

        if denominator == 0:
            return q  # Edge case: fall back to honest revenue

        revenue = numerator / denominator

        # Revenue can't be negative (miner would just mine honestly)
        return max(revenue, 0.0)

    def simulate(self, num_blocks: int = 10000, seed: int = None) -> MiningResult:
        """
        Simulate selfish mining strategy using state machine from Eyal & Sirer (2014).

        State represents lead: difference between private chain and public chain length.

        Strategy:
        - State 0, attacker mines: go to state 1 (build private fork)
        - State 0, honest mines: stay at state 0 (adopt their block)
        - State 1, attacker mines: go to state 2
        - State 1, honest mines: publish private block (race), back to state 0
        - State 2+, attacker mines: increase lead
        - State 2, honest mines: publish all private blocks (override), back to state 0
        - State 3+, honest mines: publish one block, decrease lead by 1

        Args:
            num_blocks: Total number of blocks in the main chain
            seed: Random seed for reproducibility

        Returns:
            MiningResult object with simulation statistics
        """
        if seed is not None:
            np.random.seed(seed)

        q = self.q
        gamma = self.gamma

        # State: lead (private chain length - public chain length)
        state = 0

        # Statistics
        attacker_blocks_mined = 0
        attacker_blocks_in_main_chain = 0
        blocks_in_main_chain = 0

        # Simulation loop - simulate until main chain has num_blocks blocks
        while blocks_in_main_chain < num_blocks:
            # Who finds the next block?
            attacker_mines = np.random.random() < q

            if attacker_mines:
                # Attacker finds a block
                attacker_blocks_mined += 1
                state += 1  # Increase lead

            else:
                # Honest miners find a block
                if state == 0:
                    # We're tied - adopt their block
                    blocks_in_main_chain += 1

                elif state == 1:
                    # We're 1 ahead - publish our private block (race condition)
                    # This creates state 0' from Eyal & Sirer (2014):
                    # Both branches have 1 block from fork point.
                    # The next block resolves the race:
                    #   - Pool mines (prob q): publishes 2 blocks, honest orphaned
                    #   - gamma-honest extend pool's chain (prob gamma*(1-q)): pool gets 1
                    #   - (1-gamma)-honest extend honest chain: pool gets 0
                    # In all cases, 2 blocks enter main chain from the fork point.
                    r = np.random.random()
                    if r < q:
                        # Pool mines next block - wins with 2 blocks
                        attacker_blocks_in_main_chain += 2
                        blocks_in_main_chain += 2
                    elif r < q + gamma * (1 - q):
                        # Gamma-honest extend pool's chain - pool gets 1
                        attacker_blocks_in_main_chain += 1
                        blocks_in_main_chain += 2
                    else:
                        # Honest chain wins - pool gets 0
                        blocks_in_main_chain += 2
                    state = 0

                elif state == 2:
                    # We're 2 ahead - publish both blocks to override their fork
                    attacker_blocks_in_main_chain += 2
                    blocks_in_main_chain += 2
                    state = 0

                else:  # state > 2
                    # We're 3+ ahead - publish just enough to stay ahead
                    attacker_blocks_in_main_chain += 1
                    blocks_in_main_chain += 1
                    state -= 1

        relative_revenue = attacker_blocks_in_main_chain / blocks_in_main_chain

        return MiningResult(
            revenue=attacker_blocks_in_main_chain,
            blocks_mined=attacker_blocks_mined,
            blocks_accepted=attacker_blocks_in_main_chain,
            total_blocks=blocks_in_main_chain,
            relative_revenue=relative_revenue
        )

    def is_profitable(self) -> bool:
        """
        Check if selfish mining is more profitable than honest mining.

        Returns:
            True if selfish mining revenue > honest mining revenue
        """
        return self.theoretical_revenue() > self.q


class OptimalSelfishMining:
    """
    Implementation of optimal selfish mining strategy.

    The miner uses dynamic programming to determine the optimal action
    for each state (a, h), where:
    - a: number of blocks in attacker's private fork
    - h: number of blocks in honest miners' public fork
    """

    def __init__(self, q: float, gamma: float):
        """
        Initialize optimal selfish miner.

        Args:
            q: Hashing power (fraction of total network, 0 < q < 1)
            gamma: Connectivity parameter (0 <= gamma <= 1)
        """
        self.q = q
        self.gamma = gamma

    def get_optimal_action(self, a: int, h: int) -> str:
        """
        Determine optimal action for state (a, h).

        Actions:
        - "adopt": Abandon private fork, adopt public chain
        - "override": Publish entire private fork to override public chain
        - "match": Publish blocks to match public chain (create tie)
        - "wait": Continue mining on private fork without publishing

        Args:
            a: Number of blocks in attacker's private fork
            h: Number of blocks in honest miners' public fork

        Returns:
            Optimal action as string
        """
        # If we're behind, we must adopt
        if a < h:
            return "adopt"

        # If we're equal
        if a == h:
            if a == 0:
                # Starting state - wait
                return "wait"
            else:
                # Race condition - publish to match
                return "match"

        # If we're ahead by 1 (a = h + 1)
        if a == h + 1:
            # Strategy: wait for next block
            # If we mine it, we'll be 2 ahead (can override)
            # If they mine it, we match (race condition)
            return "wait"

        # If we're ahead by 2 or more (a >= h + 2)
        if a >= h + 2:
            # We can safely override and still maintain lead
            # But optimal strategy is often to wait unless h is catching up
            if h == 0:
                # They haven't mined anything yet, keep waiting
                return "wait"
            else:
                # They're catching up, override to secure blocks
                return "override"

        return "wait"

    def compute_state_values(self, max_blocks: int = 20) -> Dict[Tuple[int, int], float]:
        """
        Compute expected value for each state (a, h) using dynamic programming.

        Args:
            max_blocks: Maximum number of blocks to consider in state space

        Returns:
            Dictionary mapping (a, h) to expected value
        """
        q = self.q
        gamma = self.gamma

        # Initialize value function
        V = {}

        # Base cases
        for a in range(max_blocks + 1):
            for h in range(max_blocks + 1):
                if a < h:
                    # Behind - must adopt, value is 0
                    V[(a, h)] = 0.0
                elif a >= h + 2:
                    # Far ahead - can override
                    V[(a, h)] = float(a)
                else:
                    V[(a, h)] = 0.0

        # Iterative value computation (simplified)
        for _ in range(100):  # Iterate until convergence
            V_new = V.copy()

            for a in range(max_blocks + 1):
                for h in range(max_blocks + 1):
                    if a < h:
                        continue

                    action = self.get_optimal_action(a, h)

                    if action == "adopt":
                        V_new[(a, h)] = 0
                    elif action == "override":
                        V_new[(a, h)] = a
                    elif action == "match":
                        # Race condition
                        V_new[(a, h)] = gamma * a + (1 - gamma) * 0
                    elif action == "wait":
                        # Expected value from waiting
                        if a < max_blocks and h < max_blocks:
                            val = q * V.get((a + 1, h), 0) + (1 - q) * V.get((a, h + 1), 0)
                            V_new[(a, h)] = val

            V = V_new

        return V

    def get_strategy_matrix(self, max_a: int = 10, max_h: int = 10) -> np.ndarray:
        """
        Generate matrix of optimal actions for visualization.

        Args:
            max_a: Maximum value for attacker blocks
            max_h: Maximum value for honest blocks

        Returns:
            Matrix where each entry is encoded action:
            0: adopt, 1: wait, 2: match, 3: override
        """
        action_encoding = {
            "adopt": 0,
            "wait": 1,
            "match": 2,
            "override": 3
        }

        matrix = np.zeros((max_a + 1, max_h + 1))

        for a in range(max_a + 1):
            for h in range(max_h + 1):
                action = self.get_optimal_action(a, h)
                matrix[a, h] = action_encoding[action]

        return matrix


def compare_strategies(q: float, gamma: float, num_blocks: int = 10000,
                       num_simulations: int = 100, seed: int = None) -> Dict:
    """
    Compare honest mining vs selfish mining strategies.

    Args:
        q: Hashing power
        gamma: Connectivity parameter
        num_blocks: Blocks per simulation
        num_simulations: Number of simulation runs
        seed: Random seed

    Returns:
        Dictionary with comparison results
    """
    honest = HonestMining(q)
    selfish = SelfishMining(q, gamma)

    # Theoretical values
    honest_theory = honest.theoretical_revenue()
    selfish_theory = selfish.theoretical_revenue()

    # Simulations
    honest_revenues = []
    selfish_revenues = []

    for i in range(num_simulations):
        sim_seed = None if seed is None else seed + i

        honest_result = honest.simulate(num_blocks, sim_seed)
        selfish_result = selfish.simulate(num_blocks, sim_seed)

        honest_revenues.append(honest_result.relative_revenue)
        selfish_revenues.append(selfish_result.relative_revenue)

    return {
        'q': q,
        'gamma': gamma,
        'honest_theory': honest_theory,
        'selfish_theory': selfish_theory,
        'honest_simulation_mean': np.mean(honest_revenues),
        'honest_simulation_std': np.std(honest_revenues),
        'selfish_simulation_mean': np.mean(selfish_revenues),
        'selfish_simulation_std': np.std(selfish_revenues),
        'selfish_profitable': selfish_theory > honest_theory
    }


def find_profitability_boundary(gamma_values: np.ndarray) -> np.ndarray:
    """
    Find the minimum q where selfish mining becomes profitable for each gamma.

    Args:
        gamma_values: Array of gamma values to test

    Returns:
        Array of minimum q values for profitability
    """
    q_threshold = []

    for gamma in gamma_values:
        # Binary search for threshold q
        q_min, q_max = 0.0, 0.5

        while q_max - q_min > 0.001:
            q_mid = (q_min + q_max) / 2
            selfish = SelfishMining(q_mid, gamma)

            if selfish.is_profitable():
                q_max = q_mid
            else:
                q_min = q_mid

        q_threshold.append((q_min + q_max) / 2)

    return np.array(q_threshold)
