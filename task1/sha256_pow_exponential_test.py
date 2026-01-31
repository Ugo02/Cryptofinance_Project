#!/usr/bin/env python3
"""
Proof-of-work experiment with SHA256.

Goal:
  - Choose a hash function (SHA256)
  - Create proof-of-work (PoW) problems with a fixed difficulty
  - Measure the time required to find a valid solution for each problem
  - Check whether the distribution of solution times follows an exponential law
"""

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def sha256_hash(data: bytes) -> str:
    """
    Computes SHA256 hash of the given data and returns it as hex string.
    """
    return hashlib.sha256(data).hexdigest()


def meets_difficulty(hash_hex: str, difficulty_bits: int) -> bool:
    """
    Checks if the given hash meets the difficulty target.

    Difficulty is expressed in number of leading zero bits.
    """
    # Convert hex hash to integer
    hash_int = int(hash_hex, 16)
    # Number of bits in SHA256
    total_bits = 256
    # Condition: the top difficulty_bits must be zero
    return hash_int >> (total_bits - difficulty_bits) == 0


@dataclass
class PowResult:
    nonce: int
    hash_hex: str
    attempts: int
    duration_seconds: float


def solve_proof_of_work(
    base_data: str,
    difficulty_bits: int,
    max_nonce: int = 2**32 - 1,
) -> PowResult:
    """
    Solves a proof-of-work problem:

    Find a nonce such that SHA256(base_data || nonce) has at least
    `difficulty_bits` leading zero bits.

    Returns the found nonce, hash, number of attempts, and duration.
    """
    # Randomize starting nonce to avoid always starting from zero
    start_nonce = random.randint(0, max_nonce)

    attempts = 0
    nonce = start_nonce

    start_time = time.perf_counter()

    while True:
        attempts += 1
        message = f"{base_data}|{nonce}".encode("utf-8")
        hash_hex = sha256_hash(message)

        if meets_difficulty(hash_hex, difficulty_bits):
            duration = time.perf_counter() - start_time
            return PowResult(
                nonce=nonce,
                hash_hex=hash_hex,
                attempts=attempts,
                duration_seconds=duration,
            )

        nonce = (nonce + 1) & max_nonce


def run_pow_experiments(
    n_problems: int = 100,
    difficulty_bits: int = 18,
) -> Tuple[List[PowResult], np.ndarray]:
    """
    Runs multiple PoW problems and records their solution times.

    Args:
        n_problems: Number of independent PoW problems to solve.
        difficulty_bits: PoW difficulty in leading zero bits.
                         Higher difficulty => longer expected times.

    Returns:
        - List of PowResult objects for each problem
        - Numpy array of durations (in seconds)
    """
    results: List[PowResult] = []
    durations: List[float] = []

    print("=" * 70)
    print("Proof-of-Work experiment with SHA256")
    print("=" * 70)
    print(f"Number of problems : {n_problems}")
    print(f"Difficulty         : {difficulty_bits} leading zero bits")
    print("-" * 70)

    for i in range(n_problems):
        base_data = f"block_{i}_random_{random.getrandbits(64)}"
        result = solve_proof_of_work(base_data, difficulty_bits)
        results.append(result)
        durations.append(result.duration_seconds)

        print(
            f"Problem {i+1:3d}/{n_problems}: "
            f"time = {result.duration_seconds:.4f} s, "
            f"attempts = {result.attempts}"
        )

    durations_array = np.array(durations, dtype=float)
    return results, durations_array


def analyze_exponential_distribution(durations: np.ndarray) -> dict:
    """
    Analyzes whether the given durations follow an exponential distribution.

    For a Poisson process with constant success probability per trial,
    the waiting time until success is theoretically exponential.

    Steps:
        - Estimate lambda via MLE: lambda_hat = 1 / mean(durations)
        - Use Kolmogorov-Smirnov test against Exp(lambda_hat)
    """
    n = durations.size
    mean_duration = durations.mean()
    lambda_hat = 1.0 / mean_duration

    # KS test against exponential with scale = mean_duration
    # stats.expon has CDF F(x) = 1 - exp(-x / scale) for x >= 0
    ks_stat, p_value = stats.kstest(durations, "expon", args=(0, mean_duration))

    is_exponential = p_value > 0.05

    return {
        "n_samples": n,
        "mean_duration": mean_duration,
        "lambda_hat": lambda_hat,
        "ks_statistic": ks_stat,
        "p_value": p_value,
        "is_exponential": is_exponential,
        "interpretation": (
            "Durations are compatible with an exponential distribution"
            if is_exponential
            else "Durations deviate from an exponential distribution"
        ),
    }


def summarize_attempts(results: List[PowResult]) -> dict:
    """
    Computes basic statistics on the number of attempts per PoW problem.
    """
    attempts = np.array([r.attempts for r in results], dtype=float)
    return {
        "min_attempts": int(attempts.min()),
        "max_attempts": int(attempts.max()),
        "mean_attempts": float(attempts.mean()),
        "std_attempts": float(attempts.std(ddof=1)) if attempts.size > 1 else 0.0,
    }


def plot_exponential_visualizations(durations: np.ndarray, output_dir='results'):
    """
    Generate visualizations for the exponential distribution analysis.

    Produces:
    - Histogram of durations with fitted exponential PDF overlay
    - QQ-plot against exponential distribution
    - Empirical vs theoretical CDF

    Args:
        durations: Array of PoW solution times
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    mean_dur = durations.mean()
    lambda_hat = 1.0 / mean_dur

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Histogram + fitted exponential PDF ---
    ax1 = axes[0]
    ax1.hist(durations, bins=30, density=True, color='steelblue', alpha=0.7,
             edgecolor='black', linewidth=0.3, label='Observed durations')

    x = np.linspace(0, durations.max() * 1.1, 200)
    pdf = lambda_hat * np.exp(-lambda_hat * x)
    ax1.plot(x, pdf, 'r-', linewidth=2,
             label=f'Fitted Exp(lambda={lambda_hat:.2f})')
    ax1.set_xlabel('Duration (seconds)', fontsize=11)
    ax1.set_ylabel('Probability density', fontsize=11)
    ax1.set_title('PoW Duration Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    # --- Plot 2: QQ-plot ---
    ax2 = axes[1]
    sorted_durations = np.sort(durations)
    n = len(sorted_durations)
    theoretical_quantiles = stats.expon.ppf(
        (np.arange(1, n + 1) - 0.5) / n, scale=mean_dur
    )

    ax2.scatter(theoretical_quantiles, sorted_durations, s=15, alpha=0.6,
                color='steelblue', edgecolor='none')
    max_val = max(theoretical_quantiles.max(), sorted_durations.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', linewidth=2,
             label='Perfect exponential')
    ax2.set_xlabel('Theoretical quantiles (Exponential)', fontsize=11)
    ax2.set_ylabel('Observed quantiles', fontsize=11)
    ax2.set_title('QQ-Plot: Durations vs Exponential', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')

    # --- Plot 3: Empirical CDF vs Theoretical CDF ---
    ax3 = axes[2]
    ecdf_y = np.arange(1, n + 1) / n
    ax3.step(sorted_durations, ecdf_y, where='post', color='steelblue',
             linewidth=1.5, label='Empirical CDF')

    x_cdf = np.linspace(0, sorted_durations.max() * 1.1, 200)
    theoretical_cdf = 1 - np.exp(-lambda_hat * x_cdf)
    ax3.plot(x_cdf, theoretical_cdf, 'r--', linewidth=2,
             label=f'Theoretical CDF Exp(lambda={lambda_hat:.2f})')
    ax3.set_xlabel('Duration (seconds)', fontsize=11)
    ax3.set_ylabel('Cumulative probability', fontsize=11)
    ax3.set_title('CDF Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'pow_exponential_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    """
    Main entry point:
      - Run PoW experiments
      - Analyze exponentiality of solution times
      - Print a textual summary
      - Generate visualizations
    """

    n_problems = 100
    difficulty_bits = 18

    results, durations = run_pow_experiments(
        n_problems=n_problems,
        difficulty_bits=difficulty_bits,
    )

    print("\n" + "-" * 70)
    print("Durations statistics (seconds)")
    print("-" * 70)
    print(f"Min      : {durations.min():.6f}")
    print(f"Max      : {durations.max():.6f}")
    print(f"Mean     : {durations.mean():.6f}")
    print(f"Std dev  : {durations.std(ddof=1):.6f}")

    attempt_stats = summarize_attempts(results)
    print("\n" + "-" * 70)
    print("Attempts statistics (number of hashes per solution)")
    print("-" * 70)
    print(f"Min attempts  : {attempt_stats['min_attempts']}")
    print(f"Max attempts  : {attempt_stats['max_attempts']}")
    print(f"Mean attempts : {attempt_stats['mean_attempts']:.2f}")
    print(f"Std attempts  : {attempt_stats['std_attempts']:.2f}")

    print("\n" + "-" * 70)
    print("Kolmogorov-Smirnov test for exponential distribution")
    print("-" * 70)
    exp_analysis = analyze_exponential_distribution(durations)
    print(f"Number of samples      : {exp_analysis['n_samples']}")
    print(f"Mean duration (seconds): {exp_analysis['mean_duration']:.6f}")
    print(f"Lambda_hat (1/mean)    : {exp_analysis['lambda_hat']:.6f}")
    print(f"KS statistic           : {exp_analysis['ks_statistic']:.6f}")
    print(f"P-value                : {exp_analysis['p_value']:.6f}")
    print(f"Result                 : {exp_analysis['interpretation']}")

    print("\n" + "=" * 70)
    if exp_analysis["is_exponential"]:
        print(
            "CONCLUSION: Solution times are CONSISTENT with an exponential "
            "distribution (Poisson process hypothesis not rejected)."
        )
    else:
        print(
            "CONCLUSION: Solution times DO NOT appear exponential "
            "(Poisson process hypothesis rejected)."
        )
    print("=" * 70)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_exponential_visualizations(durations)


if __name__ == "__main__":
    main()


