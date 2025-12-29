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
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
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


def main():
    """
    Main entry point:
      - Run PoW experiments
      - Analyze exponentiality of solution times
      - Print a textual summary
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


if __name__ == "__main__":
    main()


