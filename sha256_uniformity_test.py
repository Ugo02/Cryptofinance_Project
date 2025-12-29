#!/usr/bin/env python3
"""
Statistical test of SHA256 distribution uniformity
Generates a list of hashes and checks if the distribution is uniform
"""

import hashlib
import random
import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt


def generate_sha256_hash(data):
    """
    Generates a SHA256 hash from input data
    
    Args:
        data: Data to hash (bytes or string)
    
    Returns:
        str: SHA256 hexadecimal hash
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def generate_hash_list(n_samples=10000):
    """
    Generates a list of SHA256 hashes from random inputs.
    
    Args:
        n_samples: Number of hashes to generate.
    
    Returns:
        list: List of hexadecimal hashes.
    """
    hashes = []
    # Generate random inputs
    for _ in range(n_samples):
        random_data = str(random.randint(0, 2**64)).encode('utf-8')
        hashes.append(generate_sha256_hash(random_data))
    return hashes


def analyze_bit_distribution(hashes):
    """
    Analyzes the bit distribution in the hashes
    
    Args:
        hashes: List of hexadecimal hashes
    
    Returns:
        dict: Statistics on bit distribution
    """
    # Convert each hash to binary
    all_bits = []
    for hash_hex in hashes:
        # Convert hex to int then to binary
        hash_int = int(hash_hex, 16)
        bits = bin(hash_int)[2:].zfill(256)  # SHA256 produces 256 bits
        all_bits.extend([int(bit) for bit in bits])
    
    # Count 0s and 1s
    bit_counts = Counter(all_bits)
    total_bits = len(all_bits)
    
    return {
        'zeros': bit_counts[0],
        'ones': bit_counts[1],
        'total': total_bits,
        'zero_ratio': bit_counts[0] / total_bits,
        'one_ratio': bit_counts[1] / total_bits
    }


def analyze_byte_distribution(hashes):
    """
    Analyzes the byte distribution in the hashes
    
    Args:
        hashes: List of hexadecimal hashes
    
    Returns:
        dict: Statistics on byte distribution
    """
    # Convert each hash to bytes
    all_bytes = []
    for hash_hex in hashes:
        hash_bytes = bytes.fromhex(hash_hex)
        all_bytes.extend(hash_bytes)
    
    # Count frequency of each byte value (0-255)
    byte_counts = Counter(all_bytes)
    
    # Calculate statistics
    expected_frequency = len(all_bytes) / 256  # Expected uniform distribution
    
    return {
        'byte_counts': byte_counts,
        'total_bytes': len(all_bytes),
        'expected_frequency': expected_frequency,
        'unique_bytes': len(byte_counts)
    }


def chi_square_test_uniformity(hashes, test_type='byte'):
    """
    Chi-square test to verify distribution uniformity
    
    Args:
        hashes: List of hexadecimal hashes
        test_type: Type of test ('byte' or 'bit')
    
    Returns:
        dict: Chi-square test results
    """
    if test_type == 'byte':
        # Analyze byte distribution
        all_bytes = []
        for hash_hex in hashes:
            hash_bytes = bytes.fromhex(hash_hex)
            all_bytes.extend(hash_bytes)
        
        # Count frequency of each byte value (0-255)
        observed_freq = [0] * 256
        for byte_val in all_bytes:
            observed_freq[byte_val] += 1
        
        n = len(all_bytes)
        expected_freq = n / 256  # Expected uniform distribution
        
        # Calculate chi-square
        chi_square = sum((observed_freq[i] - expected_freq)**2 / expected_freq 
                        for i in range(256))
        
        # Degrees of freedom: 256 - 1 = 255
        degrees_of_freedom = 255
    
    elif test_type == 'bit':
        # Analyze bit distribution
        all_bits = []
        for hash_hex in hashes:
            hash_int = int(hash_hex, 16)
            bits = bin(hash_int)[2:].zfill(256)
            all_bits.extend([int(bit) for bit in bits])
        
        # Count 0s and 1s
        bit_counts = Counter(all_bits)
        observed_freq = [bit_counts[0], bit_counts[1]]
        
        n = len(all_bits)
        expected_freq = n / 2  # Expected uniform distribution
        
        # Calculate chi-square
        chi_square = sum((observed_freq[i] - expected_freq)**2 / expected_freq 
                        for i in range(2))
        
        # Degrees of freedom: 2 - 1 = 1
        degrees_of_freedom = 1
    
    # Calculate p-value
    p_value = 1 - stats.chi2.cdf(chi_square, degrees_of_freedom)
    
    # Interpretation (significance threshold at 0.05)
    is_uniform = p_value > 0.05
    
    return {
        'chi_square': chi_square,
        'degrees_of_freedom': degrees_of_freedom,
        'p_value': p_value,
        'is_uniform': is_uniform,
        'interpretation': 'Uniform distribution' if is_uniform else 'Non-uniform distribution'
    }


def kolmogorov_smirnov_test(hashes):
    """
    Kolmogorov-Smirnov test to verify uniformity
    
    Args:
        hashes: List of hexadecimal hashes
    
    Returns:
        dict: KS test results
    """
    # Convert hashes to normalized numerical values (0-1)
    normalized_values = []
    for hash_hex in hashes:
        # Take first 8 hex characters to get a 32-bit value
        hash_int = int(hash_hex[:8], 16)
        normalized = hash_int / (2**32 - 1)  # Normalize between 0 and 1
        normalized_values.append(normalized)
    
    # KS test against uniform distribution
    ks_statistic, p_value = stats.kstest(normalized_values, 'uniform')
    
    is_uniform = p_value > 0.05
    
    return {
        'ks_statistic': ks_statistic,
        'p_value': p_value,
        'is_uniform': is_uniform,
        'interpretation': 'Uniform distribution' if is_uniform else 'Non-uniform distribution'
    }


def main():
    """
    Main function to execute statistical tests
    """
    print("=" * 70)
    print("Statistical test of SHA256 distribution uniformity")
    print("=" * 70)
    
    # Parameters
    n_samples = 10000
    
    print(f"\nGenerating {n_samples} random SHA256 hashes...")
    hashes = generate_hash_list(n_samples)
    print(f"✓ {len(hashes)} hashes generated")
    
    # Analyze bit distribution
    print("\n" + "-" * 70)
    print("1. Bit distribution analysis")
    print("-" * 70)
    bit_stats = analyze_bit_distribution(hashes)
    print(f"Total number of bits: {bit_stats['total']}")
    print(f"Number of 0s: {bit_stats['zeros']} ({bit_stats['zero_ratio']*100:.2f}%)")
    print(f"Number of 1s: {bit_stats['ones']} ({bit_stats['one_ratio']*100:.2f}%)")
    print(f"Expected ratio for uniform distribution: 50% / 50%")
    
    # Analyze byte distribution
    print("\n" + "-" * 70)
    print("2. Byte distribution analysis")
    print("-" * 70)
    byte_stats = analyze_byte_distribution(hashes)
    print(f"Total number of bytes: {byte_stats['total_bytes']}")
    print(f"Number of unique byte values: {byte_stats['unique_bytes']}/256")
    print(f"Expected frequency per byte (uniform): {byte_stats['expected_frequency']:.2f}")
    
    # Chi-square test on bits
    print("\n" + "-" * 70)
    print("3. Chi-square test - Bit distribution")
    print("-" * 70)
    chi_bit = chi_square_test_uniformity(hashes, test_type='bit')
    print(f"Chi-square statistic: {chi_bit['chi_square']:.4f}")
    print(f"Degrees of freedom: {chi_bit['degrees_of_freedom']}")
    print(f"P-value: {chi_bit['p_value']:.6f}")
    print(f"Result: {chi_bit['interpretation']}")
    
    # Chi-square test on bytes
    print("\n" + "-" * 70)
    print("4. Chi-square test - Byte distribution")
    print("-" * 70)
    chi_byte = chi_square_test_uniformity(hashes, test_type='byte')
    print(f"Chi-square statistic: {chi_byte['chi_square']:.4f}")
    print(f"Degrees of freedom: {chi_byte['degrees_of_freedom']}")
    print(f"P-value: {chi_byte['p_value']:.6f}")
    print(f"Result: {chi_byte['interpretation']}")
    
    # Kolmogorov-Smirnov test
    print("\n" + "-" * 70)
    print("5. Kolmogorov-Smirnov test")
    print("-" * 70)
    ks_result = kolmogorov_smirnov_test(hashes)
    print(f"KS statistic: {ks_result['ks_statistic']:.6f}")
    print(f"P-value: {ks_result['p_value']:.6f}")
    print(f"Result: {ks_result['interpretation']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Bit distribution: {chi_bit['interpretation']}")
    print(f"✓ Byte distribution: {chi_byte['interpretation']}")
    print(f"✓ KS test: {ks_result['interpretation']}")
    
    # General conclusion
    all_uniform = (chi_bit['is_uniform'] and chi_byte['is_uniform'] and 
                   ks_result['is_uniform'])
    
    print("\n" + "=" * 70)
    if all_uniform:
        print("CONCLUSION: SHA256 distribution appears UNIFORM according to statistical tests")
    else:
        print("CONCLUSION: Some tests suggest a NON-UNIFORM distribution")
    print("=" * 70)
    
    return hashes, bit_stats, byte_stats, chi_bit, chi_byte, ks_result


if __name__ == "__main__":
    main()


