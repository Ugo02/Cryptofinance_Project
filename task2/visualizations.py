"""
Visualization Module for Mining Strategies
==========================================

This module provides comprehensive visualization functions for:
- Strategy comparison (simulation vs theory)
- Selfish mining profitability regions
- Optimal selfish mining decision matrices

Author: PhD Student
Course: CryptoFinance
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from mining_strategies import (
    HonestMining, SelfishMining, OptimalSelfishMining,
    compare_strategies, find_profitability_boundary
)


# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def plot_strategy_comparison(q_values: List[float], num_blocks: int = 10000,
                             num_simulations: int = 100, gamma: float = 0.5,
                             save_path: str = None):
    """
    Plot comparison between honest and selfish mining.

    Compares theoretical predictions with simulation results across
    different hashing power values.

    Args:
        q_values: List of hashing power values to test
        num_blocks: Blocks per simulation
        num_simulations: Number of simulation runs
        gamma: Connectivity parameter for selfish mining
        save_path: Path to save figure (if None, display only)
    """
    honest_theory = []
    selfish_theory = []
    honest_sim_mean = []
    honest_sim_std = []
    selfish_sim_mean = []
    selfish_sim_std = []

    for q in q_values:
        result = compare_strategies(q, gamma, num_blocks, num_simulations, seed=42)

        honest_theory.append(result['honest_theory'])
        selfish_theory.append(result['selfish_theory'])
        honest_sim_mean.append(result['honest_simulation_mean'])
        honest_sim_std.append(result['honest_simulation_std'])
        selfish_sim_mean.append(result['selfish_simulation_mean'])
        selfish_sim_std.append(result['selfish_simulation_std'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Theory vs Simulation
    ax1.plot(q_values, honest_theory, 'b-', linewidth=2, label='Honest (Theory)')
    ax1.plot(q_values, selfish_theory, 'r-', linewidth=2, label=f'Selfish (Theory, γ={gamma})')

    ax1.errorbar(q_values, honest_sim_mean, yerr=honest_sim_std,
                 fmt='bo', alpha=0.6, markersize=6, capsize=4,
                 label='Honest (Simulation)')
    ax1.errorbar(q_values, selfish_sim_mean, yerr=selfish_sim_std,
                 fmt='ro', alpha=0.6, markersize=6, capsize=4,
                 label='Selfish (Simulation)')

    ax1.set_xlabel('Hashing Power (q)', fontsize=12)
    ax1.set_ylabel('Relative Revenue', fontsize=12)
    ax1.set_title('Mining Strategy Performance: Theory vs Simulation', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(q_values), max(q_values)])

    # Plot 2: Relative Advantage of Selfish Mining
    advantage_theory = np.array(selfish_theory) - np.array(honest_theory)
    advantage_sim = np.array(selfish_sim_mean) - np.array(honest_sim_mean)

    ax2.plot(q_values, advantage_theory, 'g-', linewidth=2, label='Theory')
    ax2.plot(q_values, advantage_sim, 'go', alpha=0.6, markersize=6, label='Simulation')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)

    ax2.set_xlabel('Hashing Power (q)', fontsize=12)
    ax2.set_ylabel('Revenue Advantage (Selfish - Honest)', fontsize=12)
    ax2.set_title(f'Selfish Mining Advantage (γ={gamma})', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(q_values), max(q_values)])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_profitability_heatmap(q_range: Tuple[float, float] = (0.0, 0.5),
                               gamma_range: Tuple[float, float] = (0.0, 1.0),
                               resolution: int = 50,
                               save_path: str = None):
    """
    Create heatmap showing where selfish mining is profitable.

    Args:
        q_range: Tuple of (min_q, max_q) for hashing power
        gamma_range: Tuple of (min_gamma, max_gamma) for connectivity
        resolution: Number of points in each dimension
        save_path: Path to save figure
    """
    q_values = np.linspace(q_range[0], q_range[1], resolution)
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], resolution)

    # Compute profitability matrix
    profitability = np.zeros((len(gamma_values), len(q_values)))
    revenue_advantage = np.zeros((len(gamma_values), len(q_values)))

    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            if q == 0 or q >= 1:
                continue

            selfish = SelfishMining(q, gamma)
            selfish_rev = selfish.theoretical_revenue()
            honest_rev = q

            revenue_advantage[i, j] = selfish_rev - honest_rev
            profitability[i, j] = 1 if selfish_rev > honest_rev else 0

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Profitability regions
    im1 = ax1.imshow(profitability, extent=[q_range[0], q_range[1],
                                            gamma_range[0], gamma_range[1]],
                     origin='lower', aspect='auto', cmap='RdYlGn', alpha=0.8)

    # Add profitability boundary
    boundary_q = []
    boundary_gamma = []
    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            if j > 0 and profitability[i, j] != profitability[i, j-1]:
                boundary_q.append(q)
                boundary_gamma.append(gamma)

    if boundary_q:
        ax1.scatter(boundary_q, boundary_gamma, c='blue', s=2, alpha=0.6,
                   label='Profitability Boundary')

    ax1.set_xlabel('Hashing Power (q)', fontsize=12)
    ax1.set_ylabel('Connectivity (γ)', fontsize=12)
    ax1.set_title('Selfish Mining Profitability Regions', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Profitable (1) vs Not Profitable (0)', fontsize=10)

    # Add text annotations
    ax1.text(0.25, 0.95, 'SELFISH MINING\nPROFITABLE',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(0.25, 0.05, 'HONEST MINING\nOPTIMAL',
             transform=ax1.transAxes, fontsize=11, fontweight='bold',
             ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Plot 2: Revenue advantage heatmap
    im2 = ax2.imshow(revenue_advantage, extent=[q_range[0], q_range[1],
                                                 gamma_range[0], gamma_range[1]],
                     origin='lower', aspect='auto', cmap='RdBu_r', vmin=-0.05, vmax=0.05)

    ax2.set_xlabel('Hashing Power (q)', fontsize=12)
    ax2.set_ylabel('Connectivity (γ)', fontsize=12)
    ax2.set_title('Revenue Advantage: Selfish - Honest', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Revenue Advantage', fontsize=10)

    # Add zero contour line
    contour = ax2.contour(q_values, gamma_values, revenue_advantage,
                          levels=[0], colors='black', linewidths=2)
    ax2.clabel(contour, inline=True, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_profitability_boundary_curves(save_path: str = None):
    """
    Plot the profitability boundary as curves for different gamma values.

    Args:
        save_path: Path to save figure
    """
    gamma_values = np.linspace(0, 1, 100)
    q_threshold = find_profitability_boundary(gamma_values)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the boundary
    ax.plot(gamma_values, q_threshold, 'b-', linewidth=3, label='Profitability Threshold')

    # Fill regions
    ax.fill_between(gamma_values, q_threshold, 0.5,
                     alpha=0.3, color='green', label='Selfish Mining Profitable')
    ax.fill_between(gamma_values, 0, q_threshold,
                     alpha=0.3, color='red', label='Honest Mining Better')

    # Add important points
    # Theoretical threshold at gamma=0
    idx_gamma_0 = np.argmin(np.abs(gamma_values - 0))
    ax.plot(0, q_threshold[idx_gamma_0], 'ro', markersize=10,
            label=f'γ=0: q > {q_threshold[idx_gamma_0]:.3f}')

    # Theoretical threshold at gamma=0.5
    idx_gamma_05 = np.argmin(np.abs(gamma_values - 0.5))
    ax.plot(0.5, q_threshold[idx_gamma_05], 'go', markersize=10,
            label=f'γ=0.5: q > {q_threshold[idx_gamma_05]:.3f}')

    # Theoretical threshold at gamma=1
    idx_gamma_1 = np.argmin(np.abs(gamma_values - 1.0))
    ax.plot(1.0, q_threshold[idx_gamma_1], 'mo', markersize=10,
            label=f'γ=1: q > {q_threshold[idx_gamma_1]:.3f}')

    ax.set_xlabel('Connectivity Parameter (γ)', fontsize=13)
    ax.set_ylabel('Minimum Hashing Power (q) for Profitability', fontsize=13)
    ax.set_title('Selfish Mining Profitability Boundary', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 0.5])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_optimal_strategy_matrix(q: float, gamma: float,
                                 max_a: int = 15, max_h: int = 15,
                                 save_path: str = None):
    """
    Visualize optimal selfish mining decisions for different states (a, h).

    Args:
        q: Hashing power
        gamma: Connectivity parameter
        max_a: Maximum attacker blocks to display
        max_h: Maximum honest blocks to display
        save_path: Path to save figure
    """
    optimal = OptimalSelfishMining(q, gamma)
    strategy_matrix = optimal.get_strategy_matrix(max_a, max_h)

    # Create custom colormap
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']  # adopt, wait, match, override
    n_bins = 4
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(strategy_matrix, cmap=cmap, aspect='auto',
                   origin='lower', vmin=0, vmax=3, interpolation='nearest')

    # Add text annotations
    for a in range(max_a + 1):
        for h in range(max_h + 1):
            action = optimal.get_optimal_action(a, h)
            action_short = action[0].upper()  # First letter
            color = 'white' if strategy_matrix[a, h] in [0, 3] else 'black'
            ax.text(h, a, action_short, ha='center', va='center',
                   color=color, fontsize=9, fontweight='bold')

    # Customize axes
    ax.set_xlabel('Honest Chain Length (h)', fontsize=13)
    ax.set_ylabel('Attacker Chain Length (a)', fontsize=13)
    ax.set_title(f'Optimal Selfish Mining Strategy Matrix\n(q={q:.2f}, γ={gamma:.2f})',
                fontsize=14, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(max_h + 1))
    ax.set_yticks(range(max_a + 1))

    # Add gridlines
    ax.set_xticks(np.arange(-0.5, max_h + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, max_a + 1, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Adopt (A): Abandon private fork'),
        Patch(facecolor=colors[1], label='Wait (W): Continue mining privately'),
        Patch(facecolor=colors[2], label='Match (M): Publish to create tie'),
        Patch(facecolor=colors[3], label='Override (O): Publish entire fork')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             bbox_to_anchor=(1.02, 1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def plot_multiple_optimal_strategies(q_values: List[float], gamma: float = 0.5,
                                     max_a: int = 10, max_h: int = 10,
                                     save_path: str = None):
    """
    Plot optimal strategies for multiple hashing power values.

    Args:
        q_values: List of hashing power values
        gamma: Connectivity parameter
        max_a: Maximum attacker blocks
        max_h: Maximum honest blocks
        save_path: Path to save figure
    """
    n_plots = len(q_values)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    for idx, (ax, q) in enumerate(zip(axes, q_values)):
        optimal = OptimalSelfishMining(q, gamma)
        strategy_matrix = optimal.get_strategy_matrix(max_a, max_h)

        im = ax.imshow(strategy_matrix, cmap=cmap, aspect='auto',
                      origin='lower', vmin=0, vmax=3, interpolation='nearest')

        # Add text annotations
        for a in range(max_a + 1):
            for h in range(max_h + 1):
                action = optimal.get_optimal_action(a, h)
                action_short = action[0].upper()
                color = 'white' if strategy_matrix[a, h] in [0, 3] else 'black'
                ax.text(h, a, action_short, ha='center', va='center',
                       color=color, fontsize=8, fontweight='bold')

        ax.set_xlabel('h', fontsize=11)
        ax.set_ylabel('a', fontsize=11)
        ax.set_title(f'q={q:.2f}', fontsize=12, fontweight='bold')

        ax.set_xticks(range(0, max_h + 1, 2))
        ax.set_yticks(range(0, max_a + 1, 2))

        ax.set_xticks(np.arange(-0.5, max_h + 1, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, max_a + 1, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

    # Add overall title
    fig.suptitle(f'Optimal Selfish Mining Strategies (γ={gamma})',
                fontsize=14, fontweight='bold', y=1.02)

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], label='Adopt'),
        Patch(facecolor=colors[1], label='Wait'),
        Patch(facecolor=colors[2], label='Match'),
        Patch(facecolor=colors[3], label='Override')
    ]
    fig.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1.0, 0.5), fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()


def generate_all_plots(output_dir: str = 'results'):
    """
    Generate all plots and save them to the output directory.

    Args:
        output_dir: Directory to save all plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating all plots...")

    # Plot 1: Strategy comparison for different gamma values
    print("\n1. Generating strategy comparison plots...")
    q_values = np.linspace(0.05, 0.45, 15)

    for gamma in [0.0, 0.5, 1.0]:
        plot_strategy_comparison(
            q_values, num_blocks=10000, num_simulations=50, gamma=gamma,
            save_path=f'{output_dir}/strategy_comparison_gamma_{gamma:.1f}.png'
        )

    # Plot 2: Profitability heatmap
    print("\n2. Generating profitability heatmap...")
    plot_profitability_heatmap(
        q_range=(0.0, 0.5), gamma_range=(0.0, 1.0), resolution=100,
        save_path=f'{output_dir}/profitability_heatmap.png'
    )

    # Plot 3: Profitability boundary
    print("\n3. Generating profitability boundary curves...")
    plot_profitability_boundary_curves(
        save_path=f'{output_dir}/profitability_boundary.png'
    )

    # Plot 4: Optimal strategy matrices for single configurations
    print("\n4. Generating optimal strategy matrices...")
    configurations = [
        (0.25, 0.0),
        (0.25, 0.5),
        (0.35, 0.5),
        (0.40, 1.0)
    ]

    for q, gamma in configurations:
        plot_optimal_strategy_matrix(
            q, gamma, max_a=15, max_h=15,
            save_path=f'{output_dir}/optimal_strategy_q_{q:.2f}_gamma_{gamma:.1f}.png'
        )

    # Plot 5: Multiple optimal strategies comparison
    print("\n5. Generating multiple optimal strategies comparison...")
    plot_multiple_optimal_strategies(
        q_values=[0.2, 0.3, 0.4], gamma=0.5,
        save_path=f'{output_dir}/optimal_strategies_comparison.png'
    )

    print(f"\nAll plots saved to {output_dir}/")
