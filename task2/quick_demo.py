"""Quick demonstration of key results"""
import numpy as np
import matplotlib.pyplot as plt
from mining_strategies import HonestMining, SelfishMining, OptimalSelfishMining

print("Generating quick demonstration plots...")
print("=" * 60)

# Create output directory
import os
os.makedirs('results', exist_ok=True)

# Task 2.1: Strategy comparison for γ=0.5
print("\n1. Generating strategy comparison (γ=0.5)...")
q_values = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
gamma = 0.5

honest_theory = []
selfish_theory = []
honest_sim = []
selfish_sim = []

for q in q_values:
    print(f"   Testing q={q:.2f}...")
    honest = HonestMining(q)
    selfish = SelfishMining(q, gamma)

    # Theory (for honest it's simple, for selfish we use simulation)
    honest_theory.append(q)
    selfish_theory.append(selfish.theoretical_revenue())

    # Simulation
    h_result = honest.simulate(num_blocks=5000, seed=42)
    s_result = selfish.simulate(num_blocks=5000, seed=42)

    honest_sim.append(h_result.relative_revenue)
    selfish_sim.append(s_result.relative_revenue)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Revenues
ax1.plot(q_values, honest_theory, 'b-', linewidth=2, label='Honest (Theory=q)')
ax1.plot(q_values, selfish_theory, 'r-', linewidth=2, label=f'Selfish (Simulation-based)')
ax1.plot(q_values, honest_sim, 'bo', alpha=0.6, markersize=8, label='Honest (Sim)')
ax1.plot(q_values, selfish_sim, 'ro', alpha=0.6, markersize=8, label='Selfish (Sim)')

ax1.set_xlabel('Hashing Power (q)', fontsize=12)
ax1.set_ylabel('Relative Revenue', fontsize=12)
ax1.set_title(f'Mining Strategy Performance (γ={gamma})', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Advantage
advantage_theory = np.array(selfish_theory) - np.array(honest_theory)
advantage_sim = np.array(selfish_sim) - np.array(honest_sim)

ax2.plot(q_values, advantage_theory, 'g-', linewidth=2, label='Theory/Simulation')
ax2.plot(q_values, advantage_sim, 'go', alpha=0.6, markersize=8, label='Direct Simulation')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax2.fill_between(q_values, 0, advantage_theory, where=np.array(advantage_theory)>0,
                 alpha=0.3, color='green', label='Selfish Profitable')
ax2.fill_between(q_values, advantage_theory, 0, where=np.array(advantage_theory)<0,
                 alpha=0.3, color='red', label='Honest Better')

ax2.set_xlabel('Hashing Power (q)', fontsize=12)
ax2.set_ylabel('Revenue Advantage (Selfish - Honest)', fontsize=12)
ax2.set_title(f'Selfish Mining Advantage (γ={gamma})', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/task_2_1_strategy_comparison.png', dpi=300, bbox_inches='tight')
print(f"   Saved: results/task_2_1_strategy_comparison.png")
plt.close()

# Task 2.3: Optimal strategy matrix
print("\n2. Generating optimal strategy matrix (q=0.35, γ=0.5)...")
q = 0.35
gamma = 0.5
optimal = OptimalSelfishMining(q, gamma)

max_a = 10
max_h = 10
strategy_matrix = optimal.get_strategy_matrix(max_a, max_h)

colors = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4']
cmap = plt.matplotlib.colors.ListedColormap(colors)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(strategy_matrix, cmap=cmap, aspect='auto',
               origin='lower', vmin=0, vmax=3, interpolation='nearest')

# Add text annotations
for a in range(max_a + 1):
    for h in range(max_h + 1):
        action = optimal.get_optimal_action(a, h)
        action_short = action[0].upper()
        color = 'white' if strategy_matrix[a, h] in [0, 3] else 'black'
        ax.text(h, a, action_short, ha='center', va='center',
               color=color, fontsize=10, fontweight='bold')

ax.set_xlabel('Honest Chain Length (h)', fontsize=13)
ax.set_ylabel('Attacker Chain Length (a)', fontsize=13)
ax.set_title(f'Optimal Selfish Mining Strategy (q={q}, γ={gamma})',
            fontsize=14, fontweight='bold')

ax.set_xticks(range(max_h + 1))
ax.set_yticks(range(max_a + 1))

ax.set_xticks(np.arange(-0.5, max_h + 1, 1), minor=True)
ax.set_yticks(np.arange(-0.5, max_a + 1, 1), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors[0], label='Adopt: Abandon private fork'),
    Patch(facecolor=colors[1], label='Wait: Continue mining'),
    Patch(facecolor=colors[2], label='Match: Create race'),
    Patch(facecolor=colors[3], label='Override: Publish fork')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
         bbox_to_anchor=(1.02, 1))

plt.tight_layout()
plt.savefig('results/task_2_3_optimal_strategy.png', dpi=300, bbox_inches='tight')
print(f"   Saved: results/task_2_3_optimal_strategy.png")
plt.close()

print("\n" + "=" * 60)
print("Demo plots generated successfully!")
print("Check the results/ directory for outputs.")
