"""Generate profitability heatmap for Task 2.2"""
import numpy as np
import matplotlib.pyplot as plt
from mining_strategies import SelfishMining
import os

print("Generating profitability heatmap...")
print("=" * 60)

os.makedirs('results', exist_ok=True)

# Define parameter ranges
q_values = np.linspace(0.05, 0.45, 20)
gamma_values = np.linspace(0.0, 1.0, 20)

print(f"Testing {len(q_values)} x {len(gamma_values)} = {len(q_values) * len(gamma_values)} configurations...")

# Compute profitability matrix
profitability = np.zeros((len(gamma_values), len(q_values)))
revenue_advantage = np.zeros((len(gamma_values), len(q_values)))

for i, gamma in enumerate(gamma_values):
    print(f"  γ = {gamma:.2f}...")
    for j, q in enumerate(q_values):
        selfish = SelfishMining(q, gamma)

        # Use simulation to get revenue
        result = selfish.simulate(num_blocks=3000, seed=42)
        selfish_rev = result.relative_revenue
        honest_rev = q

        advantage = selfish_rev - honest_rev
        revenue_advantage[i, j] = advantage
        profitability[i, j] = 1 if advantage > 0 else 0

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Profitability regions
im1 = ax1.imshow(profitability, extent=[q_values[0], q_values[-1],
                                        gamma_values[0], gamma_values[-1]],
                 origin='lower', aspect='auto', cmap='RdYlGn', alpha=0.8)

ax1.set_xlabel('Hashing Power (q)', fontsize=12)
ax1.set_ylabel('Connectivity (γ)', fontsize=12)
ax1.set_title('Selfish Mining Profitability Regions', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)

cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Profitable (1) vs Not Profitable (0)', fontsize=10)

# Add text annotations
ax1.text(0.15, 0.9, 'SELFISH MINING\nPROFITABLE',
         fontsize=11, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.text(0.15, 0.1, 'HONEST MINING\nBETTER',
         fontsize=11, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

# Plot 2: Revenue advantage heatmap
vmax = max(abs(revenue_advantage.min()), abs(revenue_advantage.max()))
im2 = ax2.imshow(revenue_advantage, extent=[q_values[0], q_values[-1],
                                             gamma_values[0], gamma_values[-1]],
                 origin='lower', aspect='auto', cmap='RdBu_r',
                 vmin=-vmax/2, vmax=vmax/2)

ax2.set_xlabel('Hashing Power (q)', fontsize=12)
ax2.set_ylabel('Connectivity (γ)', fontsize=12)
ax2.set_title('Revenue Advantage: Selfish - Honest', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)

cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Revenue Advantage', fontsize=10)

# Add contour at zero
contour = ax2.contour(q_values, gamma_values, revenue_advantage,
                      levels=[0], colors='black', linewidths=2.5, linestyles='--')
ax2.clabel(contour, inline=True, fontsize=10, fmt='Threshold')

plt.tight_layout()
plt.savefig('results/task_2_2_profitability_heatmap.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: results/task_2_2_profitability_heatmap.png")

print("\n" + "=" * 60)
print("Profitability heatmap generated successfully!")
