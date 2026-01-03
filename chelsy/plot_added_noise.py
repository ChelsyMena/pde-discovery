import numpy as np
import jax
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt

u0_idx = 1
frame_to_plot = 1650
noise_levels = [0.01, 0.05, 0.1]

# Create figure with 3 rows, 1 column
fig, axes = plt.subplots(3, 1, figsize=(6, 8))

# Load original data once
with h5py.File(f"data/{u0_idx}_noiseless.h5", "r") as f:
    u_original = f["u"][:]
    x = f["x"][:]

original_frame = u_original[frame_to_plot]

# Plot each noise level
for idx, noise_level in enumerate(noise_levels):
    with h5py.File(f"data/{u0_idx}_noiseless.h5", "r") as f:
        u = f["u"][:]
        
        # Add noise to the original data
        noisy_frame = original_frame + noise_level * np.std(original_frame) * np.random.randn(original_frame.shape[0])
        
        # Plot
        ax = axes[idx]
        ax.plot(x, original_frame, 'k-', linewidth=1, label='Original')
        ax.plot(x, noisy_frame, 'r--', linewidth=1, label=f'Noisy (σ={noise_level})')
        ax.set_ylabel('u')
        ax.set_xlabel('x')
        ax.set_title(f'Noise level: {noise_level}')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/comparison_noise_levels.pdf', dpi=300)
plt.show()