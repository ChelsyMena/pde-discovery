import numpy as np
import jax
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt

files = {
	"1_roll_denoised_color": "Spatial Roll + Forwards and Backwards Alignment",
	"1_spectral_denoised_rollavg": "Spectral Roll + Denoising & Derivatives",
    "1_noisy_0.01_denoised_derivative": "Spatial Roll & 1% Noise + Derivatives",
    "1_noisy_0.05_denoised_rollavg": "Spatial Roll & 5% Noise + Denoising & Derivatives",
    "1_noisy_0.1_denoised_color": "Spatial Roll & 10% Noise + Forwards and Backwards Alignment",
    "1_spliced_denoised_color": "Different Initial Conditions + Forwards and Backwards Alignment",
}

# Create a 2x3 grid (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 7))
axes = axes.flatten()  # Flatten to easily iterate

for idx, (filename, description) in enumerate(files.items()):
	with h5py.File(f"data/{filename}.h5", "r") as f:
		u = f["u"][:]

	ax = axes[idx]
	im = ax.imshow(
		u.T,
		cmap="RdBu",
		aspect="auto",
		origin="lower",
	)
	ax.set_title(f"{description}", fontsize=10)
	# ax.set_xlabel("Time")
	# ax.set_ylabel("Space")

# Set common labels on outer axes only
fig.text(0.5, 0.02, 'Time', ha='center', fontsize=12)
fig.text(0.02, 0.5, 'Space', va='center', rotation='vertical', fontsize=12)

plt.tight_layout(rect=[0.03, 0.03, 1, 1])
plt.savefig(f"figures/all_datasets_grid.pdf", dpi=300, bbox_inches='tight')
#plt.show()
plt.close()