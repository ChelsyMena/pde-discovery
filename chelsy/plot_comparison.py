import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# two files to compare (first is reference, second is the comparison)
files = [
    "1_noiseless",
    "1_noisy_0.1_denoised_color",
]

# load arrays
data_dir = Path("data")
paths = [data_dir / f"{name}.h5" for name in files]
arrays = []
for p in paths:
    with h5py.File(p, "r") as f:
        arrays.append(f["u"][:2000])

if len(arrays) != 2:
    raise RuntimeError(f"Expected 2 files, got {len(arrays)}")

u_ref, u_cmp = arrays

# ensure shapes match
if u_ref.shape != u_cmp.shape:
    raise ValueError(f"Shape mismatch: reference {u_ref.shape} vs comparison {u_cmp.shape}")

# compute metrics
diff = u_ref - u_cmp
rmse = float(np.sqrt(np.mean((diff) ** 2)))
ss_res = float(np.sum((diff) ** 2))
ss_tot = float(np.sum((u_ref - u_ref.mean()) ** 2))
if ss_tot == 0:
    r2 = float('nan')
else:
    r2 = 1.0 - (ss_res / ss_tot)

print(rmse)

# # plot reference, comparison, and absolute difference
#fig, axes = plt.subplots(1, 1, figsize=(12, 5))

# axes[0].imshow(u_ref.T, aspect='auto', cmap='RdBu')
# axes[0].set_title(f"{files[0]}")

# axes[1].imshow(u_cmp.T, aspect='auto', cmap='RdBu')
# axes[1].set_title(f"{files[1]}")

#im = axes.imshow(np.abs(diff).T, aspect='auto', cmap='Reds')
#cbar = fig.colorbar(im, ax=axes, orientation='vertical', pad=0.02)
#cbar.set_label('Absolute Difference', rotation=270, labelpad=12)
#axes.set_title(f"Absolute difference — RMSE={rmse:.4g}")

# colorbar for difference
#fig.colorbar(im, ax=axes[2], orientation='vertical')

# Overlapped plot (reference and comparison side-by-side transparency)
#fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# Use two different colormaps with transparency to show both datasets
# im1 = axes[2].imshow(u_ref.T, aspect='auto', cmap='Blues', alpha=0.6, label=files[0])
# im2 = axes[2].imshow(u_cmp.T, aspect='auto', cmap='Reds', alpha=0.6, label=files[1])

# axes[2].set_title(f"Overlapped: {files[0]} (blue) vs {files[1]} (red)\nRMSE={rmse:.4g}, R²={r2:.4g}")
# axes[2].set_xlabel("Time index")
# axes[2].set_ylabel("Space index")

# # Create custom legend
# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor='blue', alpha=0.6, label=files[0]),
#                     Patch(facecolor='red', alpha=0.6, label=files[1])]
# axes[2].legend(handles=legend_elements, loc='upper right')

# plt.tight_layout()
# plt.savefig(f"figures/{files[1]}_vs_{files[0]}_comparison.pdf", dpi=300)
# plt.show()