import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

files = [
    #"2_noiseless",
	#"2_roll_denoised_rollavg",
	#"2_spectral_denoised_rollavg",
	# "2_noisy_0.01_denoised_rollavg",
    "1_noiseless",
	"1_noisy_0.1",
	# "2_noisy_0.1_denoised_rollavg",
	# "2_spliced_denoised_rollavg",
	# "2_spliced10s_denoised_rollavg",
	# "2_spliced20s_denoised_rollavg",
]
frame_idx = 1650

fig, ax = plt.subplots(figsize=(10, 6))
for filename in files:
    with h5py.File(Path("data") / f"{filename}.h5", "r") as f:
        u = f["u"][:]

    frame = u[frame_idx, :]
    ax.plot(frame, label=filename)

# Title and labels
ax.set_title(f"Frame Comparison — dt {frame_idx}", fontsize=14)
ax.set_xlabel("x", fontsize=11)
ax.set_ylabel("u", fontsize=11)

# Place legend outside the plot to the right, use compact font and no frame
legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize=9)
legend.set_frame_on(False)

plt.tight_layout()
plt.show()

# Save a copy (optional)
out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / f"frame_comparisons.png", bbox_inches="tight", dpi=200)
plt.close(fig)