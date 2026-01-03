import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

files = [
	"2_roll",
	"2_spectral",
	"2_noisy_0.01",
	"2_noisy_0.05",
	"2_noisy_0.1",
	"2_spliced",
	"2_spliced10s",
	"2_spliced20s",
]

for filename in files:
	print(f"\nDenoising file: {filename}\n")

	with h5py.File(f"data/{filename}.h5", "r") as f:
		u = f["u"][:]

	new_u = u[0:1,:]
	for dt in tqdm(range(1, len(u))):
		u_base = new_u[-1, :]
		u_noisy = u[dt, :]
		
		# print(f"Denoising time step {dt}...")
		window_size = 3
		u_noisy = jnp.convolve(u_noisy, jnp.ones(window_size) / window_size, mode='same')
		
			# rolling max by absolute value over a symmetric window (preserve sign of max-abs element)
			# N = u_noisy.shape[0]
			# pad = window_size // 2
			# # pad using reflection to avoid artificial wrap-around
			# u_pad = jnp.pad(u_noisy, (pad, pad), mode='reflect')
			# # build sliding windows: shape (window_size, N)
			# windows = jnp.stack([u_pad[i : i + N] for i in range(window_size)], axis=0)
			# # choose the element with maximum absolute value in each column
			# idx = jnp.argmax(jnp.abs(windows), axis=0)  # shape (N,)
			# # pick values preserving sign
			# u_noisy = jnp.take_along_axis(windows, idx[None, :], axis=0).squeeze(0)

		# detranslate

		#minimize derivatives
		best_du = jnp.inf
		best_du_shift = 0
		
		for x in range(len(u_base)):
			du = (jnp.roll(u_noisy, x) - u_base) / 0.1
			sum_du = jnp.mean(jnp.abs(du))
			if sum_du < best_du:
				best_du = sum_du
				best_du_shift = x
		
		u_corrected = jnp.roll(u_noisy, best_du_shift)
		# print(f"Time {dt}: best shift = {best_du_shift}, best du sum = {best_du}")
		new_u = jnp.vstack([new_u, u_corrected[None, :]])

	# save denoised data
	with h5py.File(f"data/{filename}_denoised_rollavg.h5", "w") as f:
		f.create_dataset("u", data=new_u)