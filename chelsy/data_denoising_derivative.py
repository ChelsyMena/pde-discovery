#import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
#from scipy.signal import correlate

filename = "2_noisy"

with h5py.File(f"data/{filename}.h5", "r") as f:
	u = f["u"][:]

new_u = u[0:1,:]
for dt in range(1, len(u)):
	#u_base = u[dt - 1, :]
	u_base = new_u[-1, :]
	u_noisy = u[dt, :]

	#minimize derivatives
	best_du = jnp.inf
	best_du_shift = 0
	
	for x in range(len(u_base)):
		du = (jnp.roll(u_noisy, x) - u_base) / 0.1
		#du = (jnp.roll(u_noisy, x) - u_base)**2 # minimize distances by color
		sum_du = jnp.mean(jnp.abs(du))
		if sum_du < best_du:
			best_du = sum_du
			best_du_shift = x
	
	u_corrected = jnp.roll(u_noisy, best_du_shift)
	#print(f"Time {dt}: best shift = {best_du_shift}, best du sum = {best_du}")
	new_u = jnp.vstack([new_u, u_corrected[None, :]])

	# correlation = correlate(u_noisy, u_base, mode='same', method='fft')
	# best_shift = np.argmax(correlation) - len(u_base) // 2
	# u_corrected = jnp.roll(u_noisy, best_shift)
	# print(f"Time {dt}: best shift = {best_shift}")
	# new_u = jnp.vstack([new_u, u_corrected[None, :]])

# save denoised data
with h5py.File(f"data/{filename}_denoised.h5", "w") as f:
	f.create_dataset("u", data=new_u)

# plot original data and denoised data
plt.figure(figsize=(20, 15))
plt.subplot(3, 1, 1)
plt.imshow(u.T, aspect='auto', cmap='RdBu')
plt.title("Original Data")

plt.subplot(3, 1, 2)
plt.imshow(new_u.T, aspect='auto', cmap='RdBu')
plt.title("Denoised Data")
plt.tight_layout()

plt.subplot(3, 1, 3)
plt.imshow((u-new_u).T, aspect='auto', cmap='RdBu')
plt.title("Difference")
plt.tight_layout()
plt.savefig(f"figures/{filename}_denoising.png")
plt.show()