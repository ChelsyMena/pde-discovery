import numpy as np
import jax
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt

trj = None
with h5py.File("perturbed/8_data_noiseless.h5", "r") as f:
	u = f["u"][:]

	u_new = []
	u_simple_new = []
	for dt in u:
		new = dt + 0.05*jnp.std(dt)*jax.random.normal(jax.random.PRNGKey(0),shape=dt.shape)
		n_index = np.random.randint(0, 15) #dt.shape[0])
		new = jnp.concatenate([new[n_index:], new[0:n_index]])
		u_new.append(new)

		simple_new = jnp.concatenate([dt[n_index:], dt[0:n_index]])
		u_simple_new.append(simple_new)

	trj2 = jnp.stack(u_new)
	trj = trj2.copy()

	trj3 = jnp.stack(u_simple_new)
	trj_simple = trj3.copy()

# save noisy data
with h5py.File(f"perturbed/8_data_noisy.h5", "w") as f:
	f.create_dataset("u", data=trj)
	# f.create_dataset("x", data=f["x"][:])
	# f.create_dataset("t", data=f["t"][:])

	# f.attrs["domain_size"] = f.attrs["domain_size"]
	# f.attrs["n_dof"] = f.attrs["n_dof"]
	# f.attrs["dt"] = f.attrs["dt"]
	# f.attrs["n_steps"] = trj.shape[0]

with h5py.File(f"perturbed/8_data_simpletraslating.h5", "w") as f:
	f.create_dataset("u", data=trj_simple)

#Plot noisy data
plt.figure(figsize=(20, 5))
plt.imshow(
	trj.T,
	cmap="RdBu",
	aspect="auto",
	origin="lower",
	extent=(0, trj.shape[0] * 0.1, 0, 100)
)
plt.colorbar(label="u")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title("Original Data, Translation and different Noise in each dt")
plt.savefig("figures/8_noisy.png", dpi=300)
#plt.show()
plt.close()