#%%
import jax.numpy as jnp
import h5py
import matplotlib.pyplot as plt
import glob as glob

#%%
data_folder = "data/sims/"
files = glob.glob(data_folder + "*.h5")

# Plot Baseline Data
u0_idx = 2

# with h5py.File(f"data/{u0_idx}_noiseless.h5", "r") as f:
# 	u = f["u"][:]

# 	trj = jnp.stack(u)
# 	plt.figure(figsize=(20, 5))
# 	plt.imshow(
# 		trj.T,
# 		cmap="RdBu",
# 		aspect="auto",
# 		origin="lower",
# 		extent=(0, trj.shape[0] * 0.1, 0, 100)
# 	)
# 	plt.colorbar(label="u")
# 	plt.xlabel("Time")
# 	plt.ylabel("Space")
# 	plt.title("Original Data, No Noise")
# 	plt.savefig("figures/2_noiseless.png", dpi=300)
# 	#plt.show()
# 	plt.close()

# Load and concatenate data
data_perturbed = []
for file in enumerate(files):
	try:
		with h5py.File(fr'data/sims/data_{file[0]}.h5', "r") as f:
			u = f["u"][file[0], :]
			data_perturbed.append(u) #,x, t))
	except:
		print(f"Error loading file: data_{file[0]}")

data_10s = []
for i in range(0, 2000, 10):
	try:
		with h5py.File(fr'data/sims/data_{i}.h5', "r") as f:
			u_10s = f["u"][i:i+10, :]
			for u in u_10s:
				data_10s.append(u) #,x, t))
	except:
		print(f"Error loading file: data_{i}")

data_20s = []
for i in range(0, 2000, 20):
	try:
		with h5py.File(fr'data/sims/data_{i}.h5', "r") as f:
			u_20s = f["u"][i:i+20, :]
			for u in u_20s:
				data_20s.append(u) #,x, t))
	except:
		print(f"Error loading file: data_{i}")

# plot spliced Data
# trj = jnp.stack([data for data in data_perturbed])

# plt.figure(figsize=(20, 5))
# plt.imshow(
# 	trj.T,
# 	cmap="RdBu",
# 	aspect="auto",
# 	origin="lower",
# 	#extent=(0, trj.shape[0] * 0.1, 0, 100)
# )
# plt.colorbar(label="u")
# plt.xlabel("Time")
# plt.ylabel("Space")
# plt.title("Concatenated Data, every 1 steps")
# plt.savefig(f"figures/{u0_idx}_spliced.png", dpi=300)
# #plt.show()
# plt.close()

#trj = jnp.stack([data for data in data_10s])

# plt.figure(figsize=(20, 5))
# plt.imshow(
# 	trj.T,
# 	cmap="RdBu",
# 	aspect="auto",
# 	origin="lower",
# 	extent=(0, trj.shape[0] * 0.1, 0, 100)
# )
# plt.colorbar(label="u")
# plt.xlabel("Time")
# plt.ylabel("Space")
# plt.title("Concatenated Data, every 10 steps")
# plt.savefig(f"figures/{u0_idx}_spliced10s.png", dpi=300)
# #plt.show()
# plt.close()

#trj = jnp.stack([data for data in data_20s])

# plt.figure(figsize=(20, 5))
# plt.imshow(
# 	trj.T,
# 	cmap="RdBu",
# 	aspect="auto",
# 	origin="lower",
# 	extent=(0, trj.shape[0] * 0.1, 0, 100)
# )
# plt.colorbar(label="u")
# plt.xlabel("Time")
# plt.ylabel("Space")
# plt.title("Concatenated Data, every 20 steps")
# plt.savefig(f"figures/{u0_idx}_spliced20s.png", dpi=300)
# #plt.show()
# plt.close()


# Plot some of the data datasets to verify

# for i in [0, 1000, 1999]:

# 	with h5py.File(fr'data/sims/data_{i}.h5', "r") as f:
# 		u = f["u"][:]

# 		trj = jnp.stack(u)
# 		plt.figure(figsize=(20, 5))
# 		plt.imshow(
# 			trj.T,
# 			cmap="RdBu",
# 			aspect="auto",
# 			origin="lower",
# 			extent=(0, trj.shape[0] * 0.1, 0, 100)
# 		)
# 		plt.colorbar(label="u")
# 		plt.xlabel("time")
# 		plt.ylabel("space")
# 		plt.title(f"Original Data with Noise, {i+1} iteration")
# 		plt.savefig(fr"figures/2_data_{i}.png", dpi=300)
# 		#plt.show()
# 		plt.close()

# Plot first frame of the  noiseless and perturbation in 1D 
# comparison = []
# with h5py.File(fr'data/{u0_idx}_noiseless.h5', "r") as f:
# 	u = f["u"][:]
# 	x = f["x"][:]
# 	comparison.append(u[0, :]) #,x, t))
with h5py.File(fr'data/sims/data_0.h5', "r") as f:
	u = f["u"][:]
	x = f["x"][:]
# 	comparison.append(u[0, :]) #,x, t))

# 	plt.figure(figsize=(10, 5))
# 	plt.plot(x, comparison[1], label="Noisy", color="red", linestyle="dashed")
# 	plt.plot(x, comparison[0], label="Original", color="black")
# 	#plt.plot(x, u[1, :], label="After 1 Step", color="blue")
# 	#plt.plot(x, u[2, :], label="After 2 Steps", color="green")
# 	plt.xlabel("Space")
# 	plt.ylabel("u")
# 	plt.title("Initial Condition with and without Noise")
# 	plt.legend()
# 	plt.grid()
# 	plt.savefig(f"figures/{u0_idx}_noise_visualization.png", dpi=300)
# 	#plt.show()
# 	plt.close()

# Save concatenated data
with h5py.File(f"data/{u0_idx}_spliced.h5", "w") as f:
	f.create_dataset("u", data=jnp.stack([data for data in data_perturbed]))
	f.create_dataset("x", data=x)
	t_concatenated = jnp.arange(jnp.stack([data for data in data_perturbed]).shape[0]) * 0.1
	f.create_dataset("t", data=t_concatenated)

	f.attrs["domain_size"] = 100.0
	f.attrs["n_dof"] = 256
	f.attrs["dt"] = 0.1
	f.attrs["n_steps"] = jnp.stack([data for data in data_perturbed]).shape[0]

with h5py.File(f"data/{u0_idx}_spliced10s.h5", "w") as f:
	f.create_dataset("u", data=jnp.stack([data for data in data_10s]))
	f.create_dataset("x", data=x)
	t_concatenated = jnp.arange(jnp.stack([data for data in data_10s]).shape[0]) * 0.1
	f.create_dataset("t", data=t_concatenated)

	f.attrs["domain_size"] = 100.0
	f.attrs["n_dof"] = 256
	f.attrs["dt"] = 0.1
	f.attrs["n_steps"] = jnp.stack([data for data in data_10s]).shape[0]

with h5py.File(f"data/{u0_idx}_spliced20s.h5", "w") as f:
	f.create_dataset("u", data=jnp.stack([data for data in data_20s]))
	f.create_dataset("x", data=x)
	t_concatenated = jnp.arange(jnp.stack([data for data in data_20s]).shape[0]) * 0.1
	f.create_dataset("t", data=t_concatenated)

	f.attrs["domain_size"] = 100.0
	f.attrs["n_dof"] = 256
	f.attrs["dt"] = 0.1
	f.attrs["n_steps"] = jnp.stack([data for data in data_20s]).shape[0]