#%% DATA GENERATION

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py

#%% SetUp

DOMAIN_SIZE = 100.0
N_DOF = 200
DT = 0.1

class KuramotoSivashinsky():
    def __init__(
        self,
        L,
        N,
        dt,
        a=-1.0,
        b=-1.0,
        c=-0.5,
    ):
        self.L = L
        self.N = N
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c

        self.dx = L / N

        # Frequencies (cycles per unit length)
        freqs = jnp.fft.rfftfreq(N, d=self.dx)
        # Convert to angular wavenumbers k = 2pi * freq
        k = 2 * jnp.pi * freqs
        self.derivative_operator = 1j * k

        # Linear operator in Fourier space: a*d^2/dx^2 + b*d^4/dx^4
        linear_operator = self.a * (-k**2) + self.b * (k**4)

        self.exp_term = jnp.exp(dt * linear_operator)
        self.coef = jnp.where(
            linear_operator == 0.0,
            dt,
            (self.exp_term - 1.0) / linear_operator,
        )

        # 2/3 rule dealiasing mask for nonlinear term
        self.alias_mask = (freqs < 2/3 * jnp.max(freqs))

    def __call__(self, u):
        # Nonlinear term c * d/dx (u^2)
        u_nonlin = self.c * u**2
        u_hat = jnp.fft.rfft(u)
        u_nonlin_hat = jnp.fft.rfft(u_nonlin)
        u_nonlin_hat = self.alias_mask * u_nonlin_hat

        u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat

        # Exponential time differencing step
        u_next_hat = self.exp_term * u_hat + self.coef * u_nonlin_der_hat
        u_next = jnp.fft.irfft(u_next_hat, n=self.N)

        return u_next

# Create spatial mesh
mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)

ks_stepper = KuramotoSivashinsky(
    L=DOMAIN_SIZE,
    N=N_DOF,
    dt=DT,
    a=-1.0,
    b=-1.0,
    c=-0.5,
	)
ks_stepper = jax.jit(ks_stepper)

# Initial Conditions
#u_0 = 0.5*jnp.sin(16 * jnp.pi * mesh / DOMAIN_SIZE) # 1-low noise sigma
#u_0 = 0.5 * jnp.sin(16 * jnp.pi * mesh / DOMAIN_SIZE) #
u_0 = jnp.sin(16 * jnp.pi * mesh / DOMAIN_SIZE)/10 # 4
#u_0 = 0.5*jnp.exp(-100 * (mesh - DOMAIN_SIZE / 2)**2) #
#u_0 = jnp.sin(2 * jnp.pi * mesh / DOMAIN_SIZE) + 0.5 * jnp.sin(4 * jnp.pi * mesh / DOMAIN_SIZE) + 0.25 * jnp.sin(8 * jnp.pi * mesh / DOMAIN_SIZE) #2-low
#u_0 = jnp.where(mesh < DOMAIN_SIZE / 2, 1.0, -1.0) # 5
#u_0 = 0.25 * jnp.sin(64 * jnp.pi * mesh / DOMAIN_SIZE) # 7
#u_0 = jnp.cos(3 * jnp.pi * mesh / DOMAIN_SIZE + 0.3)/10 # 3-low

#u_0 = 0.5 * jnp.exp(-100 * (mesh - DOMAIN_SIZE / 2)**2)  #Neurips

u_0 = jnp.sin(16 * jnp.pi * mesh / DOMAIN_SIZE) #6

#%% Normal Generation
u_current = u_0
trj = [u_current]
for i in range(2000):
	u_current = ks_stepper(u_current)
	trj.append(u_current)

trj = jnp.stack(trj)

x = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
t = jnp.arange(trj.shape[0]) * DT

with h5py.File(f"data/7_noiseless.h5", "w") as f:
	f.create_dataset("u", data=trj)
	f.create_dataset("x", data=x)
	f.create_dataset("t", data=t)

	f.attrs["domain_size"] = DOMAIN_SIZE
	f.attrs["n_dof"] = N_DOF
	f.attrs["dt"] = DT
	f.attrs["n_steps"] = trj.shape[0]

#%% Generation with Noise
j=0
while j < 2000:
	sigma = jnp.std(u_0)
	u_noise = u_0 + 0.05*sigma*jax.random.normal(jax.random.PRNGKey(j), shape=mesh.shape)

	# Autoregressive rollout for 200000 steps
	u_current = u_noise
	trj = [u_current]
	for i in range(2000):
		u_current = ks_stepper(u_current)
		trj.append(u_current)

	trj = jnp.stack(trj)

	x = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
	t = jnp.arange(trj.shape[0]) * DT

	with h5py.File(f"data/sims/data_{j}.h5", "w") as f:
		f.create_dataset("u", data=trj)
		f.create_dataset("x", data=x)
		f.create_dataset("t", data=t)

		f.attrs["domain_size"] = DOMAIN_SIZE
		f.attrs["n_dof"] = N_DOF
		f.attrs["dt"] = DT
		f.attrs["n_steps"] = trj.shape[0]
	
	print(f"Finished {j}.h5")
	j += 1