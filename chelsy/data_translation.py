import jax
import jax.numpy as jnp
import numpy as np
import h5py

u0_idx = 2

def perturb_spectral_shift(
    trj: jnp.ndarray,
    L: float,
    shift_std_px: float = 1.0,
    seed: int = 0,
):
    """
    Пертурбація через фазовий зсув у Фур'є-просторі
    (smooth, sub-pixel shifts)
    """
    T, N = trj.shape
    dx = L / N

    freqs = jnp.fft.rfftfreq(N, d=dx)
    k = 2.0 * jnp.pi * freqs

    key = jax.random.PRNGKey(seed)
    shifts_px = shift_std_px * jax.random.normal(key, (T,))
    shifts_dist = shifts_px * dx 

    phase = jnp.exp(-1j * jnp.outer(shifts_dist, k))
    
    Uhat = jnp.fft.rfft(trj, axis=1)
    Uhat_pert = Uhat * phase
    trj_perturbed = jnp.fft.irfft(Uhat_pert, n=N, axis=1)
    
    return trj_perturbed, shifts_px


def perturb_spatial_roll(
    trj: jnp.ndarray,
    shift_std_px: float = 5.0,
    seed: int = 0,
):
    """
    Пертурбація через циклічний зсув (roll)
    (discrete, integer pixel shifts)
    """
    T, N = trj.shape
    
    key = jax.random.PRNGKey(seed)
    shifts_px = shift_std_px * jax.random.normal(key, (T,))
    shifts_px = jnp.round(shifts_px).astype(int)
    
    def roll_row(carry, inputs):
        row, shift = inputs
        rolled = jnp.roll(row, shift)
        return carry, rolled
    
    _, trj_perturbed = jax.lax.scan(
        roll_row, 
        None, 
        (trj, shifts_px)
    )
    
    return trj_perturbed, shifts_px

# Run it

with h5py.File(f'data/{u0_idx}_noiseless.h5', 'r') as f:
    trj = jnp.array(f['u'][:])
    x = jnp.array(f['x'][:])
    t = jnp.array(f['t'][:])
    DOMAIN_SIZE = f.attrs['domain_size']
    N_DOF = f.attrs['n_dof']

SHIFT_STD_1 = 10.0
SHIFT_STD_2 = 10.0

trj_pert1, shifts1 = perturb_spectral_shift(
    trj, 
    L=DOMAIN_SIZE, 
    shift_std_px=SHIFT_STD_1, 
    seed=42
)

trj_pert2, shifts2 = perturb_spatial_roll(
    trj, 
    shift_std_px=SHIFT_STD_2, 
    seed=43
)

# save
with h5py.File(f'data/{u0_idx}_spectral.h5', 'w') as f:
    # Original data
    f.create_dataset('u', data=trj_pert1)
    #f.create_dataset('u_original', data=trj)  # Дублюємо для сумісності
    f.create_dataset('x', data=x)
    f.create_dataset('t', data=t)
    
with h5py.File(f'data/{u0_idx}_roll.h5', 'w') as f:
    # Original data
    f.create_dataset('u', data=trj_pert2)
    #f.create_dataset('u_original', data=trj)  # Дублюємо для сумісності
    f.create_dataset('x', data=x)
    f.create_dataset('t', data=t)