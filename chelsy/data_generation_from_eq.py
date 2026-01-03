#%% DATA GENERATION

import jax
import jax.numpy as jnp
from matplotlib.patheffects import Normal
import matplotlib.pyplot as plt
import h5py
import csv
import numpy as np
from pathlib import Path

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
files = [
    "1_noiseless",
]

for filename in files:
    with h5py.File(Path("data") / f"{filename}.h5", "r") as f:
        u_0 = f["u"][:1]

    # Normal Generation
    u_current = u_0
    trj = [u_current]
    for i in range(2000):
        u_current = ks_stepper(u_current)
        trj.append(u_current)

    trj = jnp.stack(trj)

    x = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
    t = jnp.arange(trj.shape[0]) * DT

    with h5py.File(f"data/{filename}_sindy.h5", "w") as f:
        f.create_dataset("u", data=trj)
        f.create_dataset("x", data=x)
        f.create_dataset("t", data=t)

        f.attrs["domain_size"] = DOMAIN_SIZE
        f.attrs["n_dof"] = N_DOF
        f.attrs["dt"] = DT
        f.attrs["n_steps"] = trj.shape[0]


def _parse_sindy_csv(csv_path: Path):
    """Read sindy_results.csv (semicolon-delimited, comma decimals) and yield rows.

    Returns list of dicts with keys: filename (str), model_name (str), coeffs (dict feature->float)
    """
    rows = []
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"sindy CSV not found: {csv_path}")
        return rows

    # open with cp1252 (windows-1252) to tolerate special characters like ± (0xB1)
    with csv_path.open('r', newline='', encoding='cp1252') as fh:
        reader = csv.reader(fh, delimiter=';', quotechar='"')
        header = next(reader)
        # strip potential BOM from first header field
        if header and isinstance(header[0], str):
            header[0] = header[0].lstrip('\ufeff')
        for rr in reader:
            # map header->value
            data = {h: v for h, v in zip(header, rr)}
            # model name and filename
            filename = data.get('filename')
            model_name = data.get('model_name') if 'model_name' in data else None

            # parse numeric coefficients: convert comma decimals to dot
            coeffs = {}
            for h in header:
                if h in ('filename', 'model_name', 'L1'):
                    continue
                v = data.get(h, '')
                if v is None or v == '':
                    coeffs[h] = 0.0
                else:
                    # replace comma decimal with dot and remove quotes
                    vs = v.strip().strip('"').replace(',', '.')
                    try:
                        coeffs[h] = float(vs)
                    except Exception:
                        coeffs[h] = 0.0

            rows.append({'filename': filename, 'model_name': model_name, 'coeffs': coeffs, 'raw': data})

    return rows


def _make_stepper_from_coeffs(coeffs: dict, N: int, dx: float, dt: float, dealias=True):
    """Return a numpy-based ETD step function built from SINDy coefficients.

    coeffs: mapping feature name -> value (float)
    N, dx, dt: spatial num points, spacing, timestep
    """
    freqs = np.fft.rfftfreq(N, d=dx)
    k = 2.0 * np.pi * freqs
    deriv = 1j * k

    # build linear operator on rfft grid
    linear_op = np.zeros_like(k, dtype=np.complex128)
    linear_orders = {
        'u': 0,
        'u_x': 1,
        'u_xx': 2,
        'u_xxx': 3,
        'u_xxxx': 4,
        'u_xxxxx': 5,
        'u_xxxxxx': 6,
    }
    for name, order in linear_orders.items():
        c = coeffs.get(name, 0.0)
        if c != 0.0:
            linear_op += c * ((1j * k) ** order)

    # ETD factors (stable)
    z = dt * linear_op
    # clip real/imag parts to avoid overflow
    z_clipped = np.clip(z.real, -50.0, 50.0) + 1j * np.clip(z.imag, -50.0, 50.0)
    exp_term = np.exp(z_clipped)
    coef_etd = np.empty_like(linear_op, dtype=np.complex128)
    small_mask = np.abs(linear_op) < 1e-12
    coef_etd[small_mask] = dt
    safe_idx = ~small_mask
    coef_etd[safe_idx] = np.expm1(z_clipped[safe_idx]) / linear_op[safe_idx]

    alias_mask = (freqs <= (2.0 / 3.0) * freqs.max()) if dealias else np.ones_like(freqs, dtype=bool)

    def step(u: np.ndarray):
        # u: real-valued array length N
        nonlinear_hat = np.zeros_like(k, dtype=np.complex128)

        def add_phys(arr, factor=1.0):
            fh = np.fft.rfft(arr)
            nonlinear_hat[:] += factor * fh

        # simple nonlinear features
        if abs(coeffs.get('u²', 0.0)) > 0.0:
            add_phys(u ** 2, coeffs['u²'])
        if abs(coeffs.get('u³', 0.0)) > 0.0:
            add_phys(u ** 3, coeffs['u³'])
        if abs(coeffs.get('(u²)_x', 0.0)) > 0.0:
            fh = np.fft.rfft(u ** 2)
            nonlinear_hat[:] += coeffs['(u²)_x'] * (deriv * fh)
        if abs(coeffs.get('(u²)_xx', 0.0)) > 0.0:
            fh = np.fft.rfft(u ** 2)
            nonlinear_hat[:] += coeffs['(u²)_xx'] * ((deriv ** 2) * fh)
        if abs(coeffs.get('(u³)_x', 0.0)) > 0.0:
            fh = np.fft.rfft(u ** 3)
            nonlinear_hat[:] += coeffs['(u³)_x'] * (deriv * fh)
        if abs(coeffs.get('u·u_x', 0.0)) > 0.0:
            ux = np.fft.irfft(deriv * np.fft.rfft(u), n=N).real
            add_phys(u * ux, coeffs['u·u_x'])
        if abs(coeffs.get('u·u_xx', 0.0)) > 0.0:
            uxx = np.fft.irfft((deriv ** 2) * np.fft.rfft(u), n=N).real
            add_phys(u * uxx, coeffs['u·u_xx'])

        nonlinear_hat = nonlinear_hat * alias_mask

        u_hat = np.fft.rfft(u)
        u_hat_new = exp_term * u_hat + coef_etd * nonlinear_hat
        u_new = np.fft.irfft(u_hat_new, n=N).real
        return u_new

    return step


def generate_from_sindy_csv(csv_path: str = "sindy_results.csv", n_steps: int = 2000, dt: float = DT, domain_size: float = DOMAIN_SIZE):
    rows = _parse_sindy_csv(Path(csv_path))
    if len(rows) == 0:
        print("No rows found in sindy CSV; nothing to do.")
        return

    for row in rows:
        file_key = row['filename']
        if file_key is None or file_key == '':
            print("Skipping row without filename")
            continue

        data_path = Path("data") / f"{file_key}.h5"
        if not data_path.exists():
            print(f"Data file not found for {file_key}: {data_path}; skipping")
            continue

        with h5py.File(data_path, 'r') as f:
            # take first time snapshot as initial condition (match existing code lines 82-88)
            u0 = np.array(f['u'][0]).ravel()

        N = u0.size
        dx = domain_size / N
        coeffs = row['coeffs']

        stepper = _make_stepper_from_coeffs(coeffs, N=N, dx=dx, dt=dt, dealias=True)

        u = u0.copy()
        traj = [u.copy()]
        print(f"Generating {n_steps} steps for model {row.get('model_name')} from file {file_key}")
        for i in range(n_steps):
            u = stepper(u)
            traj.append(u.copy())

        traj = np.stack(traj)

        out_path = Path('data') / f"{file_key}_sindy.h5"
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('u', data=traj)
            f.create_dataset('x', data=np.linspace(0.0, domain_size, N, endpoint=False))
            f.create_dataset('t', data=np.arange(traj.shape[0]) * dt)
            f.attrs['domain_size'] = domain_size
            f.attrs['n_dof'] = N
            f.attrs['dt'] = dt
            f.attrs['n_steps'] = traj.shape[0]

        print(f"Saved simulation to {out_path} shape={traj.shape}")


def generate_and_plot_from_dict(coeffs_dict: dict, ref_filename: str, n_steps: int = 2000, dt: float = DT, domain_size: float = DOMAIN_SIZE):
    """
    Generate a trajectory from a coefficient dictionary and plot it against reference data.
    
    Args:
        coeffs_dict: mapping feature name -> coefficient value (e.g., {'u_xx': -1.0, 'u_xxxx': -1.0, ...})
        ref_filename: basename of reference file in data/ (e.g., "1_noiseless") to load initial condition
        n_steps: number of time steps to generate
        dt: time step size
        domain_size: spatial domain size
    """
    # load reference data and extract initial condition
    data_path = Path("data") / f"{ref_filename}.h5"
    if not data_path.exists():
        print(f"Reference data file not found: {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        u_ref = np.array(f['u'][:])  # full reference trajectory
        x = np.array(f['x'][:]) if 'x' in f else np.linspace(0.0, domain_size, u_ref.shape[1], endpoint=False)
        t_ref = np.array(f['t'][:]) if 't' in f else np.arange(u_ref.shape[0]) * dt
        u0 = u_ref[0].copy()  # first snapshot as initial condition
    
    N = u0.size
    dx = domain_size / N
    
    # build stepper from coefficient dictionary
    stepper = _make_stepper_from_coeffs(coeffs_dict, N=N, dx=dx, dt=dt, dealias=True)
    
    # generate trajectory
    u = u0.copy()
    traj = [u.copy()]
    print(f"Generating {n_steps} steps from coefficient dict")
    for i in range(n_steps):
        u = stepper(u)
        traj.append(u.copy())
    
    traj = np.stack(traj)
    t_gen = np.arange(traj.shape[0]) * dt
    
    # plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # reference
    im0 = axes[0].contourf(x, t_ref[:min(len(t_ref), n_steps+1)], u_ref[:min(len(u_ref), n_steps+1)], levels=20, cmap='RdBu_r')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('t')
    axes[0].set_title(f'Reference: {ref_filename}')
    plt.colorbar(im0, ax=axes[0])
    
    # generated
    im1 = axes[1].contourf(x, t_gen, traj, levels=20, cmap='RdBu_r')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('t')
    axes[1].set_title('Generated from Dict')
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(f'pinns/generated_vs_ref_{ref_filename}.png', dpi=150)
    plt.close()
    
    print(f"Plotted and saved to pinns/generated_vs_ref_{ref_filename}.png")
    return traj


if __name__ == '__main__':
    # If run as script, generate datasets for all rows in the CSV
    # generate_from_sindy_csv(
    #     csv_path='sindy_results.csv', 
    #     n_steps=2000, dt=DT, domain_size=DOMAIN_SIZE)

    found_eq = {
        'u': 0.0,
        'u_x': 0.0,
        'u_xx': -0.198359,
        'u_xxx': 0.0,
        'u_xxxx': -0.193881,
        'u_xxxxx': 0.0,
        'u_xxxxxx': 0.0,
        'u²': 0.0,
        'u³': 0.0,
        '(u²)_x': -0.102369,
        '(u²)_xx': 0.0,
        '(u³)_x': 0.0,
        'u·u_x': -0.063583,
        'u·u_xx': 0.0,
    }

    generate_and_plot_from_dict(found_eq, ref_filename="1_roll_denoised_color", n_steps=2000)