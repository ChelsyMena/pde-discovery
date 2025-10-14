import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

with h5py.File("perturbed/8_data_concatenated_20s.h5", "r") as f:
    u_data = f["u"][:]
    x = f["x"][:]
    t = f["t"][:]
    dt = f.attrs["dt"]

n_train = min(5000, len(u_data))
u_train = u_data[:n_train, :]
t_train = t[:n_train]

print(f"\nTraining data size: {u_train.shape}")
print(f"Time range: {t_train[0]:.1f} - {t_train[-1]:.1f}")

print("\n" + "="*80)
print("METHOD: SINDY WITH FULL LIBRARY DISCOVERY")
print("="*80)

def spectral_derivative(u, dx, order=1):
    n = u.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    u_hat = np.fft.fft(u, axis=-1)
    u_der_hat = (1j * k)**order * u_hat
    u_der = np.fft.ifft(u_der_hat, axis=-1).real
    return u_der

dx = x[1] - x[0]
dt_train = t_train[1] - t_train[0]

u_dot = np.zeros_like(u_train)
u_dot[1:-1] = (u_train[2:] - u_train[:-2]) / (2 * dt_train)
u_dot[0] = (u_train[1] - u_train[0]) / dt_train
u_dot[-1] = (u_train[-1] - u_train[-2]) / dt_train

u_x = spectral_derivative(u_train, dx, order=1)
u_xx = spectral_derivative(u_train, dx, order=2)
u_xxx = spectral_derivative(u_train, dx, order=3)
u_xxxx = spectral_derivative(u_train, dx, order=4)
u_xxxxx = spectral_derivative(u_train, dx, order=5)
u_xxxxxx = spectral_derivative(u_train, dx, order=6)

u_squared = u_train**2
u_cubed = u_train**3
u_squared_x = spectral_derivative(u_squared, dx, order=1)
u_squared_xx = spectral_derivative(u_squared, dx, order=2)
u_cubed_x = spectral_derivative(u_cubed, dx, order=1)
u_u_x = u_train * u_x
u_u_xx = u_train * u_xx

features_dict = {
    'u': u_train.flatten(),
    'u_x': u_x.flatten(),
    'u_xx': u_xx.flatten(),
    'u_xxx': u_xxx.flatten(),
    'u_xxxx': u_xxxx.flatten(),
    'u_xxxxx': u_xxxxx.flatten(),
    'u_xxxxxx': u_xxxxxx.flatten(),
    'u²': u_squared.flatten(),
    'u³': u_cubed.flatten(),
    '(u²)_x': u_squared_x.flatten(),
    '(u²)_xx': u_squared_xx.flatten(),
    '(u³)_x': u_cubed_x.flatten(),
    'u·u_x': u_u_x.flatten(),
    'u·u_xx': u_u_xx.flatten(),
}

X = np.column_stack([features_dict[key] for key in features_dict.keys()])
y = u_dot.flatten()

print(f"Feature matrix shape: {X.shape}")
print(f"Number of candidate terms: {len(features_dict)}")
print(f"Candidate terms: {list(features_dict.keys())}")

# Training with different methods
print("\n" + "="*80)
print("COMPARING DIFFERENT METHODS")
print("="*80)

results = []

# Method 1: Ordinary least squares
print("\n1. Ordinary Least Squares:")
coef_ols = np.linalg.lstsq(X, y, rcond=None)[0]
y_pred_ols = X @ coef_ols
rmse_ols = np.sqrt(np.mean((y - y_pred_ols)**2))
n_nonzero_ols = np.sum(np.abs(coef_ols) > 1e-6)
print(f"   RMSE: {rmse_ols:.6f}, Non-zero terms: {n_nonzero_ols}")
results.append(('OLS', coef_ols, rmse_ols))

# Method 2: Ridge regression (different alphas)
print("\n2. Ridge Regression:")
for alpha in [0.001, 0.01, 0.1, 1.0]:
    ridge = Ridge(alpha=alpha)
    coef_ridge = ridge.fit(X, y).coef_
    y_pred_ridge = X @ coef_ridge
    rmse_ridge = np.sqrt(np.mean((y - y_pred_ridge)**2))
    n_nonzero = np.sum(np.abs(coef_ridge) > 1e-6)
    print(f"   Alpha={alpha}: RMSE={rmse_ridge:.6f}, Non-zero terms={n_nonzero}")
    results.append((f'Ridge(α={alpha})', coef_ridge, rmse_ridge))

# Method 3: STLSQ with different thresholds
print("\n3. STLSQ (Sequential Thresholded Least Squares):")
for threshold in [0.001, 0.01, 0.05, 0.1, 0.2]:
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    
    for iteration in range(20):
        mask = np.abs(coef) > threshold
        if np.sum(mask) == 0:
            coef = np.zeros_like(coef)
            break
            
        X_masked = X[:, mask]
        coef_masked = np.linalg.lstsq(X_masked, y, rcond=None)[0]
        
        coef_new = np.zeros_like(coef)
        coef_new[mask] = coef_masked
        
        if np.allclose(coef, coef_new, atol=1e-8):
            break
        coef = coef_new
    
    y_pred = X @ coef
    rmse = np.sqrt(np.mean((y - y_pred)**2))
    n_terms = np.sum(np.abs(coef) > 1e-10)
    print(f"   Threshold={threshold}: RMSE={rmse:.6f}, terms={n_terms}")
    results.append((f'STLSQ(τ={threshold})', coef, rmse))

# Select best model - PREFER SPARSE MODELS
print("\n" + "="*80)
print("SELECTING BEST MODEL...")
print("="*80)

# Sort by RMSE
results.sort(key=lambda x: x[2])

# Show top 5 models
print("\nTop 5 models by RMSE:")
for i, (name, coef, rmse) in enumerate(results[:5]):
    n_terms = np.sum(np.abs(coef) > 1e-10)
    print(f"{i+1}. {name:<25} RMSE={rmse:.6f}, terms={n_terms}")

# MANUALLY SELECT BEST STLSQ MODEL (sparse and accurate)
print("\n" + "="*80)
print("SMART MODEL SELECTION (prefer sparse models):")
print("="*80)

best_name, best_coef, best_rmse = None, None, None

# First, try to find STLSQ model with 3-6 terms
for name, coef, rmse in results:
    if 'STLSQ' in name:
        n_terms = np.sum(np.abs(coef) > 1e-10)
        if 3 <= n_terms <= 6 and rmse < 0.01:  # Want sparse and accurate
            best_name, best_coef, best_rmse = name, coef, rmse
            print(f"✓ Selected: {name} with {n_terms} terms (RMSE={rmse:.6f})")
            break

# If no good STLSQ found, use STLSQ with threshold 0.05
if best_name is None:
    for name, coef, rmse in results:
        if 'STLSQ(τ=0.05)' in name or 'STLSQ(τ=0.1)' in name:
            best_name, best_coef, best_rmse = name, coef, rmse
            n_terms = np.sum(np.abs(coef) > 1e-10)
            print(f"✓ Selected: {name} with {n_terms} terms (RMSE={rmse:.6f})")
            break

# Fallback to best RMSE if nothing else works
if best_name is None:
    best_name, best_coef, best_rmse = results[0]
    print(f"⚠ Fallback to: {best_name}")

print(f"\n{'='*80}")
print(f"FINAL SELECTED MODEL: {best_name}")
print(f"{'='*80}")

feature_names = list(features_dict.keys())
print("\nDiscovered equation:")
print("du/dt = ", end="")
equation_parts = []
for name, coef_val in zip(feature_names, best_coef):
    if np.abs(coef_val) > 1e-10:  # Only show non-zero terms
        sign = "+" if coef_val >= 0 and equation_parts else ""
        equation_parts.append(f"{sign}{coef_val:.4f}·{name}")
print(" ".join(equation_parts) if equation_parts else "0")

print("\n" + "="*80)
print("DISCOVERED COEFFICIENTS:")
print("="*80)
for name, coef_val in zip(feature_names, best_coef):
    if np.abs(coef_val) > 1e-10:
        print(f"  {name:<12}: {coef_val:>10.6f}")

print("\n" + "="*80)
print("COMPARISON WITH ORIGINAL:")
print("="*80)
print("Original equation: du/dt = -1.0·u_xx - 1.0·u_xxxx - 0.5·(u²)_x")
print("\nKey coefficients comparison:")
key_terms = {'u_xx': -1.0, 'u_xxxx': -1.0, '(u²)_x': -0.5}
for name, expected in key_terms.items():
    idx = list(features_dict.keys()).index(name)
    discovered = best_coef[idx]
    error = abs(discovered - expected)
    print(f"  {name:<12}: discovered={discovered:>8.4f}, expected={expected:>6.1f}, error={error:.4f}")

# Simulation function
print("\n" + "="*80)
print("SIMULATION (ETD scheme)")
print("="*80)

def simulate_discovered_equation(u0, t_sim, coef, feature_names, dx, dt):
    """Simulate using discovered equation"""
    u = u0.copy()
    trajectory = [u.copy()]
    
    n = len(u)
    k_wave = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    
    # Identify linear terms for ETD
    linear_coef = np.zeros_like(k_wave)
    if 'u_xx' in feature_names:
        idx = feature_names.index('u_xx')
        if np.abs(coef[idx]) > 1e-10:
            linear_coef += coef[idx] * (-k_wave**2)
    if 'u_xxxx' in feature_names:
        idx = feature_names.index('u_xxxx')
        if np.abs(coef[idx]) > 1e-10:
            linear_coef += coef[idx] * (k_wave**4)
    
    exp_L = np.exp(dt * linear_coef)
    coef_etd = np.where(np.abs(linear_coef) < 1e-10, dt, (exp_L - 1) / linear_coef)
    
    for step in range(len(t_sim) - 1):
        # Compute all nonlinear terms
        nonlinear = np.zeros_like(u)
        
        for name, coef_val in zip(feature_names, coef):
            if np.abs(coef_val) < 1e-10:
                continue
            
            # Skip linear terms (already in ETD)
            if name in ['u_xx', 'u_xxxx']:
                continue
            
            # Compute nonlinear terms
            if name == 'u':
                nonlinear += coef_val * u
            elif name == 'u_x':
                nonlinear += coef_val * spectral_derivative(u, dx, 1)
            elif name == 'u_xxx':
                nonlinear += coef_val * spectral_derivative(u, dx, 3)
            elif name == 'u_xxxxx':
                nonlinear += coef_val * spectral_derivative(u, dx, 5)
            elif name == 'u_xxxxxx':
                nonlinear += coef_val * spectral_derivative(u, dx, 6)
            elif name == 'u²':
                nonlinear += coef_val * u**2
            elif name == 'u³':
                nonlinear += coef_val * u**3
            elif name == '(u²)_x':
                nonlinear += coef_val * spectral_derivative(u**2, dx, 1)
            elif name == '(u²)_xx':
                nonlinear += coef_val * spectral_derivative(u**2, dx, 2)
            elif name == '(u³)_x':
                nonlinear += coef_val * spectral_derivative(u**3, dx, 1)
            elif name == 'u·u_x':
                nonlinear += coef_val * u * spectral_derivative(u, dx, 1)
            elif name == 'u·u_xx':
                nonlinear += coef_val * u * spectral_derivative(u, dx, 2)
        
        # ETD step
        u_hat = np.fft.fft(u)
        nonlinear_hat = np.fft.fft(nonlinear)
        u_hat_new = exp_L * u_hat + coef_etd * nonlinear_hat
        u = np.fft.ifft(u_hat_new).real
        
        trajectory.append(u.copy())
        
        if step % 100 == 0:
            max_val = np.max(np.abs(u))
            if max_val > 10 or np.any(np.isnan(u)):
                print(f"  ⚠ Instability detected at step {step}, max_val={max_val:.2f}")
                break
    
    return np.array(trajectory)

print("Starting simulation...")
u_pred_sim = simulate_discovered_equation(u_train[0, :], t_train, best_coef, feature_names, dx, dt_train)

min_len = min(len(u_train), len(u_pred_sim))
u_train_cut = u_train[:min_len]
u_pred_sim_cut = u_pred_sim[:min_len]
t_train_cut = t_train[:min_len]

print(f"Simulation completed: {len(u_pred_sim)} steps")

# Visualization - ПРАВИЛЬНО показати індекси зрізів
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

vmin, vmax = -2, 2

im1 = axes[0].imshow(
    u_train_cut.T, cmap="RdBu", aspect="auto", origin="lower",
    extent=(0, len(t_train_cut)-1, x[0], x[-1]), vmin=vmin, vmax=vmax
)
axes[0].set_title("Original Data", fontsize=15, fontweight='bold')
axes[0].set_xlabel("Time Snapshot Index", fontsize=12)
axes[0].set_ylabel("Space", fontsize=12)
plt.colorbar(im1, ax=axes[0], label="u")

n_discovered = np.sum(np.abs(best_coef) > 1e-10)
im2 = axes[1].imshow(
    u_pred_sim_cut.T, cmap="RdBu", aspect="auto", origin="lower",
    extent=(0, len(t_train_cut)-1, x[0], x[-1]), vmin=vmin, vmax=vmax
)
axes[1].set_title(f"SINDy Prediction ({best_name}, {n_discovered} terms discovered)", fontsize=15, fontweight='bold')
axes[1].set_xlabel("Time Snapshot Index", fontsize=12)
axes[1].set_ylabel("Space", fontsize=12)
plt.colorbar(im2, ax=axes[1], label="u")

error = u_train_cut - u_pred_sim_cut
max_err = max(np.abs(error).max(), 0.1)
im3 = axes[2].imshow(
    error.T, cmap="seismic", aspect="auto", origin="lower",
    extent=(0, len(t_train_cut)-1, x[0], x[-1]), vmin=-max_err, vmax=max_err
)
rmse_final = np.sqrt(np.mean(error**2))
axes[2].set_title(f"Error (RMSE={rmse_final:.4f})", fontsize=15, fontweight='bold')
axes[2].set_xlabel("Time Snapshot Index", fontsize=12)
axes[2].set_ylabel("Space", fontsize=12)
plt.colorbar(im3, ax=axes[2], label="Error")

plt.tight_layout()
plt.savefig("sindy_ks_discovered.png", dpi=300, bbox_inches='tight')
print(f"\nSaved: sindy_ks_discovered.png")
plt.show()