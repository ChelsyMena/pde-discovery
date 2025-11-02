import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler

filename = "1_translated_denoised"

with h5py.File(f"data/{filename}.h5", "r") as f:
    u_data = f["u"][:]
    x = [i for i in range(f["u"].shape[1])] #f["x"][:] 
    t = [i for i in range(f["u"].shape[0])] #f["t"][:]
    dt = 0.1 #f.attrs["dt"]
    dx = 100/200

n_train = min(5000, len(u_data))
u_train = u_data[:n_train, :]
t_train = t[:n_train]
#%%
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

#dx = x[1] - x[0]
#dt_train = t_train[1] - t_train[0]
dt_train = 0.1

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

# --- Build finite-difference (FD) derivative features for comparison ---
u_x_fd = (np.roll(u_train, -1, axis=1) - np.roll(u_train, 1, axis=1)) / (2 * dx)
u_xx_fd = (np.roll(u_train, -1, axis=1) - 2 * u_train + np.roll(u_train, 1, axis=1)) / (dx**2)
u_xxx_fd = (np.roll(u_xx_fd, -1, axis=1) - np.roll(u_xx_fd, 1, axis=1)) / (2 * dx)
u_xxxx_fd = (np.roll(u_xx_fd, -1, axis=1) - 2 * u_xx_fd + np.roll(u_xx_fd, 1, axis=1)) / (dx**2)
u_xxxxx_fd = (np.roll(u_xxxx_fd, -1, axis=1) - np.roll(u_xxxx_fd, 1, axis=1)) / (2 * dx)
u_xxxxxx_fd = (np.roll(u_xxxx_fd, -1, axis=1) - 2 * u_xxxx_fd + np.roll(u_xxxx_fd, 1, axis=1)) / (dx**2)

u_squared_fd = u_train**2
u_cubed_fd = u_train**3
u_squared_x_fd = (np.roll(u_squared_fd, -1, axis=1) - np.roll(u_squared_fd, 1, axis=1)) / (2 * dx)
u_squared_xx_fd = (np.roll(u_squared_fd, -1, axis=1) - 2 * u_squared_fd + np.roll(u_squared_fd, 1, axis=1)) / (dx**2)
u_cubed_x_fd = (np.roll(u_cubed_fd, -1, axis=1) - np.roll(u_cubed_fd, 1, axis=1)) / (2 * dx)
u_u_x_fd = u_train * u_x_fd
u_u_xx_fd = u_train * u_xx_fd

features_dict_fd = {
    'u': u_train.flatten(),
    'u_x': u_x_fd.flatten(),
    'u_xx': u_xx_fd.flatten(),
    'u_xxx': u_xxx_fd.flatten(),
    'u_xxxx': u_xxxx_fd.flatten(),
    'u_xxxxx': u_xxxxx_fd.flatten(),
    'u_xxxxxx': u_xxxxxx_fd.flatten(),
    'u²': u_squared_fd.flatten(),
    'u³': u_cubed_fd.flatten(),
    '(u²)_x': u_squared_x_fd.flatten(),
    '(u²)_xx': u_squared_xx_fd.flatten(),
    '(u³)_x': u_cubed_x_fd.flatten(),
    'u·u_x': u_u_x_fd.flatten(),
    'u·u_xx': u_u_xx_fd.flatten(),
}

X_fd = np.column_stack([features_dict_fd[key] for key in features_dict_fd.keys()])

# expose feature names and ground-truth key terms for L1 evaluation
feature_names = list(features_dict.keys())
# ground truth: prefer the (u^2)_x term name used in the feature library
key_terms = {'u_xx': -1.0, 'u_xxxx': -1.0, '(u²)_x': -0.5}


def compute_l1(coef, key_terms, feature_names):
    """Compute L1 norm between discovered coefficients and ground-truth key terms.

    If a key term is not present in feature_names, its discovered coef is treated as 0.
    """
    l1 = 0.0
    for name, expected in key_terms.items():
        if name in feature_names:
            idx = feature_names.index(name)
            discovered = coef[idx]
        else:
            discovered = 0.0
        l1 += abs(discovered - expected)
    return l1

# Training with different methods
print("\n" + "="*80)
print("COMPARING DIFFERENT METHODS")
print("="*80)

results = []

# --- scale features and target (StandardScaler) for both spectral and FD features ---
X_scaler_spec = StandardScaler()
X_scaler_fd = StandardScaler()
yscaler = StandardScaler()

# fit scalers
Xs_spec = X_scaler_spec.fit_transform(X)
Xs_fd = X_scaler_fd.fit_transform(X_fd)
ys = yscaler.fit_transform(y.reshape(-1, 1)).ravel()

# store scaler params for unscaling coefficients later
X_mean_spec = X_scaler_spec.mean_
X_scale_spec = X_scaler_spec.scale_
X_mean_fd = X_scaler_fd.mean_
X_scale_fd = X_scaler_fd.scale_
y_mean = yscaler.mean_[0]
y_scale = yscaler.scale_[0]

# datasets to evaluate: (label, original X, scaled Xs, X_mean, X_scale)
datasets = [
    ('Spectral', X, Xs_spec, X_mean_spec, X_scale_spec),
    ('FD', X_fd, Xs_fd, X_mean_fd, X_scale_fd),
]

rcond_values = [None, 1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
alphas = [0.001, 0.01, 0.1, 1.0]
thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]

for ds_name, X_orig, Xs_ds, X_mean_ds, X_scale_ds in datasets:
    print(f"\n---- Running methods on {ds_name} features ----")

    # OLS grid over rcond
    print("\nOLS (varying rcond):")
    for r in rcond_values:
        if r is None:
            coef_s = np.linalg.lstsq(Xs_ds, ys, rcond=None)[0]
        else:
            coef_s = np.dot(np.linalg.pinv(Xs_ds, rcond=r), ys)

        coef = coef_s * (y_scale / X_scale_ds)
        intercept = y_mean - np.dot(X_mean_ds, coef)

        y_pred = X_orig @ coef + intercept
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        n_nonzero = np.sum(np.abs(coef) > 1e-6)
        l1 = compute_l1(coef, key_terms, feature_names)
        print(f"  {ds_name} rcond={r}: RMSE={rmse:.6e}, L1={l1:.6e}, nonzero={n_nonzero}")
        results.append((f'OLS({ds_name}, rcond={str(r)})', coef, rmse, l1, intercept))

    # Ridge alphas
    print("\nRidge (varying alpha):")
    for alpha in alphas:
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        coef_s_ridge = ridge.fit(Xs_ds, ys).coef_

        coef_ridge = coef_s_ridge * (y_scale / X_scale_ds)
        intercept = y_mean - np.dot(X_mean_ds, coef_ridge)

        y_pred_ridge = X_orig @ coef_ridge + intercept
        rmse_ridge = np.sqrt(np.mean((y - y_pred_ridge) ** 2))
        n_nonzero = np.sum(np.abs(coef_ridge) > 1e-6)
        l1 = compute_l1(coef_ridge, key_terms, feature_names)
        print(f"  {ds_name} Alpha={alpha}: RMSE={rmse_ridge:.6f}, L1={l1:.6e}, Non-zero terms={n_nonzero}")
        results.append((f'Ridge({ds_name}, α={alpha})', coef_ridge, rmse_ridge, l1, intercept))

    # Adaptive Lasso (iterative reweighted Lasso)
    print("\nAdaptive Lasso (reweighted Lasso):")
    try:
        # initial estimator (use LassoCV first; fallback to Ridge if needed)
        try:
            init = LassoCV(cv=5, fit_intercept=False, n_jobs=-1, max_iter=5000).fit(Xs_ds, ys)
            coef_init_s = init.coef_
            init_alpha = init.alpha_
        except Exception:
            # fallback to tiny Ridge for initial weights
            init_alpha = None
            coef_init_s = Ridge(alpha=1e-6, fit_intercept=False).fit(Xs_ds, ys).coef_

        # adaptive weights: 1 / (|beta_init| + eps)^gamma
        eps = 1e-6
        gamma = 1.0
        weights = 1.0 / (np.abs(coef_init_s) + eps) ** gamma

        # weight the design matrix columns (divide columns by weights)
        Xs_weighted = Xs_ds / weights[np.newaxis, :]

        # fit Lasso on weighted data
        lasso_w = LassoCV(cv=5, fit_intercept=False, n_jobs=-1, max_iter=5000)
        coef_s_w = lasso_w.fit(Xs_weighted, ys).coef_

        # recover scaled coefficients for original (unweighted) problem
        coef_s_adaptive = coef_s_w / weights

        # unscale to original units
        coef_adaptive = coef_s_adaptive * (y_scale / X_scale_ds)
        intercept = y_mean - np.dot(X_mean_ds, coef_adaptive)

        y_pred_adaptive = X_orig @ coef_adaptive + intercept
        rmse_adaptive = np.sqrt(np.mean((y - y_pred_adaptive) ** 2))
        n_nonzero = np.sum(np.abs(coef_adaptive) > 1e-6)
        l1 = compute_l1(coef_adaptive, key_terms, feature_names)
        print(f"  {ds_name} AdaptiveLasso: RMSE={rmse_adaptive:.6f}, L1={l1:.6e}, Non-zero terms={n_nonzero}, init_alpha={init_alpha}")
        results.append((f'AdaptiveLasso({ds_name})', coef_adaptive, rmse_adaptive, l1, intercept))
    except Exception as e:
        print(f"  {ds_name} AdaptiveLasso failed: {e}")

    # STLSQ thresholds
    print("\nSTLSQ (thresholding):")
    for threshold in thresholds:
        coef_s = np.linalg.lstsq(Xs_ds, ys, rcond=None)[0]

        for iteration in range(20):
            mask = np.abs(coef_s) > threshold
            if np.sum(mask) == 0:
                coef_s = np.zeros_like(coef_s)
                break

            Xs_masked = Xs_ds[:, mask]
            coef_s_masked = np.linalg.lstsq(Xs_masked, ys, rcond=None)[0]

            coef_s_new = np.zeros_like(coef_s)
            coef_s_new[mask] = coef_s_masked

            if np.allclose(coef_s, coef_s_new, atol=1e-8):
                break
            coef_s = coef_s_new

        coef = coef_s * (y_scale / X_scale_ds)
        intercept = y_mean - np.dot(X_mean_ds, coef)

        y_pred = X_orig @ coef + intercept
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        n_terms = np.sum(np.abs(coef) > 1e-10)
        l1 = compute_l1(coef, key_terms, feature_names)
        print(f"  {ds_name} Threshold={threshold}: RMSE={rmse:.6f}, L1={l1:.6e}, terms={n_terms}")
        results.append((f'STLSQ({ds_name}, τ={threshold})', coef, rmse, l1, intercept))

# Select best model - PREFER SPARSE MODELS
print("\n" + "="*80)
print("SELECTING BEST MODEL...")
print("="*80)

# Sort by L1 (sum absolute error on key terms)
results.sort(key=lambda x: x[3])

# Show top 5 models by L1
print("\nTop 5 models by L1 (sum abs error on key terms):")
for i, (name, coef, rmse, l1, *_rest) in enumerate(results[:5]):
    n_terms = np.sum(np.abs(coef) > 1e-10)
    print(f"{i+1}. {name:<25} L1={l1:.6e}, RMSE={rmse:.6f}, terms={n_terms}")

## SMART MODEL SELECTION: minimize objective = L1 + lambda_sparsity * n_terms
print("\n" + "="*80)
print("SMART MODEL SELECTION (objective = L1 + lambda * n_terms):")
print("="*80)

# tunable sparsity penalty (change as desired)
lambda_sparsity = 0.1

# compute objective for all results
scored = []
for (name, coef, rmse, l1, intercept) in results:
    n_terms = int(np.sum(np.abs(coef) > 1e-10))
    objective = l1 + lambda_sparsity * n_terms
    scored.append((name, coef, rmse, l1, intercept, n_terms, objective))

# show top 5 by objective
scored.sort(key=lambda x: x[6])
print("\nTop 5 models by objective (L1 + lambda*n_terms):")
for i, (name, coef, rmse, l1, intercept, n_terms, obj) in enumerate(scored[:5]):
    print(f"{i+1}. {name:<30} objective={obj:.6e}, L1={l1:.6e}, terms={n_terms}, RMSE={rmse:.6f}")

# choose best by objective
best_name, best_coef, best_rmse, best_l1, best_intercept, best_n_terms, best_obj = scored[0]
print(f"\n✓ Selected by objective: {best_name} (objective={best_obj:.6e}, L1={best_l1:.6e}, terms={best_n_terms}, RMSE={best_rmse:.6f})")
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
        
        # if step % 100 == 0:
        #     max_val = np.max(np.abs(u))
        #     if max_val > 10 or np.any(np.isnan(u)):
        #         print(f"  ⚠ Instability detected at step {step}, max_val={max_val:.2f}")
        #         break
    
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

vmin, vmax = -3, 3

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
axes[2].set_title(f"Error (L1 = {best_l1:.4f}, RMSE={rmse_final:.4f})", fontsize=15, fontweight='bold')
axes[2].set_xlabel("Time Snapshot Index", fontsize=12)
axes[2].set_ylabel("Space", fontsize=12)
plt.colorbar(im3, ax=axes[2], label="Error")

plt.tight_layout()
plt.savefig(f"figures/{filename}_sindy.png", dpi=300, bbox_inches='tight')
#print(f"\nSaved: sindy_ks_discovered.png")
plt.show()