import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import csv
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# config
# --------------------
FILENAME = "1_noiseless"
TOL = 1e-1            # threshold for reporting coeffs
PHYS_WEIGHT = 10.0    # weight on physics loss
L1_LAMBDA = 1e-4      # L1 sparsity strength
N_ITERS = 20000
PRINT_EVERY = 1000
BATCH_SIZE = 4096
NUM_FREQUENCIES = 16  # Fourier features
LOG_EVERY = 100  # how often to write a CSV row (set to 1 to log every iter)

# --------------------
# load data
# --------------------
with h5py.File(f"data/{FILENAME}.h5", "r") as f:
    u_np = np.array(f["u"][:])            # (T, X)
    x_np = np.array(f["x"][:]) if "x" in f else None
    t_np = np.array(f["t"][:]) if "t" in f else None

T, N = u_np.shape
if x_np is None:
    x_np = np.linspace(0.0, 1.0, N, endpoint=False)
if t_np is None:
    t_np = np.arange(T, dtype=float)

print(f"Loaded data: u {u_np.shape}, x {x_np.shape}, t {t_np.shape}")

# --------------------
# build full spacetime dataset and normalize t,x
# --------------------
T_grid, X_grid = np.meshgrid(t_np, x_np, indexing="ij")  # (T, N)

t_mean, t_std = T_grid.mean(), T_grid.std()
x_mean, x_std = X_grid.mean(), X_grid.std()

T_norm = (T_grid - t_mean) / t_std
X_norm = (X_grid - x_mean) / x_std

tx = np.stack([T_norm.ravel(), X_norm.ravel()], axis=-1).astype(np.float32)  # (T*N, 2)
u_vals = u_np.ravel().astype(np.float32)

tx_tensor = torch.from_numpy(tx).to(device)
u_tensor = torch.from_numpy(u_vals).to(device)

dataset_size = tx_tensor.shape[0]
print("Flattened dataset size:", dataset_size)

# --------------------
# Fourier feature layer
# --------------------
class FourierFeatures(nn.Module):
    def __init__(self, in_dim=2, num_frequencies=16, sigma=5.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.register_buffer("B", B)  # (2, num_frequencies)

    def forward(self, x):
        # x: (B, 2) normalized (t_norm, x_norm)
        proj = x @ self.B            # (B, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, 2*num_frequencies)

ff = FourierFeatures(in_dim=2, num_frequencies=NUM_FREQUENCIES).to(device)
ff_out_dim = 2 * NUM_FREQUENCIES

# --------------------
# MLP on top of Fourier features
# --------------------
class BigMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        sizes = [in_dim, 128, 128, 128, 128, 1]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers.append(nn.Linear(a, b))
            layers.append(nn.Tanh())
        layers = layers[:-1]  # remove last activation
        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (B,)

torch.manual_seed(0)
net = BigMLP(ff_out_dim).to(device)

# --------------------
# PDE library and coeffs (KS-style up to 4th derivative)
# --------------------
term_names = [
    'u', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx',
    'u²', 'u³', '(u²)_x', '(u²)_xx', '(u³)_x', #'u·u_x'
]

coeffs = nn.Parameter(torch.zeros(len(term_names), dtype=torch.float32, device=device))

# --------------------
# autograd derivatives (physical coords)
# --------------------
def derivatives(u_out, tx_batch, t_std, x_std):
    """
    u_out: (B,) predictions u(t,x)
    tx_batch: (B, 2) normalized (t_norm, x_norm) with requires_grad_(True)
    returns: u_t, u_x, u_xx, u_xxx, u_xxxx in *physical* coordinates
    """
    grads = torch.autograd.grad(
        u_out, tx_batch,
        grad_outputs=torch.ones_like(u_out),
        create_graph=True,
        retain_graph=True
    )[0]  # (B, 2): derivatives wrt normalized t,x

    u_t_norm = grads[:, 0]
    u_x_norm = grads[:, 1]

    u_xx_norm = torch.autograd.grad(
        u_x_norm, tx_batch,
        grad_outputs=torch.ones_like(u_x_norm),
        create_graph=True,
        retain_graph=True
    )[0][:, 1]

    u_xxx_norm = torch.autograd.grad(
        u_xx_norm, tx_batch,
        grad_outputs=torch.ones_like(u_xx_norm),
        create_graph=True,
        retain_graph=True
    )[0][:, 1]

    u_xxxx_norm = torch.autograd.grad(
        u_xxx_norm, tx_batch,
        grad_outputs=torch.ones_like(u_xxx_norm),
        create_graph=True,
        retain_graph=True
    )[0][:, 1]

    # rescale: t_phys = t_mean + t_std * t_norm → ∂/∂t_phys = (1/t_std) ∂/∂t_norm
    #          x_phys = x_mean + x_std * x_norm → ∂/∂x_phys^k = (1/x_std^k) ∂/∂x_norm^k
    u_t    = u_t_norm    / t_std
    u_x    = u_x_norm    / x_std
    u_xx   = u_xx_norm   / (x_std**2)
    u_xxx  = u_xxx_norm  / (x_std**3)
    u_xxxx = u_xxxx_norm / (x_std**4)

    return u_t, u_x, u_xx, u_xxx, u_xxxx

def build_phi_terms(u_pred, u_x, u_xx, u_xxx, u_xxxx):
    """
    Build feature library Φ (B, n_terms) matching term_names order.
    """
    phi = torch.stack([
        u_pred,                         # u
        u_x,                            # u_x
        u_xx,                           # u_xx
        u_xxx,                          # u_xxx
        u_xxxx,                         # u_xxxx
        u_pred ** 2,                    # u²
        u_pred ** 3,                    # u³
        2.0 * u_pred * u_x,             # (u²)_x
        2.0 * (u_x**2 + u_pred * u_xx), # (u²)_xx
        3.0 * u_pred**2 * u_x,          # (u³)_x
        #u_pred * u_x                    # u·u_x
    ], dim=1)
    return phi

# --------------------
# optimizer
# --------------------
optimizer = optim.Adam(list(net.parameters()) + [coeffs], lr=3e-4)

# --------------------
# training: data + physics + L1 (three losses)
# --------------------

# CSV logging setup: one row per iteration with losses and coefficients
log_path = f"pinns/pinn_{FILENAME}_training_ignore.csv"
with open(log_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["iter", "loss", "data_loss", "phys_loss", "l1_pen"] + term_names
    writer.writerow(header)

    for it in range(N_ITERS):
        idx = np.random.choice(dataset_size, size=BATCH_SIZE, replace=False)
        txb = tx_tensor[idx].clone().detach().requires_grad_(True)  # (B, 2) normalized
        ub = u_tensor[idx]                                          # (B,)

        # forward
        txb_ff = ff(txb)          # (B, 2*NUM_FREQUENCIES)
        u_pred = net(txb_ff)      # (B,)

        # derivatives in physical coords
        u_t, u_x, u_xx, u_xxx, u_xxxx = derivatives(u_pred, txb, t_std, x_std)

        # feature matrix and PDE model
        phi = build_phi_terms(u_pred, u_x, u_xx, u_xxx, u_xxxx)   # (B, n_terms)
        u_t_model = phi @ coeffs                                  # (B,)

        # losses
        data_loss = torch.mean((u_pred - ub) ** 2)
        phys_loss = torch.mean((u_t - u_t_model) ** 2)
        l1_pen    = L1_LAMBDA * torch.norm(coeffs, p=1)

        loss = data_loss + PHYS_WEIGHT * phys_loss + l1_pen

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # prepare row (records updated coefficients)
        coeffs_vals = coeffs.detach().cpu().numpy().tolist()
        row = [it, loss.item(), data_loss.item(), phys_loss.item(), l1_pen.item()] + coeffs_vals

        # write CSV only at configured frequency to reduce I/O, but always write final iter
        if (it % LOG_EVERY == 0) or (it == N_ITERS - 1):
            writer.writerow(row)

        # flush and print periodically
        if it % PRINT_EVERY == 0:
            csvfile.flush()
            print(
                f"iter {it}, "
                f"loss {loss.item():.4e}, "
                f"data {data_loss.item():.4e}, "
                f"phys {phys_loss.item():.4e}, "
                f"l1 {l1_pen.item():.4e}"
            )

# --------------------
# report discovered coefficients
# --------------------
coeffs_final = coeffs.detach().cpu().numpy().copy()
coeffs_final[np.abs(coeffs_final) <= TOL] = 0.0
result: Dict[str, float] = {name: float(coeffs_final[i]) for i, name in enumerate(term_names)}

print("\nDiscovered PDE coefficients (|c| <= {:.1e} set to 0):".format(TOL))
for k, v in result.items():
    print(f"  {k:10s}: {v:.6g}")

# Save the trained model
model_path = f"pinns/pinn_{FILENAME}_model.pt"
metadata = {
    'net_state_dict': net.state_dict(),
    'coeffs': coeffs.detach().cpu().numpy().copy(),
    'term_names': term_names,
    'ff_B': ff.B.cpu().numpy(),  # Fourier feature matrix
    'NUM_FREQUENCIES': NUM_FREQUENCIES,
    't_mean': float(t_mean),
    't_std': float(t_std),
    'x_mean': float(x_mean),
    'x_std': float(x_std),
    'TOL': TOL,
    'FILENAME': FILENAME,
}
torch.save(metadata, model_path)
print(f"\nModel saved to {model_path}")


# --------------------
# Plot from CSV log (low memory) - losses and coefficient trajectories
# --------------------
# import numpy as _np
# import matplotlib.pyplot as plt

# log_path = f"pinns/pinn_{FILENAME}_training.csv"
# try:
#     with open(log_path, "r") as f:
#         reader = csv.reader(f)
#         header = next(reader)
#         rows = list(reader)

#     if len(rows) == 0:
#         print(f"No rows found in {log_path}, skipping plots")
#     else:
#         data = _np.array([[float(val) for val in row] for row in rows])
#         iters = data[:, 0].astype(int)
#         loss_hist = data[:, 1]
#         data_hist = data[:, 2]
#         phys_hist = data[:, 3]
#         l1_hist = data[:, 4]
#         coeffs_arr = data[:, 5:]

#         # losses plot
#         plt.figure(figsize=(9, 5))
#         plt.plot(iters, loss_hist, label='total')
#         plt.plot(iters, data_hist, label='data')
#         plt.plot(iters, phys_hist, label='phys')
#         plt.plot(iters, l1_hist, label='l1')
#         plt.yscale('log')
#         plt.xlabel('Iteration')
#         plt.ylabel('Loss (log scale)')
#         plt.legend()
#         plt.title(f'PINN Training Losses ({FILENAME})')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'pinns/pinn_{FILENAME}_losses.png', dpi=150)
#         plt.close()

#         # coefficients plot
#         plt.figure(figsize=(10, 6))
#         highlight = {'u_xx', 'u_xxxx', '(u²)_x'}
#         n_terms = coeffs_arr.shape[1]
#         for i in range(n_terms):
#             name = term_names[i]
#             series = coeffs_arr[:, i]
#             if name in highlight:
#                 plt.plot(iters, series, linewidth=2.2, label=name)
#             else:
#                 plt.plot(iters, series, color='0.6', linewidth=0.8, alpha=0.7)
#         # add horizontal dotted guide lines at -0.5 and -1.0
#         plt.axhline(-0.5, color='green', linestyle='--', linewidth=1, alpha=0.9)
#         plt.axhline(-1.0, color='green', linestyle='--', linewidth=1, alpha=0.9)

#         # annotate final coefficient values for highlighted terms
#         try:
#             final_vals = coeffs_arr[-1, :]
#             x_pos = iters[-1]
#             for name in highlight:
#                 if name in term_names:
#                     idx = term_names.index(name)
#                     val = final_vals[idx]
#                     # place a small offset text to the right of the last datapoint
#                     plt.annotate(f"{val:.3g}", xy=(x_pos, val), xytext=(6, 0),
#                                  textcoords='offset points', va='center', ha='left', fontsize=9,
#                                  bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
#         except Exception:
#             # if something goes wrong (e.g., empty array), skip annotations
#             pass

#         plt.xlabel('iteration')
#         plt.ylabel('Coefficient Value')
#         plt.title(f'Coefficient trajectories ({FILENAME})')
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(f'pinns/pinn_{FILENAME}_coeffs.png', dpi=150)
#         plt.close()
# except FileNotFoundError:
#     print(f"Log file {log_path} not found — skipping plots.")
