# infer_pinn.py
import torch
import numpy as np
import h5py
from typing import Dict
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# Reconstruct layers
# --------------------
class FourierFeatures(torch.nn.Module):
    def __init__(self, B: np.ndarray):
        super().__init__()
        # B: (2, num_frequencies)
        self.register_buffer("B", torch.from_numpy(B).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B      # (B, num_frequencies)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class BigMLP(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        sizes = [in_dim, 128, 128, 128, 128, 1]
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers.append(torch.nn.Linear(a, b))
            layers.append(torch.nn.Tanh())
        layers = layers[:-1]  # remove last activation
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# --------------------
# Load trained model
# --------------------
def load_pinn_model(model_path: str):
    metadata = torch.load(model_path, map_location=device, weights_only=False)

    ff_B = metadata["ff_B"]
    num_freq = metadata["NUM_FREQUENCIES"]
    term_names = metadata["term_names"]
    coeffs = metadata["coeffs"]

    ff = FourierFeatures(ff_B).to(device)
    ff_out_dim = 2 * num_freq

    net = BigMLP(ff_out_dim).to(device)
    net.load_state_dict(metadata["net_state_dict"])
    net.eval()

    return net, ff, metadata

# --------------------
# Derivatives for inference (u_t only)
# --------------------
def infer_ut_autodiff(u_out: torch.Tensor,
                      tx_batch: torch.Tensor,
                      t_std: float) -> torch.Tensor:
    """
    Compute u_t in physical coordinates for a batch of points using autodiff.
    u_out: (B,) prediction
    tx_batch: (B,2) normalized coordinates with requires_grad_(True)
    """
    grads = torch.autograd.grad(
        u_out, tx_batch,
        grad_outputs=torch.ones_like(u_out),
        create_graph=False,
        retain_graph=False
    )[0]  # (B,2) derivatives wrt normalized t,x

    u_t_norm = grads[:, 0]
    u_t = u_t_norm / t_std
    return u_t

# --------------------
# Inference: predict u and u_t on data file
# --------------------
def predict_on_file(model_path: str,
                    data_filename: str,
                    output_file: str = None) -> Dict:
    """
    Load a trained PINN model and predict u and u_t on the grid from a data file.
    Note: model is solution-specific; predictions depend only on (t,x), not on u in the file.
    """
    net, ff, metadata = load_pinn_model(model_path)

    with h5py.File(data_filename, "r") as f:
        u_np = np.array(f["u"][:])            # just for shape/plotting
        x_np = np.array(f["x"][:]) if "x" in f else None
        t_np = np.array(f["t"][:]) if "t" in f else None

    T, N = u_np.shape
    if x_np is None:
        x_np = np.linspace(0.0, 1.0, N, endpoint=False)
    if t_np is None:
        t_np = np.arange(T, dtype=float)

    # normalization stats from training
    t_mean = metadata["t_mean"]
    t_std  = metadata["t_std"]
    x_mean = metadata["x_mean"]
    x_std  = metadata["x_std"]
    term_names = metadata["term_names"]
    coeffs_arr = metadata["coeffs"]

    # build normalized grid
    T_grid, X_grid = np.meshgrid(t_np, x_np, indexing="ij")
    T_norm = (T_grid - t_mean) / t_std
    X_norm = (X_grid - x_mean) / x_std

    tx = np.stack([T_norm.ravel(), X_norm.ravel()], axis=-1).astype(np.float32)
    tx_tensor = torch.from_numpy(tx).to(device)

    # predict u
    tx_tensor.requires_grad_(True)
    txb_ff = ff(tx_tensor)
    u_pred_t = net(txb_ff)        # (T*N,)
    u_pred = u_pred_t.detach().cpu().numpy().reshape(T, N)

    # predict u_t via autodiff
    u_t_pred_t = infer_ut_autodiff(u_pred_t, tx_tensor, t_std)
    u_t_pred = u_t_pred_t.detach().cpu().numpy().reshape(T, N)

    # package result
    result = {
        "u_pred": u_pred,
        "u_t_pred": u_t_pred,
        "x": x_np,
        "t": t_np,
        "coeffs": {name: float(c) for name, c in zip(term_names, coeffs_arr)},
    }

    # optional save
    if output_file is not None:
        with h5py.File(output_file, "w") as f:
            f.create_dataset("u_pred", data=u_pred)
            f.create_dataset("u_t_pred", data=u_t_pred)
            f.create_dataset("x", data=x_np)
            f.create_dataset("t", data=t_np)
            for name, coeff in result["coeffs"].items():
                f.attrs[name] = coeff
        print(f"Predictions saved to {output_file}")

    return result

# --------------------
# Example usage
# --------------------
if __name__ == "__main__":
    FILENAME_TRAIN = "1_noiseless"
    model_path = os.path.join("pinns", f"pinn_{FILENAME_TRAIN}_model.pt")

    # use same file the model was trained on (sanity check)
    test_filename = "1_noisy_0.01_denoised_rollavg"
    data_file = os.path.join("data", f"{test_filename}.h5")
    out_file  = os.path.join("pinns", f"predictions_{test_filename}.h5")

    result = predict_on_file(
        model_path=model_path,
        data_filename=data_file,
        output_file=out_file
    )

    print("\nDiscovered coefficients (from metadata):")
    for name, coeff in result["coeffs"].items():
        if abs(coeff) > 0:
            print(f"  {name}: {coeff:.6g}")

    # quick plot of predicted u
    u_pred = result["u_pred"]
    t = result["t"]
    x = result["x"]

    plt.figure(figsize=(10, 5))
    im = plt.imshow(u_pred.T, aspect="auto", cmap="RdBu", origin="lower",
                    extent=[t[0], t[-1], x[0], x[-1]])
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"Predicted u(t,x) from PINN ({test_filename})")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join("pinns", f"predictions_plot_{test_filename}.png"), dpi=150)
    plt.show()
