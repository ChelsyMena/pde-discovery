import numpy as _np
import matplotlib.pyplot as plt
import csv

FILENAME = "1_noiseless"

term_names = [
    'u', 'u_x', 'u_xx', 'u_xxx', 'u_xxxx',
    'u²', 'u³', '(u²)_x', '(u²)_xx', '(u³)_x', #'u·u_x'
]

log_path = f"pinns/pinn_{FILENAME}_training_ignore.csv"
try:
    with open(log_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if len(rows) == 0:
        print(f"No rows found in {log_path}, skipping plots")
    else:
        data = _np.array([[float(val) for val in row] for row in rows])
        iters = data[:, 0].astype(int)
        loss_hist = data[:, 1]
        data_hist = data[:, 2]
        phys_hist = data[:, 3]
        l1_hist = data[:, 4]
        coeffs_arr = data[:, 5:]

        # losses plot
        plt.figure(figsize=(9, 5))
        plt.plot(iters, loss_hist, label='total')
        plt.plot(iters, data_hist, label='data')
        plt.plot(iters, phys_hist, label='phys')
        plt.plot(iters, l1_hist, label='l1')
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.title(f'PINN Training Losses ({FILENAME})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pinns/pinn_{FILENAME}_losses_ignore.png', dpi=150)
        plt.close()

        # coefficients plot
        plt.figure(figsize=(10, 6))
        highlight = {'u_xx', 'u_xxxx', '(u²)_x'}
        n_terms = coeffs_arr.shape[1]
        for i in range(n_terms):
            name = term_names[i]
            series = coeffs_arr[:, i]
            if name in highlight:
                plt.plot(iters, series, linewidth=2.2, label=name)
            else:
                plt.plot(iters, series, color='0.6', linewidth=0.8, alpha=0.7)
        # add horizontal dotted guide lines at -0.5 and -1.0
        plt.axhline(-0.5, color='green', linestyle='--', linewidth=1, alpha=0.9)
        plt.axhline(-1.0, color='green', linestyle='--', linewidth=1, alpha=0.9)

        # annotate final coefficient values for highlighted terms
        try:
            final_vals = coeffs_arr[-1, :]
            x_pos = iters[-1]
            for name in highlight:
                if name in term_names:
                    idx = term_names.index(name)
                    val = final_vals[idx]
                    # place a small offset text to the right of the last datapoint
                    plt.annotate(f"{val:.3g}", xy=(x_pos, val), xytext=(6, 0),
                                 textcoords='offset points', va='center', ha='left', fontsize=9,
                                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        except Exception:
            # if something goes wrong (e.g., empty array), skip annotations
            pass

        plt.xlabel('iteration')
        plt.ylabel('Coefficient Value')
        plt.title(f'Coefficient trajectories ({FILENAME})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'pinns/pinn_{FILENAME}_coeffs_ignore.png', dpi=150)
        plt.close()
except FileNotFoundError:
    print(f"Log file {log_path} not found — skipping plots.")
