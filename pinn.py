import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import h5py
from jax import grad, jit, vmap
import optax
import numpy as np

# Domain parameters
DOMAIN_SIZE = 100.0
N_DOF = 200
DT = 0.1

class ImprovedKSPINN:
    def __init__(self, layer_sizes=[2, 128, 128, 128, 1], domain_size=DOMAIN_SIZE):
        self.layer_sizes = layer_sizes
        self.domain_size = domain_size
        self.params = self.init_params()
        
        # Equation parameters with better initialization
        self.eq_params = {
            'c1': 0.8,  # u_t coefficient - start close to expected
            'c2': 0.8,  # u*u_x coefficient  
            'c3': 1.0,  # u_xx coefficient
            'c4': 1.0,  # u_xxxx coefficient
            'c5': 0.0   # u_x coefficient (should remain small)
        }
        
        # Track best equation
        self.best_eq_params = self.eq_params.copy()
        self.best_physics_loss = float('inf')
        self.best_epoch = 0

    def init_params(self):
        """Initialize network parameters"""
        keys = jax.random.split(jax.random.PRNGKey(0), len(self.layer_sizes)-1)
        params = []
        for i in range(len(self.layer_sizes)-1):
            W = jax.random.normal(keys[i], (self.layer_sizes[i], self.layer_sizes[i+1])) * jnp.sqrt(2.0/self.layer_sizes[i])
            b = jnp.zeros((self.layer_sizes[i+1],))
            params.append({'W': W, 'b': b})
        return params

    def forward(self, params, X):
        """Neural network forward pass"""
        x = X
        for i, layer in enumerate(params[:-1]):
            x = jnp.dot(x, layer['W']) + layer['b']
            x = jnp.tanh(x)
        x = jnp.dot(x, params[-1]['W']) + params[-1]['b']
        return x

    def u_pred(self, params, x, t):
        """Predicted solution u(x,t)"""
        X = jnp.stack([x, t], axis=-1)
        return self.forward(params, X).squeeze()

    def derivatives(self, params, x, t):
        """Compute derivatives using automatic differentiation"""
        def u_scalar(x, t):
            X = jnp.stack([x, t])[None, :]
            return self.forward(params, X).squeeze()

        # First derivatives
        u_t = grad(u_scalar, argnums=1)(x, t)
        u_x = grad(u_scalar, argnums=0)(x, t)

        # Higher spatial derivatives
        u_xx = grad(grad(u_scalar, argnums=0), argnums=0)(x, t)
        u_xxx = grad(grad(grad(u_scalar, argnums=0), argnums=0), argnums=0)(x, t)
        u_xxxx = grad(grad(grad(grad(u_scalar, argnums=0), argnums=0), argnums=0), argnums=0)(x, t)

        u_val = u_scalar(x, t)
        
        return u_val, u_t, u_x, u_xx, u_xxx, u_xxxx

    def pde_residual(self, params, eq_params, x, t):
        """PDE residual with discovered coefficients"""
        u_val, u_t, u_x, u_xx, u_xxx, u_xxxx = self.derivatives(params, x, t)
        
        # General form: c1*u_t + c2*u*u_x + c3*u_xx + c4*u_xxxx + c5*u_x = 0
        residual = (eq_params['c1'] * u_t + 
                   eq_params['c2'] * u_val * u_x + 
                   eq_params['c3'] * u_xx + 
                   eq_params['c4'] * u_xxxx + 
                   eq_params['c5'] * u_x)
        
        return residual

    def loss_function(self, params, eq_params, data, physics_weight=1.0):
        """Improved loss function with coefficient constraints"""
        x_data, t_data, u_data = data

        # Data loss
        u_pred_vals = vmap(self.u_pred, (None, 0, 0))(params, x_data, t_data)
        data_loss = jnp.mean((u_pred_vals - u_data)**2)

        # Physics loss (PDE residual) with discovered coefficients
        physics_loss = jnp.mean(vmap(self.pde_residual, (None, None, 0, 0))(params, eq_params, x_data, t_data)**2)

        # IMPROVED: Sparsity regularization - only penalize small terms
        sparsity_loss = 0.001 * jnp.abs(eq_params['c5'])  # Only penalize u_x term
        
        # NEW: Coefficient constraints to prevent decay
        coefficient_constraint = (
            0.01 * (1.0 - jnp.abs(eq_params['c1']))**2 +  # Encourage u_t ~ 1.0
            0.01 * (1.0 - jnp.abs(eq_params['c2']))**2 +  # Encourage u*u_x ~ 1.0
            0.005 * jnp.exp(-5 * jnp.abs(eq_params['c1'])) +  # Penalize near-zero u_t
            0.005 * jnp.exp(-5 * jnp.abs(eq_params['c2']))    # Penalize near-zero u*u_x
        )

        total_loss = (data_loss + 
                     physics_weight * physics_loss + 
                     sparsity_loss + 
                     coefficient_constraint)

        losses = {
            'total': total_loss,
            'data': data_loss,
            'physics': physics_loss,
            'sparsity': sparsity_loss,
            'coefficient_constraint': coefficient_constraint
        }

        return total_loss, losses

    def get_equation_string(self, eq_params):
        """Convert discovered parameters to equation string"""
        terms = []
        if abs(eq_params['c1']) > 0.01:
            terms.append(f"{eq_params['c1']:.3f}*u_t")
        if abs(eq_params['c2']) > 0.01:
            terms.append(f"{eq_params['c2']:.3f}*u*u_x")
        if abs(eq_params['c3']) > 0.01:
            terms.append(f"{eq_params['c3']:.3f}*u_xx")
        if abs(eq_params['c4']) > 0.01:
            terms.append(f"{eq_params['c4']:.3f}*u_xxxx")
        if abs(eq_params['c5']) > 0.01:
            terms.append(f"{eq_params['c5']:.3f}*u_x")
        
        if terms:
            equation = " + ".join(terms) + " = 0"
            return equation
        else:
            return "0 = 0"
    
    def is_good_equation(self, eq_params, physics_loss):
        """Check if we have discovered a good equation"""
        coefficients = [eq_params['c1'], eq_params['c2'], eq_params['c3'], eq_params['c4']]
        
        # Check if main terms are significant
        has_main_terms = (jnp.abs(eq_params['c1']) > 0.3 and  # u_t
                         jnp.abs(eq_params['c2']) > 0.3 and   # u*u_x  
                         jnp.abs(eq_params['c3']) > 0.3 and   # u_xx
                         jnp.abs(eq_params['c4']) > 0.3)      # u_xxxx
        
        # Check if physics loss is low
        physics_ok = physics_loss < 1e-2
        
        return has_main_terms and physics_ok


def load_perturbed_data(h5_file_path, max_samples=20000):
    """Load perturbed data from HDF5 file"""
    with h5py.File(h5_file_path, 'r') as f:
        t_data = f['t'][:]  # shape: (time_steps,)
        u_data = f['u'][:]  # shape: (time_steps, space_points)
        x_data = f['x'][:]  # shape: (space_points,)
    
    print(f"Loaded data shapes:")
    print(f"  t: {t_data.shape}")
    print(f"  u: {u_data.shape}")
    print(f"  x: {x_data.shape}")
    
    # Transpose u_data to (space, time)
    u_data = u_data.T
    
    # Create full grid
    X, T = jnp.meshgrid(x_data, t_data, indexing='ij')
    x_flat = X.reshape(-1)
    t_flat = T.reshape(-1)
    u_flat = u_data.reshape(-1)
    
    print(f"Flattened data: {x_flat.shape} points")
    
    # Subsample if too many points
    if len(x_flat) > max_samples:
        rng = jax.random.PRNGKey(42)
        indices = jax.random.choice(rng, len(x_flat), (max_samples,), replace=False)
        x_flat = x_flat[indices]
        t_flat = t_flat[indices]
        u_flat = u_flat[indices]
        print(f"Subsampled to: {len(x_flat)} points")
    
    return x_flat, t_flat, u_flat


def train_pinn_with_equation_discovery(h5_file_path):
    """Improved training with better strategy"""
    
    # Load perturbed data
    print("Loading perturbed data...")
    x_data, t_data, u_data = load_perturbed_data(h5_file_path)
    
    print(f"\nTraining data statistics:")
    print(f"x range: [{x_data.min():.2f}, {x_data.max():.2f}]")
    print(f"t range: [{t_data.min():.2f}, {t_data.max():.2f}]")
    print(f"u range: [{u_data.min():.2f}, {u_data.max():.2f}]")
    
    # Initialize PINN
    pinn = ImprovedKSPINN(layer_sizes=[2, 64, 64, 1])
    
    # Separate optimizers with different learning rates
    nn_optimizer = optax.adam(1e-3)
    eq_optimizer = optax.adam(3e-3)  # Lower LR for equation params
    
    nn_opt_state = nn_optimizer.init(pinn.params)
    eq_opt_state = eq_optimizer.init(pinn.eq_params)
    
    data = (x_data, t_data, u_data)
    
    @jit
    def step(nn_params, eq_params, nn_opt_state, eq_opt_state, data, physics_weight):
        def loss_fn(nn_p, eq_p):
            return pinn.loss_function(nn_p, eq_p, data, physics_weight)[0]
        
        # Compute gradients for both parameter sets
        loss_val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(nn_params, eq_params)
        nn_grads, eq_grads = grads
        
        # Update neural network
        nn_updates, nn_opt_state = nn_optimizer.update(nn_grads, nn_opt_state)
        nn_params = optax.apply_updates(nn_params, nn_updates)
        
        # Update equation parameters
        eq_updates, eq_opt_state = eq_optimizer.update(eq_grads, eq_opt_state)
        eq_params = optax.apply_updates(eq_params, eq_updates)
        
        return nn_params, eq_params, nn_opt_state, eq_opt_state, loss_val
    
    # Improved training loop with progressive physics weighting
    print("\nStarting improved equation discovery...")
    losses_history = []
    patience = 10
    patience_counter = 0
    
    for epoch in range(2000):  # Reduced epochs
        # Progressive physics weighting
        if epoch < 500:
            physics_weight = 1.0
        elif epoch < 1000:
            physics_weight = 2.0
        else:
            physics_weight = 3.0
        
        pinn.params, pinn.eq_params, nn_opt_state, eq_opt_state, loss_val = step(
            pinn.params, pinn.eq_params, nn_opt_state, eq_opt_state, data, physics_weight
        )
        
        if epoch % 200 == 0 or epoch < 100:
            total_loss, loss_breakdown = pinn.loss_function(pinn.params, pinn.eq_params, data, physics_weight)
            losses_history.append(loss_breakdown)
            
            equation = pinn.get_equation_string(pinn.eq_params)
            print(f"Epoch {epoch}:")
            print(f"  Total Loss: {total_loss:.6f}")
            print(f"  Data Loss: {loss_breakdown['data']:.6f}")
            print(f"  Physics Loss: {loss_breakdown['physics']:.6f}")
            print(f"  Physics Weight: {physics_weight}")
            print(f"  Discovered Equation: {equation}")
            
            # Track best equation
            if pinn.is_good_equation(pinn.eq_params, loss_breakdown['physics']):
                if loss_breakdown['physics'] < pinn.best_physics_loss:
                    pinn.best_eq_params = {k: v.copy() for k, v in pinn.eq_params.items()}
                    pinn.best_physics_loss = loss_breakdown['physics']
                    pinn.best_epoch = epoch
                    patience_counter = 0
                    print(f"  ↗ New best equation at epoch {epoch}")
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience and epoch > 500:
                print(f"Early stopping at epoch {epoch}")
                break
                
            print("---")
    
    # Restore best equation
    pinn.eq_params = pinn.best_eq_params
    print(f"\nRestored best equation from epoch {pinn.best_epoch}")
    
    return pinn, losses_history


def plot_comprehensive_results(pinn, h5_file_path):
    """Create comprehensive visualization of results"""
    
    # Load original data
    with h5py.File(h5_file_path, 'r') as f:
        t_original = f['t'][:]
        u_original = f['u'][:]
        x_original = f['x'][:]
    
    # Create prediction grid
    X, T = jnp.meshgrid(x_original, t_original, indexing='ij')
    x_plot = X.reshape(-1)
    t_plot = T.reshape(-1)
    
    # Generate PINN predictions
    print("Generating PINN predictions...")
    u_pred = vmap(pinn.u_pred, (None, 0, 0))(pinn.params, x_plot, t_plot)
    u_pred_grid = u_pred.reshape(len(x_original), len(t_original))
    
    # Create comprehensive plots
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original data
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(u_original.T, cmap="RdBu", aspect="auto", origin="lower",
                     extent=[t_original[0], t_original[-1], x_original[0], x_original[-1]])
    plt.colorbar(im1, ax=ax1, label="u")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Space")
    ax1.set_title("Original Perturbed Data")
    
    # Plot 2: PINN reconstruction
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(u_pred_grid, cmap="RdBu", aspect="auto", origin="lower",
                     extent=[t_original[0], t_original[-1], x_original[0], x_original[-1]])
    plt.colorbar(im2, ax=ax2, label="u")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Space")
    ax2.set_title("PINN Reconstruction")
    
    # Plot 3: Difference
    ax3 = plt.subplot(2, 3, 3)
    difference = u_original.T - u_pred_grid
    im3 = ax3.imshow(difference, cmap="RdBu", aspect="auto", origin="lower",
                     extent=[t_original[0], t_original[-1], x_original[0], x_original[-1]])
    plt.colorbar(im3, ax=ax3, label="Difference")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Space")
    ax3.set_title("Difference (Original - PINN)")
    
    # Plot 4: Training loss
    ax4 = plt.subplot(2, 3, 4)
    epochs = range(0, len(pinn.losses_history)*200, 200)
    total_losses = [l['total'] for l in pinn.losses_history]
    data_losses = [l['data'] for l in pinn.losses_history]
    physics_losses = [l['physics'] for l in pinn.losses_history]
    
    ax4.plot(epochs, total_losses, 'b-', linewidth=2, label='Total Loss')
    ax4.plot(epochs, data_losses, 'r--', linewidth=2, label='Data Loss')
    ax4.plot(epochs, physics_losses, 'g-.', linewidth=2, label='Physics Loss')
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss History')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Profile comparison at specific times
    ax5 = plt.subplot(2, 3, 5)
    time_indices = [0, len(t_original)//4, len(t_original)//2, 3*len(t_original)//4, -1]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, t_idx in enumerate(time_indices):
        if t_idx < len(t_original):
            ax5.plot(x_original, u_original[t_idx], color=colors[i], linestyle='-', 
                    linewidth=2, label=f'Original t={t_original[t_idx]:.1f}')
            ax5.plot(x_original, u_pred_grid[:, t_idx], color=colors[i], linestyle='--', 
                    linewidth=1, alpha=0.8)
    
    ax5.set_xlabel('Space')
    ax5.set_ylabel('u')
    ax5.set_title('Profile Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Coefficient evolution (placeholder for now)
    ax6 = plt.subplot(2, 3, 6)
    equation_text = pinn.get_equation_string(pinn.eq_params)
    ax6.text(0.1, 0.5, f"Discovered Equation:\n{equation_text}", 
             fontsize=12, fontfamily='monospace', verticalalignment='center')
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Final Discovered Equation')
    
    plt.tight_layout()
    plt.savefig("improved_pinn_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional: Coefficient values
    print(f"\nFinal coefficient values:")
    for key, value in pinn.eq_params.items():
        print(f"  {key}: {value:.4f}")


def compare_with_known_equations(pinn):
    """Compare discovered equation with known PDEs"""
    discovered_eq = pinn.get_equation_string(pinn.eq_params)
    
    print("\n" + "="*70)
    print("EQUATION COMPARISON")
    print("="*70)
    print(f"Discovered:        {discovered_eq}")
    print(f"Standard KS:       1.000*u_t + 1.000*u*u_x + 1.000*u_xx + 1.000*u_xxxx = 0")
    print(f"Burgers:           1.000*u_t + 1.000*u*u_x - ν*u_xx = 0")
    print(f"Heat Equation:     1.000*u_t - α*u_xx = 0")
    print(f"KdV:               1.000*u_t + 1.000*u*u_x + u_xxx = 0")
    print("="*70)
    
    # Calculate similarity to standard KS
    ks_similarity = (
        abs(pinn.eq_params['c1'] - 1.0) + 
        abs(pinn.eq_params['c2'] - 1.0) + 
        abs(pinn.eq_params['c3'] - 1.0) + 
        abs(pinn.eq_params['c4'] - 1.0)
    ) / 4.0
    
    print(f"Similarity to KS: {1.0 - ks_similarity:.3f}")
    if ks_similarity < 0.3:
        print("✅ Close match to Kuramoto-Sivashinsky equation!")
    elif ks_similarity < 0.6:
        print("⚠️  Moderate match to Kuramoto-Sivashinsky equation")
    else:
        print("❌ Poor match to Kuramoto-Sivashinsky equation")


# Main execution
if __name__ == "__main__":
    # Path to your perturbed HDF5 file
    h5_file_path = "C:\\Users\\mahim\\Downloads\\wetransfer_data_concatenated-h5_2025-10-13_1515\\data_concatenated_10s.h5"
    
    try:
        print("🚀 IMPROVED PINN EQUATION DISCOVERY")
        print("="*50)
        
        # Train PINN and discover equation
        pinn, losses_history = train_pinn_with_equation_discovery(h5_file_path)
        pinn.losses_history = losses_history
        
        # Final results
        final_equation = pinn.get_equation_string(pinn.eq_params)
        print("\n" + "="*70)
        print("FINAL DISCOVERED EQUATION")
        print("="*70)
        print(final_equation)
        print("="*70)
        
        # Compare with known equations
        compare_with_known_equations(pinn)
        
        # Plot comprehensive results
        plot_comprehensive_results(pinn, h5_file_path)
        
        # Performance summary
        final_loss, final_loss_breakdown = pinn.loss_function(pinn.params, pinn.eq_params, (pinn.params, pinn.eq_params, pinn.params), 1.0)
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Final Physics Loss: {final_loss_breakdown['physics']:.6f}")
        print(f"   Final Data Loss: {final_loss_breakdown['data']:.6f}")
        print(f"   Best Epoch: {pinn.best_epoch}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the HDF5 file path is correct and the file contains 'x', 't', 'u' datasets.")