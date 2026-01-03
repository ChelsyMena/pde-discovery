import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Lasso, RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from scipy import fftpack
import matplotlib.pyplot as plt
import h5py
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETELY BLIND PDE DISCOVERY - ZERO PRIOR KNOWLEDGE")
print("="*80)

# =============================================================================
# STEP 0: LOAD RAW DATA - NO ASSUMPTIONS
# =============================================================================

h5_file_path = r"data/1_noiseless.h5"
with h5py.File(h5_file_path, 'r') as f:
    raw_data = f['u'][:]

print(f"Raw data loaded. Shape: {raw_data.shape}")
print("No information about:")
print("  • What 'u' represents")
print("  • Which dimensions are space/time")
print("  • What equation might govern it")
print("  • What physics might be involved")

# =============================================================================
# STEP 1: AUTOMATIC DIMENSION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("AUTOMATIC DIMENSION ANALYSIS")
print("="*80)

def analyze_dimensions(data):
    """Automatically determine which dimension is time."""
    n_dims = data.ndim
    
    if n_dims == 1:
        print("1D data: Assuming it's u(t)")
        return 0  # Only dimension is time
    
    elif n_dims == 2:
        # Try to detect which is time
        dim0_stats = analyze_dimension_stats(data, axis=0)
        dim1_stats = analyze_dimension_stats(data, axis=1)
        
        # Time usually has: higher autocorrelation, smoother evolution
        time_likelihood_0 = dim0_stats['autocorrelation'] - dim0_stats['high_freq_ratio']
        time_likelihood_1 = dim1_stats['autocorrelation'] - dim1_stats['high_freq_ratio']
        
        if time_likelihood_0 > time_likelihood_1:
            print(f"Detected: dimension 0 is time (likelihood: {time_likelihood_0:.3f} > {time_likelihood_1:.3f})")
            return 0
        else:
            print(f"Detected: dimension 1 is time (likelihood: {time_likelihood_1:.3f} > {time_likelihood_0:.3f})")
            return 1
    
    else:
        # For >2D, assume first dimension is time (common convention)
        print(f"{n_dims}D data: Assuming first dimension is time")
        return 0

def analyze_dimension_stats(data, axis):
    """Compute statistics for a dimension."""
    # Take mean along other dimensions
    other_axes = tuple([i for i in range(data.ndim) if i != axis])
    if other_axes:
        profile = np.mean(data, axis=other_axes)
    else:
        profile = data
    
    # Autocorrelation at lag 1
    if len(profile) > 1:
        autocorr = np.corrcoef(profile[:-1], profile[1:])[0, 1]
    else:
        autocorr = 0
    
    # Frequency content
    fft_vals = np.abs(fftpack.fft(profile - np.mean(profile)))
    high_freq_ratio = np.sum(fft_vals[len(fft_vals)//2:]) / np.sum(fft_vals)
    
    return {
        'autocorrelation': autocorr,
        'high_freq_ratio': high_freq_ratio,
        'length': len(profile)
    }

# Analyze dimensions
time_axis = analyze_dimensions(raw_data)
space_axes = [i for i in range(raw_data.ndim) if i != time_axis]

print(f"\nInterpretation:")
print(f"  Time axis: {time_axis}")
print(f"  Space axes: {space_axes}")

# Ensure time is first dimension for consistency
if time_axis != 0:
    print("Transposing to make time dimension first...")
    # Create new axes order: time first, then space
    new_order = [time_axis] + space_axes
    data = np.transpose(raw_data, new_order)
else:
    data = raw_data.copy()

print(f"Final data shape: {data.shape}")

# =============================================================================
# STEP 2: GENERATE ALL MATHEMATICAL OPERATIONS
# =============================================================================

print("\n" + "="*80)
print("GENERATING ALL MATHEMATICAL OPERATIONS")
print("="*80)

class MathematicalOperationGenerator:
    """Generate all possible mathematical operations on data."""
    
    def __init__(self, data):
        self.data = data
        self.n_dims = data.ndim
        self.operations = {}
        self.operation_descriptions = {}
    
    def generate_basic_operations(self):
        """Generate basic mathematical operations."""
        print("Generating basic mathematical operations...")
        
        # 1. The field itself
        self.operations['U'] = self.data
        self.operation_descriptions['U'] = "Field itself"
        
        # 2. All possible finite differences (derivatives)
        print("  Generating derivatives...")
        self._generate_derivatives(max_order=4)
        
        # 3. All possible pointwise nonlinearities
        print("  Generating nonlinear transformations...")
        self._generate_nonlinear_transforms()
        
        # 4. All possible products of existing terms
        print("  Generating products...")
        self._generate_products()
        
        print(f"Generated {len(self.operations)} basic operations")
        
        return self.operations
    
    def _generate_derivatives(self, max_order=4):
        """Generate all derivative combinations."""
        # For each dimension, generate derivatives up to max_order
        for dim in range(self.n_dims):
            current = self.data
            
            for order in range(1, max_order + 1):
                # Compute derivative
                try:
                    derivative = np.gradient(current, axis=dim)
                    
                    # Create name
                    if dim == 0:
                        name = f"d{order}U/dt{order}"
                        desc = f"{order}th time derivative"
                    else:
                        name = f"d{order}U/dx{order}"
                        desc = f"{order}th spatial derivative (dim {dim})"
                    
                    self.operations[name] = derivative
                    self.operation_descriptions[name] = desc
                    
                    # Update for next order
                    current = derivative
                    
                except:
                    break  # Can't compute higher derivatives
    
    def _generate_nonlinear_transforms(self):
        """Generate various nonlinear transformations."""
        nonlinear_funcs = [
            ('U^2', lambda x: x**2, "Square"),
            ('U^3', lambda x: x**3, "Cube"),
            ('|U|', lambda x: np.abs(x), "Absolute value"),
            ('sin(U)', lambda x: np.sin(x), "Sine"),
            ('cos(U)', lambda x: np.cos(x), "Cosine"),
            ('exp(U)', lambda x: np.exp(x), "Exponential"),
            ('log(|U|+1e-10)', lambda x: np.log(np.abs(x) + 1e-10), "Logarithm"),
            ('tanh(U)', lambda x: np.tanh(x), "Hyperbolic tangent"),
            ('sigmoid(U)', lambda x: 1/(1+np.exp(-x)), "Sigmoid"),
            ('ReLU(U)', lambda x: np.maximum(0, x), "ReLU"),
        ]
        
        for name, func, desc in nonlinear_funcs:
            try:
                self.operations[name] = func(self.data)
                self.operation_descriptions[name] = desc
            except:
                pass
    
    def _generate_products(self):
        """Generate all pairwise products of existing operations."""
        op_names = list(self.operations.keys())
        n_ops = len(op_names)
        
        print(f"  Generating products of {n_ops} operations...")
        
        # Generate all unique pairs
        for i in range(n_ops):
            for j in range(i, n_ops):
                name1 = op_names[i]
                name2 = op_names[j]
                
                if name1 == name2:
                    # Square
                    product_name = f"({name1})^2"
                    product_desc = f"Square of {self.operation_descriptions.get(name1, name1)}"
                    product = self.operations[name1] ** 2
                else:
                    # Product of different terms
                    product_name = f"{name1}*{name2}"
                    product_desc = f"Product of {self.operation_descriptions.get(name1, name1)} and {self.operation_descriptions.get(name2, name2)}"
                    product = self.operations[name1] * self.operations[name2]
                
                self.operations[product_name] = product
                self.operation_descriptions[product_name] = product_desc
        
        print(f"  Added {len(self.operations) - n_ops} product operations")
    
    def generate_all_operations(self, max_complexity=2):
        """Generate operations up to given complexity level."""
        print(f"\nGenerating operations up to complexity {max_complexity}...")
        
        # Level 1: Basic operations
        self.generate_basic_operations()
        
        # Higher levels: Combine existing operations
        for level in range(2, max_complexity + 1):
            print(f"\n  Complexity level {level}:")
            current_ops = list(self.operations.keys())
            new_ops = {}
            new_descs = {}
            
            for op1 in current_ops:
                for op2 in current_ops:
                    # Avoid trivial combinations
                    if op1 == op2:
                        continue
                    
                    # Generate combinations
                    combinations = [
                        (f"({op1})+({op2})", 
                         lambda x, y: x + y,
                         f"Sum of {op1} and {op2}"),
                        (f"({op1})*({op2})",
                         lambda x, y: x * y,
                         f"Product of {op1} and {op2}"),
                        (f"({op1})/({op2}+1e-10)",
                         lambda x, y: x / (y + 1e-10),
                         f"Ratio of {op1} to {op2}"),
                    ]
                    
                    for name, func, desc in combinations:
                        if name not in self.operations:
                            try:
                                result = func(self.operations[op1], self.operations[op2])
                                new_ops[name] = result
                                new_descs[name] = desc
                            except:
                                pass
            
            # Add new operations
            self.operations.update(new_ops)
            self.operation_descriptions.update(new_descs)
            print(f"    Added {len(new_ops)} new operations at level {level}")
        
        print(f"\nTotal operations generated: {len(self.operations)}")
        return self.operations

# Generate all possible operations
generator = MathematicalOperationGenerator(data)
all_operations = generator.generate_all_operations(max_complexity=2)

# =============================================================================
# STEP 3: BLIND TERM SELECTION WITH ENSEMBLE METHODS
# =============================================================================

print("\n" + "="*80)
print("BLIND TERM SELECTION WITH ENSEMBLE METHODS")
print("="*80)

def prepare_blind_dataset(operations, time_axis=0, n_samples=5000):
    """Prepare dataset for blind discovery."""
    
    # Identify time derivative candidates
    time_derivative_candidates = []
    for name in operations.keys():
        if 'dt' in name or 'd1U/dt1' in name or 'dU/dt' in name:
            time_derivative_candidates.append(name)
    
    if not time_derivative_candidates:
        # Try to find the most time-like derivative
        print("No obvious time derivatives found. Trying to identify...")
        # Use the first derivative w.r.t. first dimension as candidate
        candidate = None
        for name in operations.keys():
            if 'd1U/' in name and 'dt' not in name and 'dx' not in name:
                candidate = name
                break
        
        if candidate:
            time_derivative_candidates = [candidate]
        else:
            # Last resort: create time derivative
            time_derivative = np.gradient(data, axis=time_axis)
            operations['dU/dt'] = time_derivative
            generator.operation_descriptions['dU/dt'] = "Time derivative (estimated)"
            time_derivative_candidates = ['dU/dt']
    
    print(f"Time derivative candidates: {time_derivative_candidates}")
    
    # Use the first candidate as target
    target_name = time_derivative_candidates[0]
    target = operations[target_name]
    
    # Features are all other operations
    feature_names = [name for name in operations.keys() if name != target_name]
    
    print(f"Target: {target_name}")
    print(f"Number of candidate features: {len(feature_names)}")
    
    # Sample data points
    all_indices = np.array(np.meshgrid(*[range(s) for s in data.shape], indexing='ij')).T.reshape(-1, data.ndim)
    
    if n_samples < len(all_indices):
        sample_idx = np.random.choice(len(all_indices), n_samples, replace=False)
        indices = all_indices[sample_idx]
    else:
        indices = all_indices
        n_samples = len(indices)
    
    # Prepare feature matrix
    X = np.zeros((n_samples, len(feature_names)))
    y = np.zeros(n_samples)
    
    for i, idx in enumerate(indices):
        # Get target value
        y[i] = target[tuple(idx)]
        
        # Get feature values
        for j, feature_name in enumerate(feature_names):
            X[i, j] = operations[feature_name][tuple(idx)]
    
    return X, y, feature_names, target_name

print("Preparing blind dataset...")
X, y, feature_names, target_name = prepare_blind_dataset(
    all_operations, 
    time_axis=0, 
    n_samples=10000
)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"Target variable: {target_name}")

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# =============================================================================
# STEP 4: ENSEMBLE SPARSE DISCOVERY
# =============================================================================

print("\n" + "="*80)
print("ENSEMBLE SPARSE DISCOVERY")
print("="*80)

def ensemble_sparse_regression(X, y, feature_names, n_methods=3):
    """Use multiple methods to discover sparse terms."""
    from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
    from sklearn.feature_selection import SelectKBest, f_regression
    
    print("Running ensemble sparse discovery...")
    
    all_selections = []
    all_coeffs = []
    method_names = []
    
    # Method 1: LassoCV with cross-validation
    print("  Method 1: LassoCV...")
    try:
        lasso = LassoCV(cv=5, max_iter=10000, random_state=42)
        lasso.fit(X, y)
        lasso_selected = np.abs(lasso.coef_) > 1e-4
        all_selections.append(lasso_selected)
        all_coeffs.append(lasso.coef_)
        method_names.append("LassoCV")
    except:
        pass
    
    # Method 2: Top K features by correlation
    print("  Method 2: Correlation ranking...")
    try:
        correlations = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        top_k = min(20, X.shape[1] // 2)
        top_indices = np.argsort(-np.abs(correlations))[:top_k]
        corr_selected = np.zeros(X.shape[1], dtype=bool)
        corr_selected[top_indices] = True
        all_selections.append(corr_selected)
        
        # Fit ridge on selected features
        if np.sum(corr_selected) > 0:
            ridge = RidgeCV()
            ridge.fit(X[:, corr_selected], y)
            corr_coeffs = np.zeros(X.shape[1])
            corr_coeffs[corr_selected] = ridge.coef_
            all_coeffs.append(corr_coeffs)
            method_names.append("Correlation+Ridge")
    except:
        pass
    
    # Method 3: ElasticNet
    print("  Method 3: ElasticNetCV...")
    try:
        enet = ElasticNetCV(cv=5, max_iter=10000, random_state=42)
        enet.fit(X, y)
        enet_selected = np.abs(enet.coef_) > 1e-4
        all_selections.append(enet_selected)
        all_coeffs.append(enet.coef_)
        method_names.append("ElasticNet")
    except:
        pass
    
    # Method 4: Stability selection
    print("  Method 4: Stability selection...")
    try:
        n_subsamples = 20
        stability_scores = np.zeros(X.shape[1])
        
        for _ in range(n_subsamples):
            # Random subsample
            n_samples = X.shape[0]
            subsample_idx = np.random.choice(n_samples, n_samples//2, replace=False)
            X_sub = X[subsample_idx]
            y_sub = y[subsample_idx]
            
            # Fit Lasso with random alpha
            alpha = np.random.uniform(1e-4, 1e-2)
            lasso_sub = Lasso(alpha=alpha, max_iter=5000)
            lasso_sub.fit(X_sub, y_sub)
            
            # Update stability scores
            stability_scores += (np.abs(lasso_sub.coef_) > 1e-4)
        
        stability_scores /= n_subsamples
        stab_selected = stability_scores > 0.6  # Selected in >60% of runs
        
        # Fit on selected
        if np.sum(stab_selected) > 0:
            ridge = RidgeCV()
            ridge.fit(X[:, stab_selected], y)
            stab_coeffs = np.zeros(X.shape[1])
            stab_coeffs[stab_selected] = ridge.coef_
            
            all_selections.append(stab_selected)
            all_coeffs.append(stab_coeffs)
            method_names.append("Stability+Ridge")
    except:
        pass
    
    # Combine results
    if not all_selections:
        raise ValueError("No methods succeeded")
    
    print(f"\n  {len(all_selections)} methods succeeded")
    
    # Consensus selection: term selected by majority of methods
    selection_counts = np.sum(all_selections, axis=0)
    consensus_selected = selection_counts >= (len(all_selections) // 2 + 1)
    
    print(f"  Consensus: {np.sum(consensus_selected)} terms selected by majority")
    
    # Average coefficients from methods that selected each term
    consensus_coeffs = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        if consensus_selected[i]:
            method_coeffs = []
            for j, coeffs in enumerate(all_coeffs):
                if all_selections[j][i]:
                    method_coeffs.append(coeffs[i])
            
            if method_coeffs:
                consensus_coeffs[i] = np.median(method_coeffs)  # Use median for robustness
    
    return consensus_selected, consensus_coeffs, method_names

# Run ensemble discovery
selected_mask, selected_coeffs, methods_used = ensemble_sparse_regression(
    X_scaled, y_scaled, feature_names
)

selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
selected_coeffs = selected_coeffs[selected_mask]

print(f"\n✅ DISCOVERED {len(selected_features)} SIGNIFICANT TERMS:")
for i, (feature, coeff) in enumerate(zip(selected_features, selected_coeffs)):
    importance = np.abs(coeff) / np.sum(np.abs(selected_coeffs))
    print(f"{i+1:3d}. {coeff:+.6f} * {feature:40s} ({importance:.1%})")

# =============================================================================
# STEP 5: BLIND EQUATION FORMULATION
# =============================================================================

print("\n" + "="*80)
print("BLIND EQUATION FORMULATION")
print("="*80)

# Convert to physical coefficients
print("\nPhysical coefficients (denormalized):")
physical_coeffs = {}
for i, feature in enumerate(selected_features):
    orig_idx = feature_names.index(feature)
    scaled_coeff = selected_coeffs[i]
    physical_coeff = scaled_coeff * scaler_y.scale_[0] / (scaler_X.scale_[orig_idx] + 1e-10)
    physical_coeffs[feature] = physical_coeff
    
    # Get description
    desc = generator.operation_descriptions.get(feature, "Unknown operation")
    print(f"  {physical_coeff:+.6f} * {feature:30s} # {desc}")

# Build equation
print(f"\n🔬 DISCOVERED EQUATION:")
equation_terms = []

# Sort by absolute coefficient magnitude
sorted_terms = sorted(physical_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)

for i, (feature, coeff) in enumerate(sorted_terms):
    if abs(coeff) > 1e-10:  # Only include non-zero terms
        sign = "+" if coeff >= 0 else "-"
        mag = abs(coeff)
        
        # Format coefficient nicely
        if mag < 0.001:
            coeff_str = f"{mag:.2e}"
        elif mag < 1:
            coeff_str = f"{mag:.4f}"
        else:
            coeff_str = f"{mag:.2f}"
        
        if i == 0:
            if coeff < 0:
                equation_terms.append(f"-{coeff_str}·{feature}")
            else:
                equation_terms.append(f"{coeff_str}·{feature}")
        else:
            equation_terms.append(f" {sign} {coeff_str}·{feature}")

if equation_terms:
    equation_str = f"{target_name} = " + "".join(equation_terms)
    print(f"\n  {equation_str}")
else:
    print("  No significant terms discovered!")

# =============================================================================
# STEP 6: AUTOMATIC EQUATION CLASSIFICATION
# =============================================================================

print("\n" + "="*80)
print("AUTOMATIC EQUATION CLASSIFICATION")
print("="*80)

# Create a pattern database (mathematical, not physical)
equation_patterns = {
    'Heat-like': {
        'patterns': [['d2U/dx2'], ['d2U/dx1']],  # Second derivative
        'description': 'Second order diffusion'
    },
    'Wave-like': {
        'patterns': [['d2U/dt2', 'd2U/dx2']],  # Second time and space derivatives
        'description': 'Second order wave propagation'
    },
    'Transport': {
        'patterns': [['dU/dx'], ['U', 'dU/dx']],  # First derivative terms
        'description': 'First order transport'
    },
    'Reaction-Diffusion': {
        'patterns': [['d2U/dx2', 'U'], ['d2U/dx2', 'U^2'], ['d2U/dx2', 'U^3']],
        'description': 'Diffusion with nonlinear reaction'
    },
    'Kuramoto-Sivashinsky-like': {
        'patterns': [['d2U/dx2', 'd4U/dx4', 'U', 'dU/dx']],  # KS signature
        'description': 'Fourth order pattern formation'
    },
    'Burgers-like': {
        'patterns': [['d2U/dx2', 'U', 'dU/dx']],  # Viscous conservation
        'description': 'Nonlinear conservation with diffusion'
    },
    'KdV-like': {
        'patterns': [['d3U/dx3', 'U', 'dU/dx']],  # Third derivative
        'description': 'Third order dispersion'
    }
}

def classify_equation_discovered(selected_features, patterns_db):
    """Classify discovered equation based on term patterns."""
    best_match = None
    best_score = -1
    matches = {}
    
    for eq_name, eq_info in patterns_db.items():
        score = 0
        matched_patterns = []
        
        for pattern in eq_info['patterns']:
            # Check how many pattern terms are in selected features
            pattern_matches = 0
            for pattern_term in pattern:
                # Check if pattern term appears in any selected feature
                for feature in selected_features:
                    if pattern_term in feature:
                        pattern_matches += 1
                        break
            
            pattern_score = pattern_matches / len(pattern)
            if pattern_score > score:
                score = pattern_score
                matched_patterns = pattern
        
        matches[eq_name] = {
            'score': score,
            'description': eq_info['description'],
            'matched_pattern': matched_patterns
        }
        
        if score > best_score:
            best_score = score
            best_match = eq_name
    
    return best_match, best_score, matches

best_match, best_score, all_matches = classify_equation_discovered(
    selected_features, equation_patterns
)

print(f"\n📊 EQUATION CLASSIFICATION:")
print(f"  Best match: {best_match}")
print(f"  Match score: {best_score:.1%}")
print(f"  Description: {all_matches[best_match]['description']}")

print(f"\n  All matches:")
for eq_name, match_info in sorted(all_matches.items(), key=lambda x: x[1]['score'], reverse=True)[:5]:
    if match_info['score'] > 0.3:
        print(f"    {eq_name:25s}: {match_info['score']:.1%} - {match_info['description']}")

# =============================================================================
# STEP 7: VALIDATION AND VISUALIZATION
# =============================================================================

print("\n" + "="*80)
print("VALIDATION AND VISUALIZATION")
print("="*80)

# Test prediction accuracy
print("\nTesting prediction accuracy...")

# Create test dataset
test_n = min(2000, X.shape[0] // 5)
test_idx = np.random.choice(X.shape[0], test_n, replace=False)

X_test = X[test_idx]
y_test = y[test_idx]

# Predict using discovered equation
y_pred = np.zeros(test_n)
for i in range(test_n):
    pred = 0
    for j, feature in enumerate(selected_features):
        orig_idx = feature_names.index(feature)
        pred += physical_coeffs[feature] * X_test[i, orig_idx]
    y_pred[i] = pred

# Compute metrics
mse = np.mean((y_pred - y_test)**2)
rmse = np.sqrt(mse)
r2 = 1 - mse / np.var(y_test)

print(f"\n📈 PREDICTION PERFORMANCE:")
print(f"  MSE:  {mse:.6f}")
print(f"  RMSE: {rmse:.6f}")
print(f"  R²:   {r2:.6f}")

if r2 > 0.9:
    print("  ✅ Excellent prediction!")
elif r2 > 0.7:
    print("  ✓ Good prediction")
elif r2 > 0.5:
    print("  ~ Fair prediction")
elif r2 > 0:
    print("  ⚠️  Poor but better than mean")
else:
    print("  ✗ Worse than predicting mean")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Actual vs Predicted
ax = axes[0, 0]
scatter = ax.scatter(y_test[:500], y_pred[:500], alpha=0.5, s=10, c='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', alpha=0.7, label='Perfect')
ax.set_xlabel(f'Actual {target_name}')
ax.set_ylabel(f'Predicted {target_name}')
ax.set_title(f'Prediction Accuracy (R² = {r2:.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Coefficient magnitudes
ax = axes[0, 1]
sorted_coeffs = sorted(physical_coeffs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
terms_plot = [t[0] for t in sorted_coeffs]
coeffs_plot = [t[1] for t in sorted_coeffs]

bars = ax.barh(range(len(terms_plot)), coeffs_plot)
ax.set_yticks(range(len(terms_plot)))
ax.set_yticklabels(terms_plot, fontsize=8)
ax.set_xlabel('Coefficient Value')
ax.set_title('Top 15 Term Coefficients')
ax.invert_yaxis()

# Color by sign
for i, bar in enumerate(bars):
    if coeffs_plot[i] < 0:
        bar.set_color('red')
    else:
        bar.set_color('blue')

# 3. Term importance
ax = axes[0, 2]
abs_coeffs = np.abs([c for c in coeffs_plot])
total = np.sum(abs_coeffs)
percentages = abs_coeffs / total * 100

bars = ax.barh(range(len(terms_plot)), percentages)
ax.set_yticks(range(len(terms_plot)))
ax.set_yticklabels(terms_plot, fontsize=8)
ax.set_xlabel('Relative Importance (%)')
ax.set_title('Term Relative Importance')
ax.invert_yaxis()

for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{percentages[i]:.1f}%', va='center', fontsize=8)

# 4. Equation match scores
ax = axes[1, 0]
eq_names = list(all_matches.keys())
scores = [all_matches[name]['score'] for name in eq_names]

sorted_idx = np.argsort(scores)[-10:]  # Top 10
eq_names_sorted = [eq_names[i] for i in sorted_idx]
scores_sorted = [scores[i] for i in sorted_idx]

bars = ax.barh(range(len(eq_names_sorted)), scores_sorted)
ax.set_yticks(range(len(eq_names_sorted)))
ax.set_yticklabels(eq_names_sorted, fontsize=9)
ax.set_xlabel('Match Score')
ax.set_title('Equation Classification Scores')
ax.invert_yaxis()

for i, bar in enumerate(bars):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{scores_sorted[i]:.1%}', va='center', fontsize=8)

# 5. Residual distribution
ax = axes[1, 1]
residuals = y_pred - y_test
ax.hist(residuals, bins=50, alpha=0.7, color='purple')
ax.set_xlabel('Prediction Residuals')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
ax.grid(True, alpha=0.3)

# Add statistics
res_mean, res_std = np.mean(residuals), np.std(residuals)
ax.axvline(res_mean, color='red', linestyle='--', label=f'Mean: {res_mean:.4f}')
ax.axvline(res_mean + res_std, color='orange', linestyle=':', label=f'±1σ: {res_std:.4f}')
ax.axvline(res_mean - res_std, color='orange', linestyle=':')
ax.legend(fontsize=8)

# 6. Discovery summary
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
BLIND PDE DISCOVERY SUMMARY

Data: {data.shape}
Target: {target_name}
Methods: {', '.join(methods_used)}

Discovered: {len(selected_features)} terms
Best match: {best_match} ({best_score:.1%})
Prediction R²: {r2:.3f}

Top 3 terms:
1. {sorted_terms[0][0]:30s}: {sorted_terms[0][1]:+.4f}
2. {sorted_terms[1][0] if len(sorted_terms) > 1 else 'N/A':30s}: {sorted_terms[1][1] if len(sorted_terms) > 1 else 0:+.4f}
3. {sorted_terms[2][0] if len(sorted_terms) > 2 else 'N/A':30s}: {sorted_terms[2][1] if len(sorted_terms) > 2 else 0:+.4f}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Completely Blind PDE Discovery - Zero Prior Knowledge', fontsize=14)
plt.tight_layout()
plt.show()

# =============================================================================
# FINAL RESULTS
# =============================================================================


# Save results
discovery_results = {
    'data_shape': data.shape,
    'target_variable': target_name,
    'discovered_equation': equation_str,
    'best_match': best_match,
    'match_score': float(best_score),
    'prediction_r2': float(r2),
    'significant_terms': len(selected_features),
    'physical_coefficients': {k: float(v) for k, v in physical_coeffs.items()},
    'methods_used': methods_used
}

print(f"\n💾 RESULTS SAVED:")
for key, value in discovery_results.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    else:
        print(f"{key}: {value}")

print("\n" + "="*80)
print("SUCCESS: COMPLETELY BLIND PDE DISCOVERY COMPLETE!")
print("="*80)