import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import Lasso
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp

import pysindy as ps

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12

# Algorithm to scan over threshold values during Ridge Regression, and select
# highest performing model on the test set


def rudy_algorithm2(
    x_train,
    x_test,
    t,
    pde_lib,
    dtol,
    alpha=1e-5,
    tol_iter=25,
    normalize_columns=True,
    optimizer_max_iter=20,
    optimization="STLSQ",
):

    # Do an initial least-squares fit to get an initial guess of the coefficients
    optimizer = ps.STLSQ(
        threshold=0,
        alpha=0,
        max_iter=optimizer_max_iter,
        normalize_columns=normalize_columns,
        ridge_kw={"tol": 1e-10},
    )

    # Compute initial model
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(x_train, t=t)

    # Set the L0 penalty based on the condition number of Theta
    l0_penalty = 1e-3 * np.linalg.cond(optimizer.Theta)
    coef_best = optimizer.coef_

    # Compute MSE on the testing x_dot data (takes x_test and computes x_dot_test)
    error_best = model.score(
        x_test, metric=mean_squared_error, squared=False
    ) + l0_penalty * np.count_nonzero(coef_best)

    coef_history_ = np.zeros((coef_best.shape[0],
                              coef_best.shape[1],
                              1 + tol_iter))
    error_history_ = np.zeros(1 + tol_iter)
    coef_history_[:, :, 0] = coef_best
    error_history_[0] = error_best
    tol = dtol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer.
    for i in range(tol_iter):
        if optimization == "STLSQ":
            optimizer = ps.STLSQ(
                threshold=tol,
                alpha=alpha,
                max_iter=optimizer_max_iter,
                normalize_columns=normalize_columns,
                ridge_kw={"tol": 1e-10},
            )
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(x_train, t=t)
        coef_new = optimizer.coef_
        coef_history_[:, :, i + 1] = coef_new
        error_new = model.score(
            x_test, metric=mean_squared_error, squared=False
        ) + l0_penalty * np.count_nonzero(coef_new)
        error_history_[i + 1] = error_new

        # If error improves, set the new best coefficients
        if error_new <= error_best:
            error_best = error_new
            coef_best = coef_new
            tol += dtol
        else:
            tol = max(0, tol - 2 * dtol)
            dtol = 2 * dtol / (tol_iter - i)
            tol += dtol
    return coef_best, error_best, coef_history_, error_history_

def apply_coefficient_threshold(model, tol=1e-1):
    """
    Zero out coefficients below tolerance threshold.
    Returns the thresholded coefficients.
    """
    coef = model.optimizer.coef_.copy()
    coef[np.abs(coef) < tol] = 0
    model.coef_ = coef
    return coef


# global tolerance for considering a coefficient zero
TOL = 1e-1
filename = "1_noiseless"

with h5py.File(f"data/{filename}.h5", "r") as f:
    u = f["u"][:]
    x = f["x"][:] #[i for i in range(f["u"].shape[1])] 
    t = f["t"][:] #[i for i in range(f["u"].shape[0])] #
    dt = f.attrs["dt"]
    #dx = 100/200

    u_dot = ps.FiniteDifference(axis=1)._differentiate(u, t=dt)

    u = u.reshape(len(x), len(t), 1)
    u_dot = u_dot.reshape(len(x), len(t), 1)

    train = range(0, int(len(t) * 0.6))
    test = [i for i in np.arange(len(t)) if i not in train]
    u_train = u[:, train, :]
    u_test = u[:, test, :]
    u_dot_train = u_dot[:, train, :]
    u_dot_test = u_dot[:, test, 0]
    t_train = t[train]
    t_test = t[test]

    # Define PDE library that is quadratic in u, and
    # fourth-order in spatial derivatives of u.
    # library_functions = [lambda x: x, lambda x: x * x]
    # library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = ps.PDELibrary(
    #     library_functions=library_functions,
    #     function_names=library_function_names,
        function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
        derivative_order=4,
        spatial_grid=x,
        include_bias=True,
        is_uniform=True,
        periodic=True
    )

    # Again, loop through all the optimizers
    print('\nSTLSQ model: ')
    optimizer = ps.STLSQ(threshold=10, alpha=1e-5, normalize_columns=True)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    #print('Thresholded coefficients:', TOL, coef_thresholded)
    model.optimizer.coef_ = coef_thresholded
    model.print()
    #u_dot_stlsq = model.predict(u_test)

    print('\nSR3 model, L0 norm: ')
    optimizer = ps.SR3(
        max_iter=10000,
        tol=1e-15,
        normalize_columns=True,
        regularizer="L1"
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    model.optimizer.coef_ = coef_thresholded
    model.print()

    print('\nSR3 model, L1 norm: ')
    optimizer = ps.SR3(
        max_iter=10000, 
        tol=1e-15, 
        normalize_columns=True,
        regularizer="L1"  # Enables L1 denoising
    )
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    model.optimizer.coef_ = coef_thresholded
    model.print()

    print('\nSSR model: ')
    optimizer = ps.SSR(normalize_columns=True, kappa=1e1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    model.optimizer.coef_ = coef_thresholded
    model.print()

    print('\nSSR (metric = model residual) model: ')
    optimizer = ps.SSR(criteria="model_residual", normalize_columns=True, kappa=1e1)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    model.optimizer.coef_ = coef_thresholded
    model.print()

    print('\nFROLs model: ')
    optimizer = ps.FROLS(normalize_columns=True, kappa=1e-4)
    model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
    model.fit(u_train, t=dt)
    coef_thresholded = apply_coefficient_threshold(model, TOL)
    model.optimizer.coef_ = coef_thresholded
    model.print()