# src/set_membership.py

import numpy as np
from typing import Tuple, Dict, Optional
from tqdm import tqdm

# Imports for the MVEE optimization
from rsome import ro
from rsome import cpt_solver as cpt
import rsome as rso


def find_feasible_set(
    X_plus: np.ndarray,
    X_minus: np.ndarray,
    U_minus: np.ndarray,
    phi_matrix: np.ndarray,
    a_range: Tuple[float, float],
    b_range: Tuple[float, float],
    grid_density: int
) -> np.ndarray:
    """
    Performs a grid search to find the feasible set of parameters (A, B)
    that satisfy the eigenvalue condition.
    """
    # ... (Diese Funktion bleibt unverändert) ...
    print(f"Starting grid search over a {grid_density}x{grid_density} grid...")
    a_vec = np.linspace(a_range[0], a_range[1], grid_density)
    b_vec = np.linspace(b_range[0], b_range[1], grid_density)
    a_mesh, b_mesh = np.meshgrid(a_vec, b_vec)
    valid_pairs = []
    for i in tqdm(range(grid_density), desc="Finding Feasible Set"):
        for j in range(grid_density):
            a_val, b_val = a_mesh[i, j], b_mesh[i, j]
            W = X_plus - a_val * X_minus - b_val * U_minus
            check_matrix = np.block([[np.eye(X_minus.shape[0])], [W.T]]).T @ phi_matrix @ np.block([[np.eye(X_minus.shape[0])], [W.T]])
            try:
                eigs = np.linalg.eigvalsh(check_matrix)
                if np.all(eigs > 0):
                    valid_pairs.append((a_val, b_val))
            except np.linalg.LinAlgError:
                continue
    print(f"-> Found {len(valid_pairs)} feasible pairs.")
    return np.array(valid_pairs, dtype=np.float64)


def calculate_mvee(
    feasible_points: np.ndarray
) -> Optional[Dict[str, np.ndarray]]:
    """
    Calculates the Minimum Volume Enclosing Ellipsoid (MVEE) for a given
    set of points using the user's original, verified RSOME formulation.

    Args:
        feasible_points (np.ndarray): An array of points with shape (n_points, 2).

    Returns:
        A dictionary containing the ellipse's mathematical description ('P' matrix and 'c' vector),
        or None if no points are provided or if the solver fails.
    """
    if len(feasible_points) < 3: # A 2D ellipse needs at least 3 non-collinear points
        print("Warning: Cannot calculate MVEE for fewer than 3 points.")
        return None

    print(f"Calculating MVEE for {len(feasible_points)} points using original formulation...")
    
    m = feasible_points.shape[0]
    model = ro.Model()

    # --- Start of the RSOME implementation
    P = model.dvar((2, 2))
    c = model.dvar(2)
    Z = rso.tril(model.dvar((2, 2)))
    v = model.dvar(2)

    model.max(v.sum())
    model.st(v <= rso.log(rso.diag(Z)))
    model.st(rso.rstack([P, Z],
                       [Z.T, rso.diag(Z, fill=True)]) >> 0)
    for i in range(m):
        model.st(rso.norm(P @ feasible_points[i] - c) <= 1)
    model.st(P >> 0)
    

    try:
        model.solve(cpt, display=False) # Added display=False for cleaner output
        print(f'-> Solver finished. Determinant proxy: {np.exp(model.get())}')
    except Exception as e:
        print(f"Error during optimization with RSOME: {e}")
        return None

    # Return the results
    p_matrix_sol = P.get()
    c_vector_sol = c.get()
    
    return {'P': p_matrix_sol, 'c': c_vector_sol}