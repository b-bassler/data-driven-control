from typing import Tuple, Optional, Dict
import numpy as np

from tqdm import tqdm
from .utils import calculate_norm_error 


def estimate_least_squares_iid(
    x_data: np.ndarray, 
    u_data: np.ndarray, 
    y_data: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Performs the least-squares estimation for i.i.d. data.
    The model is assumed to be y = A*x + B*u.

    Args:
        x_data (np.ndarray): Array of state data with shape (T, n).
        u_data (np.ndarray): Array of input data with shape (T, p).
        y_data (np.ndarray): Array of output data with shape (T, n).

    Returns:
        A tuple containing the estimated matrices (A_est, B_est).
        Returns (None, None) if the matrix is singular.
    """
    # The regressor matrix Z has the shape (T, n+p)
    # Each row is [x_i, u_i]
    Z = np.hstack([x_data, u_data])

    # Perform estimation (pinv is more robust against singularities than inv)
    # Formula: theta = (Z^T * Z)^-1 * Z^T * y (from Dean et. al)
    try:
        theta_est = np.linalg.pinv(Z.T @ Z) @ Z.T @ y_data
    except np.linalg.LinAlgError:
        print("Error: The matrix Z.T @ Z is singular and cannot be inverted.")
        return None, None

    # Get dimensions from the data
    n_dims = x_data.shape[1]
    
    # Extract the estimated parameter matrices A and B
    A_est = theta_est[:n_dims, :]
    B_est = theta_est[n_dims:, :]

    return A_est, B_est








def estimate_least_squares_timeseries(
    state_data: np.ndarray, 
    input_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs least-squares estimation for time series data.
    Handles multiple rollouts (N > 1).
    """
    N, T_plus_1 = state_data.shape
    T = T_plus_1 - 1

    # Reshape data into long vectors for estimation
    X_ges_plus = state_data[:, 1:]
    X_ges_minus = state_data[:, :T]
    U_ges_minus = input_data[:, :T]

    X_N = X_ges_plus.reshape(-1, 1)
    Z_matrix = np.hstack([
        X_ges_minus.reshape(-1, 1),
        U_ges_minus.reshape(-1, 1)
    ])

    # Perform estimation
    theta_est = np.linalg.pinv(Z_matrix.T @ Z_matrix) @ Z_matrix.T @ X_N
    A_est = theta_est[:1, :]
    B_est = theta_est[1:, :]
    
    return A_est, B_est








def _simulate_system(
    A_sys: np.ndarray, 
    B_sys: np.ndarray, 
    T: int, 
    N: int, 
    sigma_u: float, 
    sigma_w: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a linear system to create synthetic data. Uses a provided RNG.
    """
    u = rng.normal(0, sigma_u, (N, T))
    w = rng.normal(0, sigma_w, (N, T))
    
    x = np.zeros((N, T + 1))
    
    for k in range(T):
        x[:, k + 1] = A_sys * x[:, k] + B_sys * u[:, k] + w[:, k]
        
    return x, u








def perform_bootstrap_analysis(
    initial_estimate: Tuple[np.ndarray, np.ndarray],
    data_shape: Tuple[int, int],
    sigmas: Dict[str, float],
    M: int,
    delta: float,
    seed: int
) -> Dict[str, float]:
    """
    Performs the core bootstrap analysis loop.
    """
    A_hat, B_hat = initial_estimate
    N_real, T_real = data_shape
    sigma_u, sigma_w = sigmas['u'], sigmas['w']

    rng = np.random.default_rng(seed)
    error_A_list = []
    error_B_list = []

    for _ in range(M):
        # Generate synthetic data based on the initial estimate
        x_synthetic, u_synthetic = _simulate_system(A_hat, B_hat, T_real, N_real, sigma_u, sigma_w, rng)
        
        # Re-estimate with the synthetic data
        A_tilde, B_tilde = estimate_least_squares_timeseries(x_synthetic, u_synthetic)
        
        # Calculate and store the error
        error_A_list.append(calculate_norm_error(A_hat, A_tilde))
        error_B_list.append(calculate_norm_error(B_hat, B_tilde))

    # Calculate the confidence bounds from the collected errors
    epsilon_A = np.percentile(error_A_list, 100 * (1 - delta))
    epsilon_B = np.percentile(error_B_list, 100 * (1 - delta))
    
    return {"epsilon_A": epsilon_A, "epsilon_B": epsilon_B}



def perform_bootstrap_analysis_iid(
    initial_estimate: Tuple[np.ndarray, np.ndarray],
    N: int,
    sigmas: Dict[str, float],
    M: int,
    delta: float,
    seed: int
) -> Dict[str, float]:
    """
    Performs parametric bootstrap analysis specifically for the I.I.D. case.
    The model is y = a*x + b*u + w.
    """
    A_hat, B_hat = initial_estimate
    # For I.I.D. data, we need the std dev for x, u, and w
    sigma_x = sigmas.get('x', 1.0) # Default to 1.0 if not provided
    sigma_u = sigmas.get('u', 1.0)
    sigma_w = sigmas.get('w')

    rng = np.random.default_rng(seed)
    error_A_list = []
    error_B_list = []

    # Use leave=False for cleaner output when called in a loop
    for _ in tqdm(range(M), desc=f"I.I.D. Bootstrap (N={N})", leave=False):
        # 1. Generate new synthetic I.I.D. data based on the initial estimate
        x_synthetic = rng.normal(0, sigma_x, (N, 1))
        u_synthetic = rng.normal(0, sigma_u, (N, 1))
        w_synthetic = rng.normal(0, sigma_w, (N, 1))
        
        # y = a*x + b*u + w
        y_synthetic = A_hat * x_synthetic + B_hat * u_synthetic + w_synthetic
        
        # 2. Re-estimate with the synthetic I.I.D. data
        A_tilde, B_tilde = estimate_least_squares_iid(x_synthetic, u_synthetic, y_synthetic)
        
        if A_tilde is not None:
            # 3. Calculate and store the error
            error_A_list.append(calculate_norm_error(A_hat, A_tilde))
            error_B_list.append(calculate_norm_error(B_hat, B_tilde))

    # 4. Calculate the confidence bounds from the collected errors
    if not error_A_list or not error_B_list:
        # Handle cases where all estimations failed
        return {"epsilon_A": np.inf, "epsilon_B": np.inf}

    epsilon_A = np.percentile(error_A_list, 100 * (1 - delta))
    epsilon_B = np.percentile(error_B_list, 100 * (1 - delta))
    
    return {"epsilon_A": epsilon_A, "epsilon_B": epsilon_B}


