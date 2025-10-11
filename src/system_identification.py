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
    Performs least-squares estimation for i.i.d. data.
    The model is Y = Theta * Z, where Theta = [A B].

    Args:
        x_data (np.ndarray): Array of state data with shape (n, T).
        u_data (np.ndarray): Array of input data with shape (m, T).
        y_data (np.ndarray): Array of output data with shape (n, T).

    Returns:
        A tuple containing the estimated matrices (A_est, B_est).
        Returns (None, None) if the matrix is singular.
    """
    # 1. Build the regressor matrix Z as defined in   Eq. 2.8.
    # It has the shape (n+m, T), where columns are the samples.
    Z = np.vstack([x_data, u_data])

    # 2. Assign the output data to Y.
    Y = y_data

    # 3. Perform estimation using the formula: Theta = Y*Z^T * (Z*Z^T)^-1
    try:
        # Theta will be a "wide" matrix of shape (n, n+p), containing [A, B]
        theta_est_wide = Y @ Z.T @ np.linalg.pinv(Z @ Z.T)
    except np.linalg.LinAlgError:
        print("Error: The matrix Z @ Z.T is singular and cannot be inverted.")
        return None, None

    # 4. Get dimensions from the data shapes.
    # n_dims is the number of state variables (rows in x_data).
    n_dims = x_data.shape[0]
    
    # 5. Extract the estimated parameter matrices A and B from the wide Theta.
    A_est = theta_est_wide[:, :n_dims]
    B_est = theta_est_wide[:, n_dims:]

    return A_est, B_est



def estimate_least_squares_trajectory(
    state_data: np.ndarray, 
    input_data: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs least-squares estimation for time series data.
    """
    N, T_plus_1 = state_data.shape
    T = T_plus_1 - 1
    
    # For scalar systems (n=1, m=1), n_dims and n_dims are 1.
    n_dims = 1 

    # 1. Prepare the data matrices X+, X-, and U-
    X_plus = state_data[:, 1:]
    X_minus = state_data[:, :T]
    U_minus = input_data[:, :T]

    # 2. Reshape data to create one long data record (features, N*T samples)
    # This combines all rollouts into a single dataset.
    Y = X_plus.reshape(n_dims, -1)
    X_reshaped = X_minus.reshape(n_dims, -1)
    U_reshaped = U_minus.reshape(n_dims, -1)

    # 3. Build the regressor matrix Z as defined in Eq. 2.8.
    Z = np.vstack([X_reshaped, U_reshaped])

    # 4. Perform estimation using the same formula as in the i.i.d. case
    theta_est_wide = Y @ Z.T @ np.linalg.pinv(Z @ Z.T)
    
    # 5. Extract A and B
    A_est = theta_est_wide[:, :n_dims]
    B_est = theta_est_wide[:, n_dims:]
    
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






    

def perform_bootstrap_analysis_trajectory(
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
        A_tilde, B_tilde = estimate_least_squares_trajectory(x_synthetic, u_synthetic)
        
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
    Performs parametric bootstrap analysis for I.I.D. data.

    This method generates M synthetic datasets based on an initial parameter
    estimate (A_hat, B_hat) and the assumed noise distributions. For each
    synthetic dataset, it re-estimates the parameters to build a distribution
    of the estimation error, from which confidence bounds are derived.

    Args:
        initial_estimate: A tuple (A_hat, B_hat) of the initial LS estimates.
        N: The number of i.i.d. samples to generate per bootstrap iteration.
        sigmas: A dictionary containing the standard deviations for the data
                generation, e.g., {'x': 1.0, 'u': 1.0, 'w': 0.1}.
        M: The number of bootstrap iterations.
        delta: The desired confidence level (e.g., 0.05 for 95% confidence).
        seed: A seed for the random number generator for reproducibility.

    Returns:
        A dictionary with the calculated confidence bounds epsilon_A and epsilon_B.
    """
    A_hat, B_hat = initial_estimate
    sigma_x = sigmas.get('x', 1.0)
    sigma_u = sigmas.get('u', 1.0)
    sigma_w = sigmas.get('w')

    rng = np.random.default_rng(seed)
    error_A_list, error_B_list = [], []

    # tqdm provides a progress bar for the bootstrap loop
    for _ in tqdm(range(M), desc=f"I.I.D. Bootstrap (N={N})", leave=False):
        # 1. Generate new synthetic data as row vectors (1, N)
        x_synthetic = rng.normal(0, sigma_x, (1, N))
        u_synthetic = rng.normal(0, sigma_u, (1, N))
        w_synthetic = rng.normal(0, sigma_w, (1, N))
        
        # Calculate the synthetic output using the initial estimate
        y_synthetic = A_hat @ x_synthetic + B_hat @ u_synthetic + w_synthetic
        
        # 2. Re-estimate parameters with the synthetic data
        A_tilde, B_tilde = estimate_least_squares_iid(
            x_synthetic, u_synthetic, y_synthetic
        )
        
        if A_tilde is not None:
            # 3. Calculate and store the norm of the estimation error
            error_A_list.append(calculate_norm_error(A_hat, A_tilde))
            error_B_list.append(calculate_norm_error(B_hat, B_tilde))
    # 4. Calculate confidence bounds from the distribution of errors
    if not error_A_list or not error_B_list:
        # Handle cases where all estimations failed, returning infinite bounds
        return {"epsilon_A": np.inf, "epsilon_B": np.inf}

    epsilon_A = np.percentile(error_A_list, 100 * (1 - delta))
    epsilon_B = np.percentile(error_B_list, 100 * (1 - delta))
    
    return {"epsilon_A": epsilon_A, "epsilon_B": epsilon_B}