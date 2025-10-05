import numpy as np
from typing import Dict, Tuple
from scipy.linalg import sqrtm
import math


def calculate_tsiams_ellipse_matrix(
    state_data: np.ndarray,
    input_data: np.ndarray,
    true_A: np.ndarray,
    true_B: np.ndarray,
    sigmas: Dict[str, float],
    delta: float,
    c: float,
    tau: int
) -> Dict[str, np.ndarray]:
    """
    Calculates the characteristic matrix P for the Tsiams data-dependent ellipse.
    """
    T = input_data.shape[1]
    
    # Calculate V_t from the measurement data
    V_t = np.array([
        [np.vdot(state_data, state_data), np.vdot(state_data, input_data)],
        [np.vdot(input_data, state_data), np.vdot(input_data, input_data)]
    ])

    # Calculate the state covariance matrix T_t using TRUE system parameters
    T_t = np.zeros_like(true_A)
    M = (sigmas['u']**2) * (true_B @ true_B.T) + (sigmas['w']**2)
    t_summation = tau // 2 # In the script, t was defined as 2 for tau=2
    for k in range(t_summation):
        Ak = np.linalg.matrix_power(true_A, k)
        T_t += Ak @ M @ Ak.T
    
    n_dim = T_t.shape[0]
    T_t_dach = np.block([[T_t, np.zeros((n_dim, n_dim))],
                         [np.zeros((n_dim, n_dim)), (sigmas['u']**2) * np.eye(n_dim)]])

    V = c * tau * math.floor(T / tau) * T_t_dach
    V_dach = V_t + V

    # Calculate the radius of the uncertainty ball
    log_term = np.log(
        (np.sqrt(np.linalg.det(V_dach)) / np.sqrt(np.linalg.det(V))) * (5**n_dim / delta)
    )
    norm_term_sq = np.linalg.norm(sqrtm(V_dach) @ np.linalg.inv(sqrtm(V_t)), ord=2)**2
    radius = 8 * (sigmas['w']**2) * log_term * norm_term_sq

    # The characteristic matrix of the ellipse
    p_matrix = V_dach / radius
    
    return {'p_matrix': p_matrix}









def calculate_p_matrix_ddbounds_iid(
    x_data: np.ndarray, 
    u_data: np.ndarray, 
    w_std_dev: float, 
    delta: float = 0.05,
    tuning_factor: float = 1.0  
) -> np.ndarray:
    """
    Calculates the shape matrix P for the confidence ellipse of the data dependent
    bounds (Dean et al. Proposition 2.4). Includes an optional tuning factor
    to scale the conservatism of the bounds.

    Args:
        x_data (np.ndarray): Array of state data used in the estimation.
        u_data (np.ndarray): Array of input data used in the estimation.
        w_std_dev (float): The standard deviation of the noise (sigma_w).
        delta (float): The confidence level delta (e.g., 0.05 for 95%).
        tuning_factor (float, optional): A factor to scale the constant C. 
                                         Defaults to 1.0 (original formulation).
                                         Values < 1.0 lead to smaller, less
                                         conservative ellipses.

    Returns:
        np.ndarray: The calculated 2x2 P-matrix for the ellipse.
    """
    n_dim = x_data.shape[1]
    p_dim = u_data.shape[1]

    # Theoretical constant C from Proposition 2.4
    C_theory = w_std_dev**2 * (np.sqrt(n_dim + p_dim) + np.sqrt(n_dim) + np.sqrt(2 * np.log(1/delta)))**2

    # Apply the tuning factor to the theoretical constant
    C_const = C_theory * tuning_factor

    # Gram matrix Z^T * Z
    Z = np.hstack([x_data, u_data])
    gram_matrix = Z.T @ Z

    # Shape matrix P of the ellipse
    p_ellipse = gram_matrix / C_const
    
    return p_ellipse



#----------------------------------------------------------------------------------------------




def analyze_ellipse_geometry(p_matrix: np.ndarray) -> Dict[str, float]:
    """
    Analyzes an ellipse's P-matrix to find its geometric properties.

    Args:
        p_matrix (np.ndarray): The 2x2 shape matrix of the ellipse.

    Returns:
        A dictionary with the ellipse's geometric properties.
    """
    eigenvalues, eigenvectors = np.linalg.eig(p_matrix)

    # Semi-axes (half-axes)
    semi_axis_1 = 1 / np.sqrt(abs(eigenvalues[0]))
    semi_axis_2 = 1 / np.sqrt(abs(eigenvalues[1]))
    
    # Sort axes to have a consistent order (largest first)
    semi_axes_sorted = sorted([semi_axis_1, semi_axis_2], reverse=True)

    # Rotation angle of the main axis in degrees
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
    angle_deg = np.degrees(angle_rad)
    
    # Worst-case deviation (longest semi-axis)
    worst_case_deviation = semi_axes_sorted[0]

    # Bounding box calculation
    phi = np.linspace(0, 2 * np.pi, 1000) 
    circle_points = np.vstack([np.cos(phi), np.sin(phi)])
    ellipse_transform = eigenvectors @ np.diag([semi_axis_1, semi_axis_2])
    ellipse_points_centered = ellipse_transform @ circle_points
    max_a = np.max(np.abs(ellipse_points_centered[0, :]))
    max_b = np.max(np.abs(ellipse_points_centered[1, :]))

    return {
        "semi_axis_major": semi_axes_sorted[0],
        "semi_axis_minor": semi_axes_sorted[1],
        "angle_degrees": angle_deg,
        "worst_case_deviation": worst_case_deviation,
        "max_deviation_a": max_a,
        "max_deviation_b": max_b
    }





#-----------------------------------------------------------------------------------------


#classes for confidence regions


class ConfidenceRectangle:
    """
    Represents a rectangular confidence region and calculates its metrics.
    """
    def __init__(self, center: Tuple[float, float], epsilons: Tuple[float, float]):
        """Initializes the rectangle."""
        self.a_hat, self.b_hat = center
        self.epsilon_a, self.epsilon_b = epsilons
        self.center = np.array(center)

    def area(self) -> float:
        """Calculates the total area of the rectangle."""
        return (2 * self.epsilon_a) * (2 * self.epsilon_b)

    def worst_case_deviation(self) -> float:
        """Calculates the worst-case deviation (distance from center to a corner)."""
        return np.sqrt(self.epsilon_a**2 + self.epsilon_b**2)

    def axis_parallel_deviations(self) -> Dict[str, float]:
        """Returns the maximum deviations along each axis."""
        return {"max_dev_a": self.epsilon_a, "max_dev_b": self.epsilon_b}

    def contains(self, point: Tuple[float, float]) -> bool:
        """
        Checks if a given point is inside the rectangle.
        """
        point_a, point_b = point
        # The point is inside if the distance from the center along each axis
        # is less than or equal to the respective epsilon (half-width).
        return (np.abs(point_a - self.a_hat) <= self.epsilon_a) and \
               (np.abs(point_b - self.b_hat) <= self.epsilon_b)
    
    def contains_per_parameter(self, point: Tuple[float, float]) -> Dict[str, bool]:
        """
        Checks for containment for each parameter individually and for both jointly.

        Returns:
            A dictionary {'a': bool, 'b': bool, 'both': bool}.
        """
        point_a, point_b = point
        a_is_contained = np.abs(point_a - self.a_hat) <= self.epsilon_a
        b_is_contained = np.abs(point_b - self.b_hat) <= self.epsilon_b
        
        return {
            'a': a_is_contained,
            'b': b_is_contained,
            'both': a_is_contained and b_is_contained
        }



class ConfidenceEllipse:
    """
    Represents an elliptical confidence region derived from a shape matrix (P)
    and calculates its metrics. This corrected version uses direct analytical formulas
    for robustness and correctness.
    """
    def __init__(self, center: Tuple[float, float], p_matrix: np.ndarray):
        """
        Initializes the ellipse and pre-calculates the covariance matrix,
        which is fundamental for all metric calculations.
        """
        self.center = np.array(center)
        self.p_matrix = p_matrix

        # --- CRITICAL CORRECTION: Pre-calculate the covariance matrix ---
        # The covariance matrix is the inverse of the shape matrix P.
        # It is calculated once here and stored as an attribute for all methods to use.
        try:
            self.covariance_matrix = np.linalg.inv(self.p_matrix)
        except np.linalg.LinAlgError:
            print("Error: The shape matrix P is singular and cannot be inverted.")
            # Set to identity to prevent further crashes, though metrics will be wrong.
            self.covariance_matrix = np.eye(2)

    def area(self) -> float:
        """Calculates the total area of the ellipse."""
        # This formula is direct and robust: Area = pi * sqrt(det(P_inv))
        # It relies on the pre-calculated covariance_matrix.
        return np.pi * np.sqrt(np.linalg.det(self.covariance_matrix))

    def worst_case_deviation(self) -> float:
        """
        Returns the worst-case deviation (the longest semi-axis).
        This is the square root of the largest eigenvalue of the covariance matrix.
        """
        eigenvalues_cov, _ = np.linalg.eig(self.covariance_matrix)
        eigenvalues_cov[eigenvalues_cov < 0] = 0 # Ensure non-negativity
        return np.sqrt(np.max(eigenvalues_cov))

    def axis_parallel_deviations(self) -> Dict[str, float]:
        """
        Calculates the maximum deviations along each parameter axis.
        This is correctly and directly calculated from the diagonal of the
        pre-calculated covariance matrix.
        """
        # This now works because self.covariance_matrix was created in __init__
        max_dev_a = np.sqrt(self.covariance_matrix[0, 0])
        max_dev_b = np.sqrt(self.covariance_matrix[1, 1])
        
        return {"max_dev_a": max_dev_a, "max_dev_b": max_dev_b}

    def contains(self, point: Tuple[float, float]) -> bool:
        """
        Checks if a given point is inside the ellipse using the quadratic form.
        """
        point_vec = np.array(point)
        diff = (point_vec - self.center).reshape(2, 1)
        value = diff.T @ self.p_matrix @ diff
        return value.item() <= 1


