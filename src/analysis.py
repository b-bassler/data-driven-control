import numpy as np
from typing import Dict, Tuple

def calculate_p_matrix_for_confidence_ellipse(
    x_data: np.ndarray, 
    u_data: np.ndarray, 
    w_std_dev: float, 
    delta: float = 0.05
) -> np.ndarray:
    """
    Calculates the shape matrix P for the confidence ellipse.

    Args:
        x_data (np.ndarray): Array of state data used in the estimation.
        u_data (np.ndarray): Array of input data used in the estimation.
        w_std_dev (float): The standard deviation of the noise (sigma_w).
        delta (float): The confidence level delta (e.g., 0.05 for 95%).

    Returns:
        np.ndarray: The calculated 2x2 P-matrix for the ellipse.
    """
    n_dim = x_data.shape[1]
    p_dim = u_data.shape[1]

    # Constant C from the proposition 2.4 from Dean et. al 
    C_const = w_std_dev**2 * (np.sqrt(n_dim + p_dim) + np.sqrt(n_dim) + np.sqrt(2 * np.log(1/delta)))**2

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


import numpy as np
from typing import Tuple, Dict

class ConfidenceRectangle:
    """
    Represents a rectangular confidence region and calculates its metrics.
    """
    def __init__(self, center: Tuple[float, float], epsilons: Tuple[float, float]):
        """
        Initializes the rectangle.

        Args:
            center (Tuple[float, float]): The center of the rectangle (a_hat, b_hat).
            epsilons (Tuple[float, float]): The half-widths of the rectangle (epsilon_A, epsilon_B).
        """
        self.a_hat, self.b_hat = center
        self.epsilon_a, self.epsilon_b = epsilons

    def area(self) -> float:
        """Calculates the total area of the rectangle."""
        # Full width is 2 * epsilon_a, full height is 2 * epsilon_b
        return (2 * self.epsilon_a) * (2 * self.epsilon_b)

    def worst_case_deviation(self) -> float:
        """Calculates the worst-case deviation (distance from center to a corner)."""
        return np.sqrt(self.epsilon_a**2 + self.epsilon_b**2)

    def axis_parallel_deviations(self) -> Dict[str, float]:
        """Returns the maximum deviations along each axis."""
        return {"max_dev_a": self.epsilon_a, "max_dev_b": self.epsilon_b}


class ConfidenceEllipse:
    """
    Represents an elliptical confidence region and calculates its metrics.
    """
    def __init__(self, center: Tuple[float, float], p_matrix: np.ndarray):
        """
        Initializes the ellipse from its center and P-matrix.
        Performs the expensive eigendecomposition only once upon creation.

        Args:
            center (Tuple[float, float]): The center of the ellipse (a_hat, b_hat).
            p_matrix (np.ndarray): The 2x2 shape matrix of the ellipse.
        """
        self.center = center
        self.p_matrix = p_matrix
        
        # --- Perform expensive calculations once and store them ---
        self._eigenvalues, self._eigenvectors = np.linalg.eig(self.p_matrix)
        
        # Semi-axes (half-axes)
        # Using abs() for numerical stability against tiny negative values
        self._semi_axis_1 = 1 / np.sqrt(abs(self._eigenvalues[0]))
        self._semi_axis_2 = 1 / np.sqrt(abs(self._eigenvalues[1]))
        self.semi_axes = sorted([self._semi_axis_1, self._semi_axis_2], reverse=True)

    def area(self) -> float:
        """Calculates the total area of the ellipse."""
        # Area of ellipse is pi * r1 * r2
        return np.pi * self.semi_axes[0] * self.semi_axes[1]

    def worst_case_deviation(self) -> float:
        """Returns the worst-case deviation (the longest semi-axis)."""
        return self.semi_axes[0] # The major semi-axis

    def axis_parallel_deviations(self) -> Dict[str, float]:
        """
        Calculates the maximum deviations along each axis (half the size of the bounding box).
        """
        # Generate points on a unit circle
        phi = np.linspace(0, 2 * np.pi, 1000) 
        circle_points = np.vstack([np.cos(phi), np.sin(phi)])

        # Transform points to the final ellipse shape, centered at (0,0)
        ellipse_transform = self._eigenvectors @ np.diag(self.semi_axes)
        ellipse_points_centered = ellipse_transform @ circle_points

        # Find the maximum absolute coordinate along each axis
        max_a = np.max(np.abs(ellipse_points_centered[0, :]))
        max_b = np.max(np.abs(ellipse_points_centered[1, :]))

        return {"max_dev_a": max_a, "max_dev_b": max_b}
    



#-------------------------------------------------------------------------------------------------------



class MVEEllipse:
    """
    Represents the Minimum Volume Enclosing Ellipsoid (MVEE) and calculates its metrics.
    This class takes the direct output from the RSOME solver as input.
    """
    def __init__(self, mvee_results: Dict[str, np.ndarray]):
        """
        Initializes the ellipse from the RSOME solver's results.
        It immediately calculates and stores the geometric properties.

        Args:
            mvee_results (Dict[str, np.ndarray]): A dict containing the solver's
                                                  'P' matrix and 'c' vector.
        """
        if mvee_results is None or 'P' not in mvee_results or 'c' not in mvee_results:
            raise ValueError("Invalid mvee_results provided to MVEEllipse constructor.")

        P_s = mvee_results['P']
        c_s = mvee_results['c']

        # --- Translate solver output into geometric properties ---
        # This logic is taken directly from your original, verified script.
        
        # Center of the ellipse
        self.center = np.linalg.inv(P_s) @ c_s

        # Calculate shape matrix A = P.T @ P to find eigenvalues for the axes
        A_shape = P_s.T @ P_s
        eigenvalues, self._eigenvectors = np.linalg.eig(A_shape)

        # Semi-axes are the inverse of the square root of the eigenvalues
        # We sort them to have a consistent order (major axis first)
        sorted_indices = np.argsort(eigenvalues)[::-1] # Sort descending
        sorted_eigenvalues = eigenvalues[sorted_indices]
        self._eigenvectors = self._eigenvectors[:, sorted_indices]
        self.semi_axes = 1 / np.sqrt(sorted_eigenvalues)

    def area(self) -> float:
        """Calculates the total area of the ellipse."""
        return np.pi * self.semi_axes[0] * self.semi_axes[1]

    def worst_case_deviation(self) -> float:
        """Returns the worst-case deviation (the longest semi-axis)."""
        return self.semi_axes[0] # The major (longest) semi-axis

    def axis_parallel_deviations(self) -> Dict[str, float]:
        """
        Calculates the maximum deviations along each axis (half the size of the bounding box).
        """
        # Generate points on a unit circle
        phi = np.linspace(0, 2 * np.pi, 1000) 
        circle_points = np.vstack([np.cos(phi), np.sin(phi)])

        # Transform points to the final ellipse shape, centered at (0,0)
        ellipse_transform = self._eigenvectors @ np.diag(self.semi_axes)
        ellipse_points_centered = ellipse_transform @ circle_points

        # Find the maximum absolute coordinate along each axis
        max_a = np.max(np.abs(ellipse_points_centered[0, :]))
        max_b = np.max(np.abs(ellipse_points_centered[1, :]))

        return {"max_dev_a": max_a, "max_dev_b": max_b}