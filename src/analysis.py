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

    # Constant C from the proposition
    C_const = w_std_dev**2 * (np.sqrt(n_dim + p_dim) + np.sqrt(n_dim) + np.sqrt(2 * np.log(1/delta)))**2

    # Gram matrix Z^T * Z
    Z = np.hstack([x_data, u_data])
    gram_matrix = Z.T @ Z

    # Shape matrix P of the ellipse
    p_ellipse = gram_matrix / C_const
    
    return p_ellipse






#----------------------------------------------------------------------------------------------



# Auch in src/analysis.py

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