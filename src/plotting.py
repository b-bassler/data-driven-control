import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle 


def plot_confidence_ellipse_from_matrix(
    true_params: Tuple[float, float],
    estimated_params: Tuple[float, float],
    p_matrix: np.ndarray,
    confidence_delta: float,
    T: int,
    output_path: str
) -> None:
    """
    Visualizes the confidence ellipse by first calculating its geometric
    properties from the given P-matrix and then plotting it.

    Args:
        true_params (Tuple[float, float]): The true parameters (a_true, b_true).
        estimated_params (Tuple[float, float]): The estimated parameters (a_hat, b_hat) which are the ellipse center.
        p_matrix (np.ndarray): The 2x2 shape matrix of the ellipse.
        confidence_delta (float): The confidence level delta (e.g., 0.05 for 95%).
        T (int): The number of timesteps, used for the plot title.
        output_path (str): The full path where the plot image will be saved.
    """
    # Calculate Ellipse Geometry from P-Matrix
    eigenvalues, eigenvectors = np.linalg.eig(p_matrix)
    
    # Semi-axis lengths of the ellipse
    # Using abs() to prevent errors from tiny negative floating point values
    semi_axis_1 = 1 / np.sqrt(abs(eigenvalues[0]))
    semi_axis_2 = 1 / np.sqrt(abs(eigenvalues[1]))

    # Generate points on a unit circle
    phi = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.vstack([np.cos(phi), np.sin(phi)])

    # Transform points to the final ellipse (scaling, rotating, translating)
    a_hat, b_hat = estimated_params
    ellipse_transform = eigenvectors @ np.diag([semi_axis_1, semi_axis_2])
    ellipse_points = ellipse_transform @ circle_points
    ellipse_a = ellipse_points[0, :] + a_hat
    ellipse_b = ellipse_points[1, :] + b_hat

    # --- Part 2: Create the Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    confidence_percent = 100 * (1 - confidence_delta)
    ax.plot(ellipse_a, ellipse_b, label=f'{confidence_percent:.0f}% Confidence Ellipse', color='blue')
    ax.fill(ellipse_a, ellipse_b, alpha=0.2, color='blue')
    
    ax.scatter(*true_params, color='red', marker='x', s=120, zorder=5, label='True Parameters (a, b)')
    ax.scatter(*estimated_params, color='green', marker='+', s=120, zorder=5, label='Estimated Parameters (â, b̂)')

    ax.set_title(f'Confidence Ellipse (T = {T})')
    ax.set_xlabel('Parameter a')
    ax.set_ylabel('Parameter b')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    
    # Save the figure to the specified path instead of showing it
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  
    
    print(f"Ellipse plot saved to: {output_path}")




#-------------------------------------------------------------------------------





def plot_bootstrap_rectangle(
    true_params: Tuple[float, float],
    estimated_params: Tuple[float, float],
    epsilons: Tuple[float, float],
    confidence_delta: float,
    output_path: str
) -> None:
    """
    Plots the confidence region from a bootstrap analysis as a rectangle.

    Args:
        true_params (Tuple[float, float]): The true parameters (a_true, b_true).
        estimated_params (Tuple[float, float]): The estimated parameters (a_hat, b_hat).
        epsilons (Tuple[float, float]): The confidence bounds (epsilon_A, epsilon_B).
        confidence_delta (float): The confidence level delta (e.g., 0.05 for 95%).
        output_path (str): The full path where the plot image will be saved.
    """
    a_hat, b_hat = estimated_params
    epsilon_A, epsilon_B = epsilons

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the rectangle patch based on the calculated epsilons
    confidence_percent = 100 * (1 - confidence_delta)
    rect = Rectangle(
        xy=(a_hat - epsilon_A, b_hat - epsilon_B), # Bottom-left corner
        width=2 * epsilon_A,
        height=2 * epsilon_B,
        edgecolor='black',
        facecolor='blue',
        alpha=0.3,
        label=f'{confidence_percent:.0f}% Confidence Region'
    )
    ax.add_patch(rect)

    # Plot the estimated and true parameter points
    ax.plot(
        *estimated_params,
        marker='x', color='red', linestyle='None',
        markersize=10, label='Estimated (â, b̂)'
    )
    ax.plot(
        *true_params,
        marker='x', color='green', linestyle='None',
        markersize=10, label='True (a, b)'
    )

    # --- Configure plot aesthetics ---
    ax.set_xlabel("Parameter a")
    ax.set_ylabel("Parameter b")
    ax.set_title("Bootstrap Model Uncertainty")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    # Set plot limits to have some padding around the rectangle
    x_min = a_hat - 1.5 * epsilon_A
    x_max = a_hat + 1.5 * epsilon_A
    y_min = b_hat - 1.5 * epsilon_B
    y_max = b_hat + 1.5 * epsilon_B
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal', adjustable='box')

    # Save the figure to the specified path
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Close the figure to free up memory
    
    print(f"Bootstrap plot saved to: {output_path}")
