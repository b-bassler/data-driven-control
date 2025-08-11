import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.patches import Ellipse, Rectangle 
import pandas as pd
from typing import List, Dict

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








#---------------------------------------------------------------------------------------



def plot_metric_comparison(
    dataframe: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str
) -> None:
    """
    Plots a comparison of a given metric for the ellipse and rectangle methods.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(dataframe['T'], dataframe[f'ellipse_{metric_name}'], marker='o', linestyle='-', label='Data-Dependent (Ellipse)')
    ax.plot(dataframe['T'], dataframe[f'rect_{metric_name}'], marker='x', linestyle='--', label='Bootstrap (Rectangle)')

    ax.set_xlabel("Number of Data Points (T)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log') # Log scale often helps to see the trend better
    
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"-> Comparison plot saved to: {output_path}")    



def plot_metric_trend(
    dataframe: pd.DataFrame,
    metric_name: str,
    y_label: str,
    title: str,
    output_path: str
) -> None:
    """
    Plots the trend of a single metric over T from a results DataFrame.
    
    This is used to visualize the results of a single method's analysis run.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the specified metric against the 'T' column
    ax.plot(dataframe['T'], dataframe[metric_name], marker='o', linestyle='-')

    ax.set_xlabel("Number of Data Points (T)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend([metric_name.replace("_", " ").title()]) # Creates a clean legend, e.g., "Set Membership Area"
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log') # Logarithmic scale is often helpful for these metrics
    
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"-> Trend plot saved to: {output_path}")





def plot_multi_metric_comparison(
    dataframe: pd.DataFrame,
    metric_configs: List[Dict[str, str]],
    x_col: str,
    y_label: str,
    title: str,
    output_path: str
) -> None:
    """
    Plots a comparison of multiple metrics from a DataFrame on a single graph.
    
    This function is highly flexible and plots lines based on a configuration list,
    allowing it to compare any number of methods.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing all the results.
        metric_configs (List[Dict[str, str]]): A list of dictionaries, where each dict
            configures one line on the plot. 
            Required keys: 'col' (column name in DataFrame), 'label' (legend name).
            Optional keys: 'marker', 'linestyle'.
        x_col (str): The name of the column to use for the x-axis (e.g., 'T').
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        output_path (str): The full path to save the plot image.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Loop through the configuration and plot each specified metric
    for config in metric_configs:
        # Check if the column exists in the DataFrame to prevent errors
        if config['col'] in dataframe.columns:
            ax.plot(dataframe[x_col], dataframe[config['col']], 
                    marker=config.get('marker', None), 
                    linestyle=config.get('linestyle', '-'), 
                    label=config['label'],
                    color=config.get('color', None))
        else:
            print(f"Warning: Column '{config['col']}' not found in DataFrame. Skipping plot.")

    ax.set_xlabel("Number of Data Points (T)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log') # Logarithmic scale is often best for comparing these metrics
    
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"-> Multi-metric comparison plot saved to: {output_path}")




def plot_mc_metric_comparison(
    summary_df: pd.DataFrame,
    metric_configs: List[Dict[str, str]],
    x_col_name: str,
    y_label: str,
    title: str,
    output_path: str
) -> None:
    """
    Plots the mean of metrics from a Monte Carlo summary DataFrame, 
    including a shaded area for +/- one standard deviation, based on a config list.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # The x-axis is the index of the summary DataFrame, which is 'T'
    x_values = summary_df.index

    for config in metric_configs:
        # Construct the multi-index column names for mean and std
        mean_col = (config['col'], 'mean')
        std_col = (config['col'], 'std')

        # Check if the columns exist in the DataFrame
        if mean_col in summary_df.columns and std_col in summary_df.columns:
            mean_values = summary_df[mean_col]
            std_values = summary_df[std_col]
            
            # Plot the mean line
            ax.plot(x_values, mean_values, 
                    label=config['label'], 
                    color=config.get('color', None), 
                    marker=config.get('marker', None), 
                    linestyle=config.get('linestyle', '-'))
            
            # Add the shaded confidence band (+/- 1 std)
            ax.fill_between(x_values, mean_values - std_values, mean_values + std_values,
                            color=config.get('color', 'gray'), alpha=0.2)
        else:
            print(f"Warning: Columns for '{config['col']}' not found in summary DataFrame. Skipping plot line.")


    ax.set_xlabel(f"Number of Data Points ({x_col_name})")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"-> Monte Carlo plot saved to: {output_path}")