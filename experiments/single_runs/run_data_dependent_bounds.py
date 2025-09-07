"""
Experiment script for the Data-Dependent Bounds method as described by Dean et al.

This script performs a single run of the experiment:
1. Generates a fresh set of I.I.D. data.
2. Estimates the system parameters using a subset of the data.
3. Calculates the confidence ellipse based on the data.
4. Analyzes the geometric properties (metrics) of the ellipse using the OOP class.
5. Saves the results and generates a plot.
"""

import os
import numpy as np
from src.system_identification import estimate_least_squares_iid
from src.analysis import calculate_p_matrix_for_confidence_ellipse, ConfidenceEllipse
from src.plotting import plot_confidence_ellipse_from_matrix
from src.data_generation import generate_iid_samples

# --- 2. Define project paths relative to the project root ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_data_dependent_bounds_experiment():
    """Orchestrates the full Data-Dependent Bounds experiment."""
    print("--- Starting Full Data-Dependent Bounds Experiment ---")

    # === 3. Central configuration for the entire experiment ===
    N_SAMPLES = 10000
    DATA_SEED = 234
    BASE_FILENAME = 'iid_run_for_dd_bounds'
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV = np.sqrt((0.01**2) / 3)
    
    T_DATA_POINTS = 10  # Number of data points to use from the generated set
    CONFIDENCE_DELTA = 0.05  # Corresponds to 1 - delta confidence

    # 4. Generate I.I.D. data for the experiment
    print(f"\nStep 1: Generating {N_SAMPLES} I.I.D. data samples...")
    generation_config = {
        'N': N_SAMPLES, 'system_params': TRUE_PARAMS,
        'params_config': {
            'x_std_dev': 1.0, 'u_std_dev': 1.0, 'w_std_dev': NOISE_STD_DEV
        },
        'output_path': GENERATED_DATA_DIR, 'base_filename': BASE_FILENAME, 'seed': DATA_SEED
    }
    x_samples, u_samples, _, y_samples = generate_iid_samples(**generation_config)
    
    # 5. Prepare data subset for analysis
    print(f"\nStep 2: Preparing first {T_DATA_POINTS} data points for analysis...")
    x, u, y = x_samples[:T_DATA_POINTS], u_samples[:T_DATA_POINTS], y_samples[:T_DATA_POINTS]

    # 6. Perform least-squares estimation 
    print("\nStep 3: Performing least-squares estimation...")
    A_est_mat, B_est_mat = estimate_least_squares_iid(x, u, y)
    if A_est_mat is None:
        print("Estimation failed. Aborting experiment.")
        return
    estimated_params = (A_est_mat[0, 0], B_est_mat[0, 0])
    print(f"-> Estimated Parameters: a_hat = {estimated_params[0]:.4f}, b_hat = {estimated_params[1]:.4f}")

    # === 7. Analyze results using the ConfidenceEllipse class ===
    print("\nStep 4: Analyzing confidence ellipse...")
    
    # First, calculate the P-matrix, which defines the ellipse's shape
    p_matrix = calculate_p_matrix_for_confidence_ellipse(x, u, NOISE_STD_DEV, CONFIDENCE_DELTA)
    
    # Second, instantiate the ConfidenceEllipse to access its metric methods
    ellipse = ConfidenceEllipse(center=estimated_params, p_matrix=p_matrix)

    # Third, calculate all required metrics by calling the object's methods
    area = ellipse.area()
    wcd = ellipse.worst_case_deviation()
    devs = ellipse.axis_parallel_deviations()

    print(f"-> Area: {area:.6f}")
    print(f"-> Worst-Case Deviation: {wcd:.6f}")
    print(f"-> Axis-Parallel Deviations: a={devs['max_dev_a']:.6f}, b={devs['max_dev_b']:.6f}")

    # === 8. Save the calculated metrics in a consistent format ===
    print("\nStep 5: Saving analysis results...")
    output_filename = f"dd_bounds_metrics_T{T_DATA_POINTS}.npz"
    results_filepath = os.path.join(RESULTS_DIR, output_filename)
    np.savez(
        results_filepath,
        a_hat=estimated_params[0],
        b_hat=estimated_params[1],
        area=area,
        worst_case_deviation=wcd,
        max_dev_a=devs['max_dev_a'],
        max_dev_b=devs['max_dev_b'],
        T=T_DATA_POINTS
    )
    print(f"-> Metrics saved to: {results_filepath}")
    
    # === 9. Generate visualization ===
    print("\nStep 6: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_filepath = os.path.join(figures_dir, "data_dependent_ellipse.png")
    
    plot_confidence_ellipse_from_matrix(
        true_params=tuple(TRUE_PARAMS.values()),
        estimated_params=estimated_params,
        p_matrix=p_matrix,
        confidence_delta=CONFIDENCE_DELTA,
        T=T_DATA_POINTS,
        output_path=plot_filepath
    )

    print("\n--- Experiment finished successfully! ---")



