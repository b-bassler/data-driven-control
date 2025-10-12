"""
Experiment script for the Data-Dependent Bounds method as described by Dean et al.

This script performs a single run of the experiment, directly following the theory:
1. Generates a set of I.I.D. data of a specified length T.
2. Estimates the system parameters using this data.
3. Calculates the confidence ellipse based on the same data.
4. Analyzes the geometric properties (metrics) of the ellipse.
5. Saves the results, generates a plot, and saves the bound geometry.
"""

import os
import sys
import numpy as np

# Path Setup 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system_identification import estimate_least_squares_iid
from src.analysis import calculate_p_matrix_ddbounds_iid, ConfidenceEllipse
from src.plotting import plot_confidence_ellipse_from_matrix
from src.data_generation import generate_iid_samples

# --- 2. Define project paths relative to the project root ---
RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')


def run_data_dependent_bounds_experiment():
    """Orchestrates the full Data-Dependent Bounds experiment."""
    print("--- Starting Single Run: Data-Dependent Bounds (I.I.D.) ---")

    # === 3. Central configuration for the entire experiment ===
    T = 100
    DATA_SEED = 2 # Consistent seed for comparison
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    TRUE_PARAMS_TUPLE = tuple(TRUE_PARAMS.values())
    NOISE_STD_DEV = 0.1
    CONFIDENCE_DELTA = 0.05

    # 4. Generate I.I.D. data for the experiment
    print(f"\nStep 1: Generating {T} I.I.D. data samples...")
    generation_config = {
        'N': T, 'system_params': TRUE_PARAMS,
        'params_config': {
            'x_std_dev': 1.0, 'u_std_dev': 1.0, 'w_std_dev': NOISE_STD_DEV
        },
        'output_path': GENERATED_DATA_DIR, 'base_filename': f'dd_bounds_iid_data_N{T}', 'seed': DATA_SEED
    }
    x_samples, u_samples, _, y_samples = generate_iid_samples(**generation_config)
    
    
    # 5. Perform least-squares estimation 
    print("\nStep 2: Performing least-squares estimation...")
    A_est, B_est = estimate_least_squares_iid(x_samples, u_samples, y_samples)
    if A_est is None:
        print("Estimation failed. Aborting experiment.")
        return
    estimated_params = (A_est[0, 0], B_est[0, 0])
    print(f"-> Estimated Parameters: a_hat = {estimated_params[0]:.4f}, b_hat = {estimated_params[1]:.4f}")

    # === 6. Analyze confidence ellipse ===
    print("\nStep 3: Analyzing confidence ellipse...")
    
    # Calculate the P-matrix, which defines the ellipse's shape
    p_matrix = calculate_p_matrix_ddbounds_iid(x_samples, u_samples, NOISE_STD_DEV, CONFIDENCE_DELTA)


    ellipse_dd = ConfidenceEllipse(center=(A_est.item(), B_est.item()), p_matrix=p_matrix)
    if not ellipse_dd.contains(TRUE_PARAMS_TUPLE):
                print("Lies outside")
    # Instantiate the ConfidenceEllipse to access its metric methods
    ellipse = ConfidenceEllipse(center=estimated_params, p_matrix=p_matrix)

    # === 7. Generate visualization ===
    print("\nStep 4: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "single_runs")
    os.makedirs(figures_dir, exist_ok=True)
    plot_filepath = os.path.join(figures_dir, f"dd_bounds_iid_N{T}.png")
    
    plot_confidence_ellipse_from_matrix(
        true_params=tuple(TRUE_PARAMS.values()),
        estimated_params=estimated_params,
        p_matrix=p_matrix,
        confidence_delta=CONFIDENCE_DELTA,
        T=T,
        output_path=plot_filepath
    )

    # === 8. Save Bound Geometry for Comparison Plot ===
    print("\nStep 5: Saving bound geometry for final comparison plot...")
    
    comparison_dir = os.path.join(RESULTS_DIR, "comparison_data")
    os.makedirs(comparison_dir, exist_ok=True)
    bound_output_path = os.path.join(comparison_dir, f"bound_dd_bounds_iid_N{T}.npz")

    np.savez(
        bound_output_path,
        type=np.array('ellipse'),
        method=np.array('Data-Dependent (Dean)'),
        center=np.array(estimated_params),
        p_matrix=p_matrix
    )
    print(f"-> Bound geometry saved to: {bound_output_path}")

    print("\n--- Experiment finished successfully! ---")


if __name__ == '__main__':
    run_data_dependent_bounds_experiment()