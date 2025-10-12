"""
Experiment script for the Bootstrap Dean method for i.i.d. data.

This script performs a single run of the experiment:
1. Generates the required time-series data.
2. Performs an initial estimation on this data.
3. Runs the bootstrap analysis loop to find confidence bounds (epsilons).
4. Analyzes the resulting confidence rectangle using the OOP class.
5. Saves the results and generates a plot.
"""

import os
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid, perform_bootstrap_analysis_iid
from src.plotting import plot_bootstrap_rectangle
from src.analysis import ConfidenceRectangle



def run_bootstrap_dean_iid():
    """
    Executes a single run of the parametric bootstrap experiment for I.I.D. data.
    
    This script performs the following steps:
    1.  Configures the experiment parameters.
    2.  Generates a set of I.I.D. data from a true system.
    3.  Computes an initial least-squares estimate of the system parameters.
    4.  Runs a parametric bootstrap analysis to find confidence bounds.
    5.  Analyzes and saves the results (metrics and plots).
    6.  Saves the confidence bound geometry for later comparison.
    """
    print("--- Starting Bootstrap Dean Experiment for I.I.D. Data ---")

    # Step 1: Central Configuration
    N = 100  # Number of i.i.d. samples
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    
    # Configuration for the initial data generation
    DATA_SEED = 1
    SIGMA_W = 0.1
    SIGMA_U = 1.0
    
    # Configuration for the bootstrap analysis
    BOOTSTRAP_ITERATIONS = 2000
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_SEED = 2

    # Step 2: Generate I.I.D. Data
    # The data is generated in the shape (features, samples), e.g., (1, N)
    print(f"\nStep 1: Generating {N} I.I.D. data samples...")
    x_iid, u_iid, _, y_iid = generate_iid_samples(
        N=N, system_params=TRUE_PARAMS,
        params_config={'x_std_dev': 1.0, 'u_std_dev': SIGMA_U, 'w_std_dev': SIGMA_W},
        output_path=GENERATED_DATA_DIR, base_filename=f"temp_iid_N{N}", seed=DATA_SEED
    )

    # Step 3: Perform Initial Least-Squares Estimation
    print("\nStep 2: Performing initial least-squares estimation...")
    A_hat, B_hat = estimate_least_squares_iid(x_iid, u_iid, y_iid)
    if A_hat is None:
        print("Initial estimation failed. Aborting experiment.")
        return

    estimated_params = (A_hat.item(), B_hat.item())
    print(f"-> Initial estimate: A_hat = {estimated_params[0]:.6f}, B_hat = {estimated_params[1]:.6f}")
    
    # Step 4: Perform Bootstrap Analysis
    print(f"\nStep 3: Running bootstrap analysis with {BOOTSTRAP_ITERATIONS} iterations...")
    bootstrap_results = perform_bootstrap_analysis_iid(
        initial_estimate=(A_hat, B_hat),
        N=N,
        sigmas={'x': 1.0, 'u': SIGMA_U, 'w': SIGMA_W},
        M=BOOTSTRAP_ITERATIONS,
        delta=CONFIDENCE_DELTA,
        seed=BOOTSTRAP_SEED
    )

    epsilons = (bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B'])
    print(f"-> Bootstrap analysis complete. Epsilon A: {epsilons[0]:.6f}, Epsilon B: {epsilons[1]:.6f}")

    # Step 5: Analyze and Save Results
    print("\nStep 4: Analyzing results and saving metrics...")
    rect = ConfidenceRectangle(center=estimated_params, epsilons=epsilons)
    area = rect.area()
    wcd = rect.worst_case_deviation()
    devs = rect.axis_parallel_deviations()

    # Save primary metrics
    output_filename = f"bootstrap_dean_metrics_N{N}_M{BOOTSTRAP_ITERATIONS}.npz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    np.savez(
        output_path,
        a_hat=estimated_params[0], b_hat=estimated_params[1],
        area=area, worst_case_deviation=wcd,
        max_dev_a=devs['max_dev_a'], max_dev_b=devs['max_dev_b'],
        N=N, M=BOOTSTRAP_ITERATIONS
    )
    print(f"-> Metrics saved to {output_path}")

    # Step 6: Generate Visualization
    print("\nStep 5: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "bootstrap_iid_confidence_region.png")
    plot_bootstrap_rectangle(
        true_params=TRUE_PARAMS, estimated_params=estimated_params,
        epsilons=epsilons, confidence_delta=CONFIDENCE_DELTA,
        output_path=plot_path
    )
    print(f"-> Plot saved to {plot_path}")

    # Step 7: Save Bound Geometry for Comparison Plot
    print("\nStep 6: Saving bound geometry for final comparison plot...")
    comparison_dir = os.path.join(RESULTS_DIR, "comparison_data")
    os.makedirs(comparison_dir, exist_ok=True)
    bound_output_path = os.path.join(comparison_dir, f"bound_bootstrap_iid_N{N}.npz")

    np.savez(
        bound_output_path,
        type=np.array('rectangle'),
        method=np.array('Bootstrap (I.I.D.)'),
        center=np.array(estimated_params),
        epsilons=np.array(epsilons)
    )
    print(f"-> Bound geometry saved to: {bound_output_path}")
    
    print("\n--- Bootstrap Dean I.I.D. Experiment Finished ---")


if __name__ == '__main__':
    run_bootstrap_dean_iid()
