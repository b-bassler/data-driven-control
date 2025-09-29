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


# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples
from src.system_identification import estimate_least_squares_iid, perform_bootstrap_analysis_iid
from src.plotting import plot_bootstrap_rectangle
from src.analysis import ConfidenceRectangle

RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')


def run_bootstrap_dean_iid():

        # === 3. Central configuration for the entire experiment ===
    N = 100
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    
    # Configuration for data generation
    DATA_SEED = 1
    SIGMA_W = 0.1
    SIGMA_U = 1.0
    
    # Configuration for the bootstrap analysis
    BOOTSTRAP_ITERATIONS = 2000
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_SEED = 2

    x_iid, u_iid, _, y_iid = generate_iid_samples(
    N=N, system_params=TRUE_PARAMS,
    params_config={'x_std_dev': 1.0, 'u_std_dev': SIGMA_U, 'w_std_dev': SIGMA_W},
    output_path=GENERATED_DATA_DIR, base_filename=f"temp_iid_T{N}", seed=DATA_SEED
    )

       # === Perform initial least-squares estimation ===
    A_hat, B_hat = estimate_least_squares_iid(x_iid, u_iid, y_iid)

    initial_estimate = (A_hat, B_hat)
    estimated_params = (A_hat.item(), B_hat.item())
    print(f"-> Initial estimate: A_hat = {estimated_params[0]:.6f}, B_hat = {estimated_params[1]:.6f}")
    
    # === Perform bootstrap analysis ===
    bootstrap_results = perform_bootstrap_analysis_iid(
                initial_estimate= (A_hat, B_hat),
                N = N,
                sigmas={'x':1.0, 'u': SIGMA_U, 'w': SIGMA_W},
                M=BOOTSTRAP_ITERATIONS,
                delta=CONFIDENCE_DELTA,
                seed=BOOTSTRAP_SEED
                )

    epsilons = (bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B'])
    print(f"-> Bootstrap analysis complete. Epsilon A: {epsilons[0]:.6f}, Epsilon B: {epsilons[1]:.6f}")

    # === 7. Analyze results using the ConfidenceRectangle class ===
    print("\nStep 3: Analyzing results with ConfidenceRectangle class...")
    rect = ConfidenceRectangle(center=estimated_params, epsilons=epsilons)
    area = rect.area()
    wcd = rect.worst_case_deviation()
    devs = rect.axis_parallel_deviations()

    print(f"-> Area: {area:.6f}")
    print(f"-> Worst-Case Deviation: {wcd:.6f}")
    print(f"-> Axis-Parallel Deviations: a={devs['max_dev_a']:.6f}, b={devs['max_dev_b']:.6f}")

    # === 8. Save the calculated metrics in a consistent format ===
    print("\nStep 4: Saving analysis results...")
    output_filename = f"bootstrap_dean_metrics_T{N}_M{BOOTSTRAP_ITERATIONS}.npz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    np.savez(
        output_path,
        a_hat=estimated_params[0],
        b_hat=estimated_params[1],
        area=area,
        worst_case_deviation=wcd,
        max_dev_a=devs['max_dev_a'],
        max_dev_b=devs['max_dev_b'],
        T=N,
        M=BOOTSTRAP_ITERATIONS
    )
    print(f"-> Metrics saved to {output_path}")

    # === 9. Generate visualization ===
    print("\nStep 5: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "bootstrap_iid_confidence_region.png")

    plot_bootstrap_rectangle(
        true_params=TRUE_PARAMS,
        estimated_params=estimated_params,
        epsilons=epsilons,
        confidence_delta=CONFIDENCE_DELTA,
        output_path=plot_path
    )

    print("\n--- Bootstrap Dean Experiment Finished ---")

 # === 10. Save Bound Geometry for Comparison Plot ===
    # =========================================================================
    print("\nStep 5: Saving bound geometry for final comparison plot...")
    
    # Define the path to the dedicated directory for comparison data
    comparison_dir = os.path.join(RESULTS_DIR, "comparison_data")
    os.makedirs(comparison_dir, exist_ok=True)
    bound_output_path = os.path.join(comparison_dir, f"bound_bootstrap_iid_N{N}.npz")

    # Save all necessary information to reconstruct the rectangle later
    np.savez(
        bound_output_path,
        type=np.array('rectangle'),
        method=np.array('Bootstrap (I.I.D.)'), # Updated label
        center=np.array(estimated_params),
        epsilons=np.array(epsilons)
    )
    print(f"-> Bound geometry saved to: {bound_output_path}")
    # =========================================================================

    print("\n--- Bootstrap Dean I.I.D. Experiment Finished ---")
    




if __name__ == '__main__':
    run_bootstrap_dean_iid()
