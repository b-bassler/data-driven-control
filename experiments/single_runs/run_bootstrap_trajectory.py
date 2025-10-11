"""
Experiment script for the Bootstrap Dean method trajectory data.

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
from src.data_generation import generate_trajectory_data
from src.system_identification import estimate_least_squares_trajectory, perform_bootstrap_analysis_trajectory
from src.plotting import plot_bootstrap_rectangle
from src.analysis import ConfidenceRectangle



# --- 2. Define project paths relative to the project root ---
RESULTS_DIR = os.path.join(project_root, 'results')
GENERATED_DATA_DIR = os.path.join(project_root, 'data', 'generated')

def run_bootstrap_dean_trajectory():
    """Orchestrates the full Bootstrap Dean experiment."""
    print("--- Starting Bootstrap Dean Experiment ---")

    # === 3. Central configuration for the entire experiment ===
    T = 200
    TRUE_PARAMS = (0.5, 0.5)  
    
    # Configuration for data generation
    DATA_SEED = 4
    SIGMA_W = 0.1
    SIGMA_U = 1.0
    
    # Configuration for the bootstrap analysis
    BOOTSTRAP_ITERATIONS = 2000
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_SEED = 200
    
    # === 4. Generate time-series data ===
    print(f"\nStep 1: Generating time-series data with T={T}...")
    generation_config = {
        'system_params': {'a': TRUE_PARAMS[0], 'b': TRUE_PARAMS[1]},
        'timesteps': T,
        'output_path': GENERATED_DATA_DIR,
        'base_filename': f'timeseries_for_bootstrap_T{T}',
        'noise_config': {'distribution': 'gaussian', 'std_dev': SIGMA_W},
        'seed': DATA_SEED
    }
    state_data_raw, input_data_raw, _ = generate_trajectory_data(**generation_config)
    # Reshape to (N, T) format for the estimator, where N=1
    state_data = np.array([state_data_raw.flatten()])
    input_data = np.array([input_data_raw.flatten()])

    # === 5. Perform initial least-squares estimation ===
    print("\nStep 2: Performing initial LS estimation...")
    A_hat, B_hat = estimate_least_squares_trajectory(state_data, input_data)
    initial_estimate = (A_hat, B_hat)
    estimated_params = (A_hat.item(), B_hat.item())
    print(f"-> Initial estimate: A_hat = {estimated_params[0]:.6f}, B_hat = {estimated_params[1]:.6f}")
    
    # === 6. Perform bootstrap analysis ===
    bootstrap_results = perform_bootstrap_analysis_trajectory(
        initial_estimate=initial_estimate,
        data_shape=(state_data.shape[0], T), # (N, T)
        sigmas={'u': SIGMA_U, 'w': SIGMA_W},
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
    output_filename = f"bootstrap_dean_metrics_T{T}_M{BOOTSTRAP_ITERATIONS}.npz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    np.savez(
        output_path,
        a_hat=estimated_params[0],
        b_hat=estimated_params[1],
        area=area,
        worst_case_deviation=wcd,
        max_dev_a=devs['max_dev_a'],
        max_dev_b=devs['max_dev_b'],
        T=T,
        M=BOOTSTRAP_ITERATIONS
    )
    print(f"-> Metrics saved to {output_path}")

    # === 9. Generate visualization ===
    print("\nStep 5: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "bootstrap_trajectory_confidence_region.png")

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
    print("\nStep 6: Saving bound geometry for final comparison plot...")
    
    # Define the path to the dedicated directory for comparison data
    comparison_dir = os.path.join(RESULTS_DIR, "comparison_data")
    os.makedirs(comparison_dir, exist_ok=True)
    bound_output_path = os.path.join(comparison_dir, f"bound_bootstrap_trajectory_N{T}.npz")

    # Save all necessary information to reconstruct the rectangle later
    np.savez(
        bound_output_path,
        type=np.array('rectangle'),
        method=np.array('Bootstrap (Trajectory)'), # A clear name for the legend
        center=np.array(estimated_params),
        epsilons=np.array(epsilons) # Key difference: saving epsilons for the rectangle
    )
    print(f"-> Bound geometry saved to: {bound_output_path}")
    # =========================================================================

    print("\n--- Bootstrap Dean Experiment Finished ---")


if __name__ == '__main__':
    run_bootstrap_dean_trajectory()
    