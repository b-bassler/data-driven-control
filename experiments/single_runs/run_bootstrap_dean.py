"""
Experiment script for the Bootstrap Dean method.

This script performs a single run of the experiment:
1. Generates the required time-series data.
2. Performs an initial estimation on this data.
3. Runs the bootstrap analysis loop to find confidence bounds (epsilons).
4. Analyzes the resulting confidence rectangle using the OOP class.
5. Saves the results and generates a plot.
"""

import os
import numpy as np

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_time_series_data
from src.system_identification import estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.plotting import plot_bootstrap_rectangle
from src.analysis import ConfidenceRectangle

# --- 2. Define project paths relative to the project root ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_bootstrap_dean_experiment():
    """Orchestrates the full Bootstrap Dean experiment."""
    print("--- Starting Bootstrap Dean Experiment ---")

    # === 3. Central configuration for the entire experiment ===
    T_DATA_POINTS_TO_USE = 400
    TRUE_PARAMS = (0.5, 0.5)  # (a_true, b_true)
    
    # Configuration for data generation
    DATA_SEED = 2
    SIGMA_W = np.sqrt((0.01**2) / 3) 
    SIGMA_U = 1.0
    
    # Configuration for the bootstrap analysis
    BOOTSTRAP_ITERATIONS = 2000
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_SEED = 1
    
    # === 4. Generate time-series data ===
    print(f"\nStep 1: Generating time-series data with T={T_DATA_POINTS_TO_USE}...")
    generation_config = {
        'system_params': {'a': TRUE_PARAMS[0], 'b': TRUE_PARAMS[1]},
        'timesteps': T_DATA_POINTS_TO_USE,
        'output_path': GENERATED_DATA_DIR,
        'base_filename': f'timeseries_for_bootstrap_T{T_DATA_POINTS_TO_USE}',
        'noise_config': {'distribution': 'gaussian', 'std_dev': SIGMA_W},
        'seed': DATA_SEED
    }
    state_data_raw, input_data_raw, _ = generate_time_series_data(**generation_config)
    # Reshape to (N, T) format for the estimator, where N=1
    state_data = np.array([state_data_raw.flatten()])
    input_data = np.array([input_data_raw.flatten()])

    # === 5. Perform initial least-squares estimation ===
    print("\nStep 2: Performing initial LS estimation...")
    A_hat, B_hat = estimate_least_squares_timeseries(state_data, input_data)
    initial_estimate = (A_hat, B_hat)
    estimated_params = (A_hat.item(), B_hat.item())
    print(f"-> Initial estimate: A_hat = {estimated_params[0]:.6f}, B_hat = {estimated_params[1]:.6f}")
    
    # === 6. Perform bootstrap analysis ===
    bootstrap_results = perform_bootstrap_analysis(
        initial_estimate=initial_estimate,
        data_shape=(state_data.shape[0], T_DATA_POINTS_TO_USE), # (N, T)
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
    output_filename = f"bootstrap_dean_metrics_T{T_DATA_POINTS_TO_USE}_M{BOOTSTRAP_ITERATIONS}.npz"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    np.savez(
        output_path,
        a_hat=estimated_params[0],
        b_hat=estimated_params[1],
        area=area,
        worst_case_deviation=wcd,
        max_dev_a=devs['max_dev_a'],
        max_dev_b=devs['max_dev_b'],
        T=T_DATA_POINTS_TO_USE,
        M=BOOTSTRAP_ITERATIONS
    )
    print(f"-> Metrics saved to {output_path}")

    # === 9. Generate visualization ===
    print("\nStep 5: Generating plot...")
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    plot_path = os.path.join(figures_dir, "bootstrap_dean_confidence_region.png")

    plot_bootstrap_rectangle(
        true_params=TRUE_PARAMS,
        estimated_params=estimated_params,
        epsilons=epsilons,
        confidence_delta=CONFIDENCE_DELTA,
        output_path=plot_path
    )

    print("\n--- Bootstrap Dean Experiment Finished ---")
    