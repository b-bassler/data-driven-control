"""
Meta-experiment to compare the performance of Data-Dependent Bounds (Ellipse)
and Bootstrap Dean (Rectangle) over a range of data points (T).
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse
from src.plotting import plot_metric_comparison

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_comparison_experiment():
    """Orchestrates the full comparison experiment."""
    print("--- Starting Comparison Experiment over T (I.I.D. vs. Time-Series) ---")

    # === 3. Central Configuration ===
    T_MAX = 400
    T_RANGE = np.arange(5, T_MAX + 1, 5)
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 2000

    # === 4. Loop over T and collect metrics for BOTH methods ===
    results_list = []
    print(f"\nRunning analysis for T in {T_RANGE}...")
    for T in tqdm(T_RANGE, desc="Comparison Progress"):
        current_seed = T

        # --- Pipeline 1: Bootstrap Dean on Time-Series Data ---
        # 1a. Generate a new trajectory of length T
        state_data_ts_raw, input_data_ts_raw, _ = generate_time_series_data(
            system_params=TRUE_PARAMS,
            timesteps=T,
            output_path=GENERATED_DATA_DIR,  
            base_filename=f'temp_timeseries_T{T}', 
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W},
            seed=current_seed
        )
        state_data_ts = np.array([state_data_ts_raw.flatten()])
        input_data_ts = np.array([input_data_ts_raw.flatten()])

        # 1b. Analyze the trajectory data
        A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_data_ts, input_data_ts)
        bootstrap_results = perform_bootstrap_analysis(
            initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
            sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
            delta=CONFIDENCE_DELTA, seed=current_seed + 1
        )
        rect = ConfidenceRectangle(
            center=(A_est_bs.item(), B_est_bs.item()),
            epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B'])
        )
        
        # --- Pipeline 2: Data-Dependent Bounds on I.I.D. Data ---
        # 2a. Generate N=T new I.I.D. samples
        x_iid, u_iid, _, y_iid = generate_iid_samples(
            N=T,
            system_params=TRUE_PARAMS,
            params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
            output_path=GENERATED_DATA_DIR, # KORRIGIERT
            base_filename=f'temp_iid_T{T}', # KORRIGIERT
            seed=current_seed
        )
        
        # 2b. Analyze the I.I.D. data
        A_est_dd, B_est_dd = estimate_least_squares_iid(x_iid, u_iid, y_iid)
        if A_est_dd is not None:
            p_matrix = calculate_p_matrix_for_confidence_ellipse(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA)
            ellipse = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
        else: # Handle case where estimation fails
            ellipse = None

        # --- Store results for this T ---
        if ellipse is not None:
            results_list.append({
                'T': T,
                'rect_area': rect.area(),
                'rect_wcd': rect.worst_case_deviation(),
                'rect_max_dev_a': rect.axis_parallel_deviations()['max_dev_a'],
                'ellipse_area': ellipse.area(),
                'ellipse_wcd': ellipse.worst_case_deviation(),
                'ellipse_max_dev_a': ellipse.axis_parallel_deviations()['max_dev_a'],
            })

    # === 5. Process, save, and plot the final results ===
    # (Dieser Teil bleibt unverändert)
    print("\nProcessing and saving collected data...")
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "comparison_pipelines_over_T.csv")
    results_df.to_csv(results_path, index=False)
    print(f"-> Full comparison data saved to {results_path}")

    print("\nGenerating comparison plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    plot_metric_comparison(results_df, 'area', 'Area of Confidence Region', 'Area vs. T', os.path.join(figures_dir, "comparison_area.png"))
    plot_metric_comparison(results_df, 'wcd', 'Worst-Case Deviation', 'WCD vs. T', os.path.join(figures_dir, "comparison_wcd.png"))
    plot_metric_comparison(results_df, 'max_dev_a', 'Max Deviation for Parameter a', 'Max Dev for "a" vs. T', os.path.join(figures_dir, "comparison_dev_a.png"))

    print("\n--- Comparison Experiment Finished Successfully! ---")