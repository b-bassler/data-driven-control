"""
Comparison experiment for the two SysId methods over a range of T:
1. Data-Dependent Bounds (Ellipse) on I.I.D. data.
2. Bootstrap Dean (Rectangle) on time-series data.
This script is designed to be runnable for a single comparison, but also
importable for use in a Monte Carlo simulation.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse
from src.plotting import plot_multi_metric_comparison

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_sysid_comparison_experiment(data_seed: int) -> pd.DataFrame:
    """
    Orchestrates the comparison of the two SysId methods for a single data seed.

    Args:
        data_seed (int): The master seed for the data generation to ensure
                         different data for each Monte Carlo run.

    Returns:
        pd.DataFrame: A DataFrame containing the collected metrics over T.
    """
    print(f"--- Starting SysId Comparison Experiment (Seed: {data_seed}) ---")

    # === 3. Central Configuration ===
    T_MAX = 500
    T_RANGE = np.arange(40, T_MAX + 1, 40)
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.1**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 1000

    # === 4. Loop over T and collect metrics ===
    results_list = []
    print(f"\nRunning analysis for T in {T_RANGE}...")
    for T in tqdm(T_RANGE, desc="SysId Comparison Progress"):
        # Use a combination of master seed and T for unique runs
        current_seed = data_seed * 1000 + T
        metrics = {'T': T}

        # --- Pipeline 1: Data-Dependent Bounds on I.I.D. Data ---
        try:
            x_iid, u_iid, _, y_iid = generate_iid_samples(
                N=T, system_params=TRUE_PARAMS,
                params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
                output_path=GENERATED_DATA_DIR, base_filename=f"temp_iid_T{T}", seed=current_seed
            )
            A_est_dd, B_est_dd = estimate_least_squares_iid(x_iid, u_iid, y_iid)
            if A_est_dd is not None:
                p_matrix = calculate_p_matrix_for_confidence_ellipse(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA)
                ellipse = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
                devs = ellipse.axis_parallel_deviations()
                metrics['dd_bounds_area'] = ellipse.area()
                metrics['dd_bounds_wcd'] = ellipse.worst_case_deviation()
                metrics['dd_bounds_max_dev_a'] = devs['max_dev_a']
                metrics['dd_bounds_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: DD-Bounds failed for T={T} with error: {e}")

        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
            state_ts_raw, input_ts_raw, _ = generate_time_series_data(
                system_params=TRUE_PARAMS, timesteps=T, 
                output_path=GENERATED_DATA_DIR, base_filename=f"temp_timeseries_T{T}",
                noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=current_seed
            )
            state_ts, input_ts = np.array([state_ts_raw.flatten()]), np.array([input_ts_raw.flatten()])
            A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_ts, input_ts)
            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=CONFIDENCE_DELTA, seed=current_seed + 1
            )
            rect = ConfidenceRectangle(center=(A_est_bs.item(), B_est_bs.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            devs = rect.axis_parallel_deviations()
            metrics['bootstrap_area'] = rect.area()
            metrics['bootstrap_wcd'] = rect.worst_case_deviation()
            metrics['bootstrap_max_dev_a'] = devs['max_dev_a']
            metrics['bootstrap_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: Bootstrap failed for T={T} with error: {e}")

        results_list.append(metrics)

    # Convert the list of results into a clean DataFrame
    results_df = pd.DataFrame(results_list)
    return results_df


if __name__ == '__main__':
    """
    This block allows the script to be run directly for a single test run.
    It will generate the CSV and the plots for one master data seed.
    """
    # Perform a single run with a default seed
    final_df = run_sysid_comparison_experiment(data_seed=1)

    # --- Save and plot the results of this single run ---
    print("\nProcessing and saving final comparison data...")
    results_path = os.path.join(RESULTS_DIR, "sysid_comparison_over_T.csv")
    final_df.to_csv(results_path, index=False)
    print(f"-> Full comparison data saved to {results_path}")

    print("\nGenerating final comparison plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "sysid_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent (Ellipse)', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_area', 'label': 'Bootstrap (Rectangle)', 'marker': 'x', 'linestyle': '--'},
    ]
    plot_multi_metric_comparison(final_df, area_configs, 'T', 'Area', 'Area Comparison vs. T', os.path.join(figures_dir, "sysid_comparison_area.png"))
    
    # --- Plot configuration for Worst-Case Deviation ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (Ellipse)', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_wcd', 'label': 'Bootstrap (Rectangle)', 'marker': 'x', 'linestyle': '--'},
    ]
    plot_multi_metric_comparison(final_df, wcd_configs, 'T', 'WCD', 'WCD Comparison vs. T', os.path.join(figures_dir, "sysid_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
        {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a)', 'marker': 'x', 'linestyle': '-'},
        {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'marker': 'o', 'linestyle': '--'},
        {'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (b)', 'marker': 'x', 'linestyle': '--'},
    ]
    plot_multi_metric_comparison(final_df, max_dev_configs, 'T', 'Max Deviation', 'Max Deviation for "a" and "b" vs. T', os.path.join(figures_dir, "sysid_comparison_dev_a_and_b.png"))
    
    print("\n--- SysId Comparison Finished Successfully! ---")