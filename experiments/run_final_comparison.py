"""
Final meta-experiment to compare all three implemented methods over a range of T:
1. Data-Dependent Bounds (Ellipse) on I.I.D. data.
2. Bootstrap Dean (Rectangle) on time-series data.
3. Set Membership (MVEE) on time-series data.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, MVEEllipse, calculate_p_matrix_for_confidence_ellipse
from src.set_membership import find_feasible_set, calculate_mvee
from src.plotting import plot_multi_metric_comparison

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# In experiments/run_final_comparison.py

def run_final_comparison_experiment():
    """Orchestrates the full comparison of all three methods."""
    print("--- Starting Final Comparison Experiment over T ---")

    # === Central Configuration ===
    T_MAX = 300
    T_RANGE = np.arange(8, T_MAX + 1, 10)
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    
    BOOTSTRAP_ITERATIONS = 1000
    SET_MEMBERSHIP_GRID_DENSITY = 500

    # === Loop over T and collect all metrics ===
    results_list = []
    print(f"\nRunning analysis for T in {T_RANGE}...")
    for T in tqdm(T_RANGE, desc="Final Comparison Progress"):
        current_seed = T
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

        # --- Pipeline 2 & 3 use the same time-series data ---
        state_ts_raw, input_ts_raw, _ = generate_time_series_data(
            system_params=TRUE_PARAMS, timesteps=T, 
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_timeseries_T{T}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=current_seed
        )
        state_ts, input_ts = np.array([state_ts_raw.flatten()]), np.array([input_ts_raw.flatten()])

        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
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

        # --- Pipeline 3: Set Membership on Time-Series Data ---
        try:
            X_plus, X_minus, U_minus = state_ts[:, 1:T+1], state_ts[:, :T], input_ts[:, :T]
            c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=2)
            Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1); Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
            Z_reg = np.vstack([X_minus, U_minus])
            Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
            Phi = np.block([[Phi11, Phi12], [Phi21, Phi22]])
            valid_pairs = find_feasible_set(X_plus, X_minus, U_minus, Phi, a_range=(0.48, 0.52), b_range=(0.48, 0.52), grid_density=SET_MEMBERSHIP_GRID_DENSITY)
            mvee_results = calculate_mvee(valid_pairs)
            if mvee_results:
                mvee_ellipse = MVEEllipse(mvee_results)
                devs = mvee_ellipse.axis_parallel_deviations()
                metrics['set_membership_area'] = mvee_ellipse.area()
                metrics['set_membership_wcd'] = mvee_ellipse.worst_case_deviation()
                metrics['set_membership_max_dev_a'] = devs['max_dev_a']
                metrics['set_membership_max_dev_b'] = devs['max_dev_b']
        except (np.linalg.LinAlgError, Exception) as e:
            print(f"Warning: Set Membership failed for T={T} with error: {e}")

        results_list.append(metrics)

    # === Process, save, and plot the final results ===
    if not results_list:
        print("\nNo results were collected. Cannot proceed.")
        return
        
    print("\nProcessing and saving final comparison data...")
    results_df = pd.DataFrame(results_list)
    results_path = os.path.join(RESULTS_DIR, "final_comparison_over_T.csv")
    results_df.to_csv(results_path, index=False)
    print(f"-> Full comparison data saved to {results_path}")

    print("\nGenerating final comparison plots...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "final_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent (Ellipse)', 'marker': 'o', 'linestyle': '-', 'color': 'blue', 'markersize': 4},
        {'col': 'bootstrap_area', 'label': 'Bootstrap (Rectangle)', 'marker': 'x', 'linestyle': '-', 'color': 'orange', 'markersize': 4},
        {'col': 'set_membership_area', 'label': 'Set Membership (MVEE)', 'marker': 's', 'linestyle': '-', 'color': 'green', 'markersize':4}
    ]
    plot_multi_metric_comparison(results_df, area_configs, 'T', 'Area of Confidence Region (log scale)', 'Area Comparison vs. T', os.path.join(figures_dir, "final_comparison_area.png"))

    # --- Plot configuration for Worst-Case Deviation ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent (Ellipse)', 'marker': 'o', 'linestyle': '-', 'color': 'blue', 'markersize': 4},
        {'col': 'bootstrap_wcd', 'label': 'Bootstrap (Rectangle)', 'marker': 'x', 'linestyle': '-', 'color': 'orange', 'markersize': 4},
        {'col': 'set_membership_wcd', 'label': 'Set Membership (MVEE)', 'marker': 's', 'linestyle': '-', 'color': 'green', 'markersize': 4}
    ]
    plot_multi_metric_comparison(results_df, wcd_configs, 'T', 'Worst-Case Deviation (log scale)', 'WCD Comparison vs. T', os.path.join(figures_dir, "final_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
    # --- Method 1: Data-Dependent ---
    {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'marker': 'o', 'linestyle': '-', 'color': 'blue', 'markersize': 4},
    {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'marker': 'o', 'linestyle': '--', 'color': 'blue', 'markersize': 4},

    # --- Method 2: Bootstrap ---
    {'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a)', 'marker': 'x', 'linestyle': '-', 'color': 'orange', 'markersize': 4},
    {'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (b)', 'marker': 'x', 'linestyle': '--', 'color': 'orange', 'markersize': 4},

    # --- Method 3: Set Membership ---
    {'col': 'set_membership_max_dev_a', 'label': 'Set Membership (a)', 'marker': 's', 'linestyle': '-', 'color': 'green', 'markersize': 4},
    {'col': 'set_membership_max_dev_b', 'label': 'Set Membership (b)', 'marker': 's', 'linestyle': '--', 'color': 'green', 'markersize': 4}
    ]
    plot_multi_metric_comparison(results_df, max_dev_configs, 'T', 'Max axis-parallel deviation (log scale)', 'Max deviation for a and b ', os.path.join(figures_dir, "final_comparison_dev_a_and_b.png"))

    print("\n--- Final Comparison Experiment Finished Successfully! ---")