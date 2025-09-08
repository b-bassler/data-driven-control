"""
WORKER SCRIPT for the final meta-experiment comparing all three methods.
This script runs a full comparison over T for a single data seed and
is designed to be called by a Monte Carlo simulation orchestrator.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import chi2

# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse
from src.set_membership import calculate_ellipse_from_qmi

# --- 2. Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_final_comparison_experiment(data_seed: int) -> pd.DataFrame:
    """
    Orchestrates the full comparison of all three methods for a single master data seed.
    """
    # === 3. Central Configuration ===
    T_RANGE = [8, 10, 15, 20, 30, 40, 50, 70, 90, 110, 150, 200, 300, 400, 500]
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    DEGREES_OF_FREEDOM = 2
    BOOTSTRAP_ITERATIONS = 2000

    # === 4. Loop over T and collect all metrics ===
    results_list = []
    for T in tqdm(T_RANGE, desc=f"Seed {data_seed} Progress", leave=False):
        current_seed = data_seed * 100 + T
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
                # Store the point estimate
                metrics['dd_bounds_a_hat'] = A_est_dd.item()
                metrics['dd_bounds_b_hat'] = B_est_dd.item()
                
                p_matrix = calculate_p_matrix_for_confidence_ellipse(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA)
                ellipse = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
                devs = ellipse.axis_parallel_deviations()
                metrics['dd_bounds_area'] = ellipse.area()
                metrics['dd_bounds_wcd'] = ellipse.worst_case_deviation()
                metrics['dd_bounds_max_dev_a'] = devs['max_dev_a']
                metrics['dd_bounds_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: DD-Bounds failed for T={T}, Seed={data_seed} with error: {e}")

        # --- Generate shared time-series data for the next two pipelines ---
        state_ts_raw, input_ts_raw, _ = generate_time_series_data(
            system_params=TRUE_PARAMS, timesteps=T, 
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_timeseries_T{T}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=current_seed
        )
        state_ts, input_ts = np.array([state_ts_raw.flatten()]), np.array([input_ts_raw.flatten()])

        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
            A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_ts, input_ts)
            # Store the point estimate
            metrics['bootstrap_a_hat'] = A_est_bs.item()
            metrics['bootstrap_b_hat'] = B_est_bs.item()

            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=(CONFIDENCE_DELTA/2), seed=current_seed + 1
            )
            rect = ConfidenceRectangle(center=(A_est_bs.item(), B_est_bs.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            devs = rect.axis_parallel_deviations()
            metrics['bootstrap_area'] = rect.area()
            metrics['bootstrap_wcd'] = rect.worst_case_deviation()
            metrics['bootstrap_max_dev_a'] = devs['max_dev_a']
            metrics['bootstrap_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: Bootstrap failed for T={T}, Seed={data_seed} with error: {e}")

        # --- Pipeline 3: Set Membership via direct QMI on Time-Series Data ---
        try:
            X_plus, X_minus, U_minus = state_ts[:, 1:T+1], state_ts[:, :T], input_ts[:, :T]
            c_delta = chi2.ppf(1 - CONFIDENCE_DELTA, df=DEGREES_OF_FREEDOM)
            Phi11 = (NOISE_STD_DEV_W**2) * c_delta * np.eye(1); Phi12 = np.zeros((1, T)); Phi21 = Phi12.T
            Z_reg = np.vstack([X_minus, U_minus])
            Phi22 = -np.linalg.pinv(Z_reg) @ Z_reg
            
            qmi_results = calculate_ellipse_from_qmi(X_plus, X_minus, U_minus, Phi11, Phi12, Phi21, Phi22)
            if qmi_results:
                # The 'center' of the QMI ellipse is its point estimate
                center = qmi_results['center']
                metrics['set_membership_a_hat'] = center[0]
                metrics['set_membership_b_hat'] = center[1]

                ellipse_qmi = ConfidenceEllipse(center=center, p_matrix=qmi_results['shape_matrix'])
                devs = ellipse_qmi.axis_parallel_deviations()
                metrics['set_membership_area'] = ellipse_qmi.area()
                metrics['set_membership_wcd'] = ellipse_qmi.worst_case_deviation()
                metrics['set_membership_max_dev_a'] = devs['max_dev_a']
                metrics['set_membership_max_dev_b'] = devs['max_dev_b']
        except (np.linalg.LinAlgError, Exception) as e:
            print(f"Warning: Set Membership (QMI) failed for T={T}, Seed={data_seed} with error: {e}")

        results_list.append(metrics)

    # --- 5. Return the collected data as a DataFrame ---
    return pd.DataFrame(results_list)

# This block allows the script to be run directly for a single test run.
if __name__ == '__main__':
    from src.plotting import plot_multi_metric_comparison
    
    # Perform a single run with a default seed
    final_df = run_final_comparison_experiment(data_seed=0)

    if final_df is not None and not final_df.empty:
        # --- Save and plot the results of this single run ---
        print("\nProcessing and saving final comparison data for single run...")
        results_path = os.path.join(RESULTS_DIR, "final_comparison_over_T_single_run.csv")
        final_df.to_csv(results_path, index=False)
        print(f"-> Full comparison data saved to {results_path}")

        print("\nGenerating final comparison plots for single run...")
        figures_dir = os.path.join(RESULTS_DIR, "figures", "final_comparison")
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot configurations for a single run
        area_configs = [
            {'col': 'dd_bounds_area', 'label': 'Data-Dependent (Ellipse)', 'marker': 'o', 'linestyle': '-'},
            {'col': 'bootstrap_area', 'label': 'Bootstrap (Rectangle)', 'marker': 'x', 'linestyle': '--'},
            {'col': 'set_membership_area', 'label': 'Set Membership (QMI)', 'marker': 's', 'linestyle': ':'}
        ]
        plot_multi_metric_comparison(final_df, area_configs, 'T', 'Area', 'Area Comparison vs. T (Single Run)', os.path.join(figures_dir, "final_comparison_area_single_run.png"))
        
        print("\n--- Single Final Comparison Run Finished Successfully! ---")