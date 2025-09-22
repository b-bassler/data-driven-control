import os
import sys

import os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import chi2


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


# --- 1. Import all required tools from the src library ---
from src.data_generation import generate_time_series_data
from src.system_identification import estimate_least_squares_timeseries, perform_bootstrap_analysis
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse, calculate_tsiams_ellipse_matrix
from src.set_membership import calculate_ellipse_from_qmi
from src.plotting import plot_mc_metric_comparison




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
    
        # Parameters specific to the Tsiams method
    C = 0.00140625
    TAU = 2


    # === 4. Loop over T and collect all metrics ===
    results_list = []
    for T in tqdm(T_RANGE, desc=f"Seed {data_seed} Progress", leave=False):
        current_seed = data_seed * 100 + T
        metrics = {'T': T}

        
        # --- Generate shared time-series data for all three pipelines ---
        state_ts_raw, input_ts_raw, _ = generate_time_series_data(
            system_params=TRUE_PARAMS, timesteps=T, 
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_timeseries_T{T}",
            noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=current_seed
        )
        state_ts, input_ts = np.array([state_ts_raw.flatten()]), np.array([input_ts_raw.flatten()])



        # --- Pipeline 1: Data-Dependent Bounds on Trajectory Data by Tsiamis ---
        try:

            A_est_dd, B_est_dd = estimate_least_squares_timeseries(state_ts, input_ts)
            if A_est_dd is not None:
                # Store the point estimate
                metrics['dd_bounds_a_hat'] = A_est_dd.item()
                metrics['dd_bounds_b_hat'] = B_est_dd.item()
                

                # Calculate Tsiamis Ellipse Matrix
                tsiams_results = calculate_tsiams_ellipse_matrix(
                    state_data=state_ts[:, :T], 
                    input_data=input_ts,
                    true_A=np.array([[TRUE_PARAMS['a']]]), 
                    true_B=np.array([[TRUE_PARAMS['b']]]),
                    sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W},
                    delta=CONFIDENCE_DELTA, 
                    c=C, 
                    tau=TAU
                )
                p_matrix = tsiams_results['p_matrix']
                ellipse = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
                devs = ellipse.axis_parallel_deviations()
                metrics['dd_bounds_area'] = ellipse.area()
                metrics['dd_bounds_wcd'] = ellipse.worst_case_deviation()
                metrics['dd_bounds_max_dev_a'] = devs['max_dev_a']
                metrics['dd_bounds_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: DD-Bounds failed for T={T}, Seed={data_seed} with error: {e}")


        # --- Pipeline 2: Bootstrap Dean on Time-Series Data ---
        try:
            A_est_bs, B_est_bs = estimate_least_squares_timeseries(state_ts, input_ts)
            # Store the point estimate
            metrics['bootstrap_a_hat'] = A_est_bs.item()
            metrics['bootstrap_b_hat'] = B_est_bs.item()

            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs, B_est_bs), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                # Bonferroni correction with delta/2 
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





def run_mc_trajectory_comparison(num_mc_runs: int = 10):
    """
    Performs the main Monte Carlo loop for all three methods.
    """
    print(f"--- Starting FINAL Monte Carlo Simulation with {num_mc_runs} runs ---")

    # A list to store the results DataFrame from each run
    all_dataframes = []

    # Use a ProcessPoolExecutor to run the Monte Carlo simulations in parallel.
    with ProcessPoolExecutor() as executor:
        seeds = range(num_mc_runs)
        
        results_iterator = executor.map(run_final_comparison_experiment, seeds)
        
        # Collect results as they are completed and add the run_id.
        for i, single_run_df in enumerate(tqdm(results_iterator, total=num_mc_runs, desc="Monte Carlo Runs")):
            if single_run_df is not None and not single_run_df.empty:
                single_run_df['run_id'] = i
                all_dataframes.append(single_run_df)

    # --- Process and analyze the collected results ---
    print("\nProcessing and analyzing all Monte Carlo results...")
    
    if not all_dataframes:
        print("Warning: No dataframes were collected. Aborting analysis.")
        return
        
    full_results_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save the raw data from all runs to a new file
    raw_results_path = os.path.join(RESULTS_DIR, "mc_final_comparison_raw.csv")
    full_results_df.to_csv(raw_results_path, index=False)
    print(f"-> Raw data from all {len(all_dataframes)} runs saved to {raw_results_path}")

    # Calculate statistics (mean and std) for each metric at each value of T
    summary_df = full_results_df.groupby('T').agg(['mean', 'std'])

    # Save the summary statistics to a new file
    summary_path = os.path.join(RESULTS_DIR, "mc_final_comparison_summary.csv")
    summary_df.to_csv(summary_path)
    print(f"-> Summary statistics saved to {summary_path}")

    # === Generate final plots with confidence bands for all three methods ===
    print("\nGenerating final comparison plots with confidence bands...")
    figures_dir = os.path.join(RESULTS_DIR, "figures", "mc_final_comparison")
    os.makedirs(figures_dir, exist_ok=True)
    
    # --- Plot configuration for Area ---
    area_configs = [
        {'col': 'dd_bounds_area', 'label': 'Data-Dependent', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_area', 'label': 'Bootstrap', 'color': 'orange', 'marker': 'x'},
        {'col': 'set_membership_area', 'label': 'Set Membership (QMI)', 'color': 'green', 'marker': 's'}
    ]
    plot_mc_metric_comparison(summary_df, area_configs, 'T', 'Mean Area (log scale)', 
                              'Monte Carlo: Mean Area Comparison', 
                              os.path.join(figures_dir, "mc_final_comparison_area.png"))
    
    # --- Plot configuration for Worst-Case Deviation ---
    wcd_configs = [
        {'col': 'dd_bounds_wcd', 'label': 'Data-Dependent', 'color': 'blue', 'marker': 'o'},
        {'col': 'bootstrap_wcd', 'label': 'Bootstrap', 'color': 'orange', 'marker': 'x'},
        {'col': 'set_membership_wcd', 'label': 'Set Membership (QMI)', 'color': 'green', 'marker': 's'}
    ]
    plot_mc_metric_comparison(summary_df, wcd_configs, 'T', 'Mean WCD (log scale)', 
                              'Monte Carlo: Mean WCD Comparison', 
                              os.path.join(figures_dir, "mc_final_comparison_wcd.png"))

    # --- Plot configuration for Max Deviation in 'a' and 'b' ---
    max_dev_configs = [
        {'col': 'dd_bounds_max_dev_a', 'label': 'DD-Bounds (a)', 'color': 'blue', 'marker': 'o', 'linestyle': '-'},
        {'col': 'bootstrap_max_dev_a', 'label': 'Bootstrap (a)', 'color': 'orange', 'marker': 'x', 'linestyle': '-'},
        {'col': 'set_membership_max_dev_a', 'label': 'Set Membership (a)', 'color': 'green', 'marker': 's', 'linestyle': '-'},
        {'col': 'dd_bounds_max_dev_b', 'label': 'DD-Bounds (b)', 'color': 'blue', 'marker': 'o', 'linestyle': '--'},
        {'col': 'bootstrap_max_dev_b', 'label': 'Bootstrap (b)', 'color': 'orange', 'marker': 'x', 'linestyle': '--'},
        {'col': 'set_membership_max_dev_b', 'label': 'Set Membership (b)', 'color': 'green', 'marker': 's', 'linestyle': '--'}
    ]
    plot_mc_metric_comparison(summary_df, max_dev_configs, 'T', 'Mean Max Deviation (log scale)', 
                              'Monte Carlo: Mean Max Deviation for "a" and "b"', 
                              os.path.join(figures_dir, "mc_final_comparison_dev_a_and_b.png"))

    print("\n--- FINAL Monte Carlo Simulation Finished Successfully! ---")

# Guard to allow direct execution for testing purposes
if __name__ == '__main__':
    run_mc_trajectory_comparison()