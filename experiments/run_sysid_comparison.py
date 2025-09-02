"""
WORKER SCRIPT to compare three SysId related methods over a range of T:
1. Data-Dependent Bounds (Ellipse) on I.I.D. data.
2. Bootstrap on I.I.D. data.
3. Bootstrap on time-series (trajectory) data.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Import all required tools ---
from src.data_generation import generate_iid_samples, generate_time_series_data
from src.system_identification import estimate_least_squares_iid, estimate_least_squares_timeseries, perform_bootstrap_analysis, perform_bootstrap_analysis_iid
from src.analysis import ConfidenceRectangle, ConfidenceEllipse, calculate_p_matrix_for_confidence_ellipse

# --- Define project paths ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GENERATED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'generated')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')


def run_sysid_methods_comparison(data_seed: int) -> pd.DataFrame:
    """
    Orchestrates the comparison of the three methods for a single master data seed.
    """
    # --- Central Configuration ---
    T_MAX = 300
    T_RANGE = np.arange(20, T_MAX + 1, 20)
    
    TRUE_PARAMS = {'a': 0.5, 'b': 0.5}
    NOISE_STD_DEV_W = np.sqrt((0.01**2) / 3)
    INPUT_STD_DEV_U = 1.0
    CONFIDENCE_DELTA = 0.05
    BOOTSTRAP_ITERATIONS = 1000

    results_list = []
    for T in tqdm(T_RANGE, desc=f"Seed {data_seed} Progress", leave=False):
        current_seed = data_seed * 10000 + T
        metrics = {'T': T}

        # --- Generate I.I.D. data for the first two pipelines ---
        x_iid, u_iid, _, y_iid = generate_iid_samples(
            N=T, system_params=TRUE_PARAMS,
            params_config={'x_std_dev': 1.0, 'u_std_dev': INPUT_STD_DEV_U, 'w_std_dev': NOISE_STD_DEV_W},
            output_path=GENERATED_DATA_DIR, base_filename=f"temp_iid_T{T}", seed=current_seed
        )

        # --- Pipeline 1: Data-Dependent Bounds on I.I.D. Data ---
        try:
            A_est_dd, B_est_dd = estimate_least_squares_iid(x_iid, u_iid, y_iid)
            if A_est_dd is not None:
                p_matrix = calculate_p_matrix_for_confidence_ellipse(x_iid, u_iid, NOISE_STD_DEV_W, CONFIDENCE_DELTA)
                ellipse = ConfidenceEllipse(center=(A_est_dd.item(), B_est_dd.item()), p_matrix=p_matrix)
                devs = ellipse.axis_parallel_deviations()
                metrics['dd_bounds_area'] = ellipse.area(); metrics['dd_bounds_wcd'] = ellipse.worst_case_deviation()
                metrics['dd_bounds_max_dev_a'] = devs['max_dev_a']; metrics['dd_bounds_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: DD-Bounds failed for T={T}, Seed={data_seed} with error: {e}")

        # --- Pipeline 2: Bootstrap on I.I.D. Data ---
        try:
            A_est_bs_iid, B_est_bs_iid = estimate_least_squares_iid(x_iid, u_iid, y_iid)
            if A_est_bs_iid is not None:
                bootstrap_results = perform_bootstrap_analysis_iid(
                    initial_estimate=(A_est_bs_iid, B_est_bs_iid), N=T,
                    sigmas={'x': 1.0, 'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                    delta=CONFIDENCE_DELTA, seed=current_seed + 1
                )
                rect = ConfidenceRectangle(center=(A_est_bs_iid.item(), B_est_bs_iid.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
                devs = rect.axis_parallel_deviations()
                metrics['bootstrap_iid_area'] = rect.area(); metrics['bootstrap_iid_wcd'] = rect.worst_case_deviation()
                metrics['bootstrap_iid_max_dev_a'] = devs['max_dev_a']; metrics['bootstrap_iid_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: Bootstrap IID failed for T={T}, Seed={data_seed} with error: {e}")

        # --- Pipeline 3: Bootstrap on Time-Series Data ---
        try:
            state_ts_raw, input_ts_raw, _ = generate_time_series_data(
                system_params=TRUE_PARAMS, timesteps=T, 
                output_path=GENERATED_DATA_DIR, base_filename=f"temp_timeseries_T{T}",
                noise_config={'distribution': 'gaussian', 'std_dev': NOISE_STD_DEV_W}, seed=current_seed
            )
            state_ts, input_ts = np.array([state_ts_raw.flatten()]), np.array([input_ts_raw.flatten()])
            A_est_bs_ts, B_est_bs_ts = estimate_least_squares_timeseries(state_ts, input_ts)
            bootstrap_results = perform_bootstrap_analysis(
                initial_estimate=(A_est_bs_ts, B_est_bs_ts), data_shape=(1, T),
                sigmas={'u': INPUT_STD_DEV_U, 'w': NOISE_STD_DEV_W}, M=BOOTSTRAP_ITERATIONS,
                delta=CONFIDENCE_DELTA, seed=current_seed + 2
            )
            rect = ConfidenceRectangle(center=(A_est_bs_ts.item(), B_est_bs_ts.item()), epsilons=(bootstrap_results['epsilon_A'], bootstrap_results['epsilon_B']))
            devs = rect.axis_parallel_deviations()
            metrics['bootstrap_ts_area'] = rect.area(); metrics['bootstrap_ts_wcd'] = rect.worst_case_deviation()
            metrics['bootstrap_ts_max_dev_a'] = devs['max_dev_a']; metrics['bootstrap_ts_max_dev_b'] = devs['max_dev_b']
        except Exception as e:
            print(f"Warning: Bootstrap TS failed for T={T}, Seed={data_seed} with error: {e}")
            
        results_list.append(metrics)

    return pd.DataFrame(results_list)